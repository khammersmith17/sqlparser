use crate::{
    data_containers::{
        page::{
            read_index_interior_page, read_index_leaf_page, scan_interior_table_page,
            scan_leaf_table_page, PageType,
        },
        record::{InteriorTablePageRecord, TableRecordLabel},
        schema::{IndexMetadata, IndexRowRef},
    },
    generate_condition_evaluator,
    sql::{Condition, FilterFn},
    utils::{parse_sqlite_varint, SqliteVarint},
    BasicSelectStatement, DatabaseHeader, Page, SqliteSchema, TableRecord,
};
use anyhow::{bail, Result};
use std::cmp::Ordering;
use std::fs::File;
mod search;
use search::filter_table_records;

fn plan_query<'a>(
    query_condition: &Option<Condition>,
    db_schema: &'a SqliteSchema,
    table_name: &str,
) -> Option<&'a IndexMetadata> {
    let Some(ref cond) = query_condition else {
        return None;
    };

    let condition_columns = cond.evaluated_columns();
    db_schema.resolve_query_metadata(table_name, &condition_columns)
}

pub fn process_query(
    file: &mut File,
    db_schema: &SqliteSchema,
    db_header: &DatabaseHeader,
    query: BasicSelectStatement,
) -> Result<Vec<TableRecord>> {
    // this should return each query result record
    let BasicSelectStatement {
        table,
        condition: query_condition,
        ..
    } = query;

    /*
     * 1. Determine columns used in the condition
     * 2. If there are none, resort to full table scan
     * 3. Evaluate whether an index can service the query
     * 3.1 If so, traverse the index pages to get rowids
     * 3.2 otherwise full table scan
     * */

    let index_opt = plan_query(&query_condition, db_schema, table.as_str());

    let cond_eval = generate_condition_evaluator(query_condition);
    let Some(table_meta) = db_schema.fetch_table(&table) else {
        bail!("Invalid query")
    };

    let records = if let Some(index) = index_opt {
        let index_root_page = Page::new(file, index.root_page as u64, &db_header)?;
        let mut row_ids = read_index(index_root_page, file, index, cond_eval, db_header)?;
        let table_column_names = table_meta.table_column_names();

        row_ids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        traverse_b_tree_for_row_ids(
            &row_ids,
            table_meta.has_row_id(),
            file,
            &table_column_names,
            table_meta.root_page as u64,
            db_header,
        )?
    } else {
        let page = Page::new(file, table_meta.root_page as u64, db_header)?;
        if matches!(page.page_type(), PageType::LeafTableBTreePage) {
            scan_leaf_table_page(page, cond_eval, table_meta)?
        } else {
            scan_interior_table_page(file, page, cond_eval, &table_meta, db_header)?
        }
    };
    Ok(records)
}

fn fetch_leaf_table_page_row_ids(
    file: &mut File,
    with_rowid: bool,
    page_num: u64,
    db_header: &DatabaseHeader,
) -> Result<Vec<TableRecordLabel>> {
    let leaf_page = Page::new(file, page_num, db_header)?;
    let mut row_ids: Vec<TableRecordLabel> = Vec::with_capacity(leaf_page.pointer_array.len());

    for ptr in &leaf_page.pointer_array {
        let page_offset = *ptr;
        // record size
        let SqliteVarint {
            byte_size: record_size_offset,
            ..
        } = parse_sqlite_varint(&leaf_page.data, page_offset.into());
        if with_rowid {
            // record id
            let SqliteVarint {
                varint: record_id, ..
            } = parse_sqlite_varint(&leaf_page.data, record_size_offset + page_offset as usize);
            row_ids.push(TableRecordLabel {
                page_offset,
                row_id: IndexRowRef::RowId(record_id as i64),
            });
        } else {
            todo!("Implement a non row id fetch")
        }
    }

    Ok(row_ids)
}

fn traverse_b_tree_for_row_ids(
    row_ids: &[IndexRowRef],
    with_rowid: bool,
    file: &mut File,
    table_column_names: &[&str],
    page_num: u64,
    db_header: &DatabaseHeader,
) -> Result<Vec<TableRecord>> {
    let page = Page::new(file, page_num, db_header)?;
    let mut records: Vec<TableRecord> = Vec::new();

    if page.page_header.page_type == PageType::InteriorTableBTreePage {
        let mut interior_page_records: Vec<InteriorTablePageRecord> =
            Vec::with_capacity(page.pointer_array.len());
        for ptr in page.pointer_array.iter() {
            interior_page_records.push(InteriorTablePageRecord::new(&page.data, *ptr as usize));
        }

        let mut row_id_ptr = 0_usize;
        let num_index_rows = row_ids.len();

        // iterate through each page
        // see what row ids fall in the range of the
        for page_record in &interior_page_records {
            let mut curr_bucket: Vec<i64> = Vec::new();
            let right = page_record.row_id;
            let range_start = row_id_ptr;

            while row_id_ptr < num_index_rows {
                match row_ids[row_id_ptr] {
                    IndexRowRef::RowId(rid) => {
                        if rid as u64 > right {
                            break;
                        }
                        curr_bucket.push(rid);
                        row_id_ptr += 1;
                    }
                    IndexRowRef::PrimaryKeyTuple(_) => {
                        todo!("Implement row search for primary key tuple")
                    }
                }
            }

            if row_id_ptr > range_start {
                records.extend(traverse_b_tree_for_row_ids(
                    &row_ids[range_start..row_id_ptr],
                    with_rowid,
                    file,
                    table_column_names,
                    page_record.child_page_num,
                    db_header,
                )?);
            }
        }

        if row_id_ptr < num_index_rows {
            records.extend(traverse_b_tree_for_row_ids(
                &row_ids[row_id_ptr..],
                with_rowid,
                file,
                table_column_names,
                page.page_header
                    .right_pointer
                    .expect("Interior BTree page does not have right pointer populated")
                    as u64,
                db_header,
            )?)
        }
    } else {
        let leaf_page_row_ids =
            fetch_leaf_table_page_row_ids(file, with_rowid, page.offset, db_header)?;

        let filtered_records = filter_table_records(leaf_page_row_ids, row_ids);

        records.extend(fetch_filtered_records(
            filtered_records,
            &page,
            table_column_names,
        )?);
    }

    Ok(records)
}

fn read_index<'a>(
    index_root_page: Page,
    file: &mut File,
    index_meta: &IndexMetadata,
    cond: FilterFn,
    db_header: &DatabaseHeader,
) -> Result<Vec<IndexRowRef>> {
    let record_row_ids = if matches!(
        index_root_page.page_header.page_type,
        PageType::LeafIndexBTreePage
    ) {
        read_index_leaf_page(index_root_page, index_meta, cond)?
    } else {
        read_index_interior_page(index_root_page, file, cond, index_meta, db_header)?
    };

    Ok(record_row_ids)
}

fn fetch_filtered_records(
    filtered_records: Vec<TableRecordLabel>,
    page: &Page,
    table_column_names: &[&str],
) -> Result<Vec<TableRecord>> {
    let mut table_records: Vec<TableRecord> = Vec::with_capacity(filtered_records.len());

    for record in filtered_records.into_iter() {
        let TableRecordLabel { page_offset, .. } = record;

        let mut offset: usize = page_offset.into();

        // PAYLOAD HEADER REGION
        let SqliteVarint {
            varint: payload_size,
            byte_size: offset_shift,
        } = parse_sqlite_varint(&page.data, offset);
        offset += offset_shift;

        let SqliteVarint {
            varint: record_id,
            byte_size: offset_shift,
        } = parse_sqlite_varint(&page.data, offset);

        offset += offset_shift;
        let payload_end = offset + payload_size;

        // PAYLOAD REGION
        // walk the header to get the header size

        let SqliteVarint {
            varint: record_header_size,
            byte_size: header_size_offset,
        } = parse_sqlite_varint(&page.data, offset);
        let header_end = offset + record_header_size;
        offset += header_size_offset;

        if (header_end) > page.data.len() {
            bail!("Malformed page")
        }

        let header = &page.data[offset..header_end];

        if (payload_end) > page.data.len() {
            bail!("Malformed page")
        }
        let record = &page.data[header_end..payload_end];

        let record = TableRecord::new(
            header,
            record,
            record_header_size - header_size_offset,
            record_id as i64,
            table_column_names,
        )?;

        table_records.push(record);
    }
    Ok(table_records)
}
