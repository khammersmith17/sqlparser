use super::headers::DatabaseHeader;
use crate::parser::FilterFn;
use crate::sqlite::data_containers::record::{IndexRecord, TableRecord, Value};
use crate::sqlite::data_containers::schema::{IndexMetadata, IndexRowRef, TableMetadata};
use crate::sqlite::utils::{SqliteVarint, parse_sqlite_varint};
use anyhow::{Result, bail};
use std::fs::File;
use std::io::{Read, Seek};
// container for the page metadata
pub struct Page {
    #[allow(dead_code)]
    pub offset: u64,
    #[allow(dead_code)]
    pub page_header: PageHeader,
    #[allow(dead_code)]
    pub pointer_array: Vec<u16>,
    #[allow(dead_code)]
    pub data: Vec<u8>,
    #[allow(dead_code)]
    pub page_start: u64,
}

impl Page {
    pub fn new(file: &mut File, page_label: u64, database_header: &DatabaseHeader) -> Result<Page> {
        // a buffer for the entire page
        let mut data = vec![0_u8; database_header.page_size.into()];

        // finding the start of the page
        let page_start = (page_label - 1) * database_header.page_size as u64;

        // moving the file descriptor to the start of the page
        file.seek(std::io::SeekFrom::Start(page_start))?;

        // if this is the first page, we need to seek past the database header
        let db_header_start = if page_label == 1 { 100 } else { 0 };

        file.read_exact(&mut data)?;
        let page_header = PageHeader::new(&data[db_header_start..db_header_start + 12])?;
        let mut pointer_array: Vec<u16> = Vec::with_capacity(page_header.num_cells.into());

        let page_header_len = page_header.len();
        for i in 0..page_header.num_cells {
            let offset = db_header_start + page_header_len as usize + (2_usize * i as usize);
            pointer_array.push(u16::from_be_bytes([data[offset], data[offset + 1]]));
        }
        Ok(Page {
            offset: page_label,
            page_header,
            pointer_array,
            data,
            page_start,
        })
    }

    pub fn page_type(&self) -> PageType {
        self.page_header.page_type
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PageType {
    InteriorIndexBTreePage,
    InteriorTableBTreePage,
    LeafIndexBTreePage,
    LeafTableBTreePage,
}

impl PageType {
    fn read_page_type(value: u8) -> Option<PageType> {
        match value {
            2_u8 => Some(PageType::InteriorIndexBTreePage),
            5_u8 => Some(PageType::InteriorTableBTreePage),
            10_u8 => Some(PageType::LeafIndexBTreePage),
            13_u8 => Some(PageType::LeafTableBTreePage),
            _ => None,
        }
    }
}

// container for the page header
#[derive(Debug)]
pub struct PageHeader {
    #[allow(dead_code)]
    pub page_type: PageType,
    #[allow(dead_code)]
    first_freeblock: u16,
    #[allow(dead_code)]
    pub num_cells: u16,
    #[allow(dead_code)]
    cell_content_start: u16,
    #[allow(dead_code)]
    free_bytes: u8,
    #[allow(dead_code)]
    pub right_pointer: Option<u32>,
}

impl PageHeader {
    pub fn new(header_buffer: &[u8]) -> Result<PageHeader> {
        let Some(page_type) = PageType::read_page_type(header_buffer[0]) else {
            bail!("Invalid page type")
        };
        let first_freeblock = u16::from_be_bytes([header_buffer[1], header_buffer[2]]);
        let num_cells = u16::from_be_bytes([header_buffer[3], header_buffer[4]]);
        let cell_content_start = u16::from_be_bytes([header_buffer[5], header_buffer[6]]);
        let free_bytes = header_buffer[7];
        let right_pointer = match page_type {
            PageType::InteriorTableBTreePage | PageType::InteriorIndexBTreePage => {
                Some(u32::from_be_bytes([
                    header_buffer[8],
                    header_buffer[9],
                    header_buffer[10],
                    header_buffer[11],
                ]))
            }
            _ => None,
        };
        Ok(PageHeader {
            page_type,
            first_freeblock,
            num_cells,
            cell_content_start,
            free_bytes,
            right_pointer,
        })
    }

    pub fn len(&self) -> u32 {
        match self.right_pointer {
            Some(_) => 12,
            None => 8,
        }
    }
}

pub fn scan_leaf_table_page(
    page: Page,
    cond_evaluator: FilterFn,
    t_meta: &TableMetadata,
) -> Result<Vec<TableRecord>> {
    let mut records: Vec<TableRecord> = Vec::new();

    let col_names = t_meta.table_schema.column_names();

    for p in &page.pointer_array {
        let mut offset = *p as usize;
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
            &col_names,
        )?;

        if (*cond_evaluator)(&record) {
            records.push(record);
        }
    }
    Ok(records)
}

pub fn scan_interior_table_page(
    file: &mut File,
    page: Page,
    cond_evaluator: FilterFn,
    t_meta: &TableMetadata,
    db_header: &DatabaseHeader,
) -> Result<Vec<TableRecord>> {
    // traverse the children
    let mut records: Vec<TableRecord> = Vec::new();

    for child_ptr in page.pointer_array {
        let offset_buffer = &page.data[child_ptr as usize..child_ptr as usize + 4];
        let child_page_offset = u32::from_be_bytes([
            offset_buffer[0],
            offset_buffer[1],
            offset_buffer[2],
            offset_buffer[3],
        ]);

        let child_page = Page::new(file, child_page_offset.into(), db_header)?;
        let child_records = if matches!(
            child_page.page_type(),
            PageType::LeafTableBTreePage | PageType::LeafIndexBTreePage
        ) {
            scan_leaf_table_page(child_page, cond_evaluator.clone(), t_meta)?
        } else {
            scan_interior_table_page(file, child_page, cond_evaluator.clone(), t_meta, db_header)?
        };

        records.extend(child_records);
    }
    Ok(records)
}

pub fn read_index_interior_page<'a>(
    page: Page,
    file: &mut File,
    cond_evaluator: FilterFn,
    index_meta: &IndexMetadata,
    db_header: &DatabaseHeader,
) -> Result<Vec<IndexRowRef>> {
    let mut records: Vec<IndexRowRef> = Vec::new();

    for ptr_offset in page.pointer_array {
        let block_start = ptr_offset as usize;
        let child_page_offset = u32::from_be_bytes([
            page.data[block_start],
            page.data[block_start + 1_usize],
            page.data[block_start + 2_usize],
            page.data[block_start + 3_usize],
        ]);

        let child_page = Page::new(file, child_page_offset.into(), db_header)?;

        let child_page_records = if matches!(
            child_page.page_header.page_type,
            PageType::LeafIndexBTreePage
        ) {
            read_index_leaf_page(child_page, index_meta, cond_evaluator.clone())?
        } else {
            read_index_interior_page(
                child_page,
                file,
                cond_evaluator.clone(),
                index_meta,
                db_header,
            )?
        };

        records.extend(child_page_records)
    }
    Ok(records)
}

pub fn read_index_leaf_page(
    page: Page,
    index_meta: &IndexMetadata,
    cond_eval: FilterFn,
) -> Result<Vec<IndexRowRef>> {
    let Page { pointer_array, .. } = page;

    let mut index_rows: Vec<IndexRowRef> = Vec::new();
    let Some(row_id_col_name) = index_meta.row_id_column_name() else {
        bail!("No primary key on index")
    };

    for p in pointer_array.iter() {
        // seek to the page start
        let cell_offset = *p as usize;

        let SqliteVarint {
            varint: payload_size,
            byte_size: offset,
        } = parse_sqlite_varint(&page.data, cell_offset);
        let payload_start = cell_offset + offset;
        let payload_end = payload_start + payload_size;

        if payload_end > page.data.len() {
            bail!("Malformed page");
        }

        // header size varint is included in the header size
        let record_buffer = &page.data[payload_start..payload_end];

        let record = IndexRecord::new(payload_size, record_buffer, index_meta)?;
        if cond_eval(&record) {
            let row_id_value = record.fetch_row_id(row_id_col_name);
            let index_row = match row_id_value {
                Some(value) => match value {
                    Value::Int(v) => IndexRowRef::RowId(*v),
                    _ => todo!(),
                },
                None => bail!("Invalid row id column name"),
            };
            index_rows.push(index_row)
        }
    }
    Ok(index_rows)
}
