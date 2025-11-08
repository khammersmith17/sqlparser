use crate::data_containers::{record::TableRecordLabel, schema::IndexRowRef};
use std::cmp::Ordering;

#[allow(unused)]
fn binary_search_leaf_page_records(
    leaf_page_records: Vec<TableRecordLabel>,
    row_ids: &[IndexRowRef],
) -> Vec<TableRecordLabel> {
    // iterate through row ids
    // binary search for each target row id
    let mut filtered_record_labels: Vec<TableRecordLabel> = Vec::with_capacity(row_ids.len());
    for row_id in row_ids.iter() {
        let mut left = 0_usize;
        let mut right = leaf_page_records.len() - 1;

        while left <= right {
            let m: usize = (left + right) / 2;

            match row_id.partial_cmp(&leaf_page_records[m].row_id) {
                Some(order) => match order {
                    Ordering::Equal => {
                        filtered_record_labels.push(leaf_page_records[m].clone());
                        break;
                    }
                    Ordering::Greater => {
                        left = m + 1;
                    }
                    Ordering::Less => {
                        right = m - 1;
                    }
                },
                None => {
                    panic!("Attempted to compare a rowid with a composite primary key IndexRowRef")
                }
            }
        }
    }
    filtered_record_labels
}

#[allow(unused)]
fn merge_join_leaf_page_records(
    leaf_page_records: Vec<TableRecordLabel>,
    row_ids: &[IndexRowRef],
) -> Vec<TableRecordLabel> {
    // use 2 pointers to find all table record labels for the row ids
    let mut row_id_ptr = 0_usize;
    let mut leaf_page_ptr = 0_usize;

    let len_row_ids = row_ids.len();
    let len_page = leaf_page_records.len();

    let mut filtered_record_labels: Vec<TableRecordLabel> = Vec::with_capacity(row_ids.len());

    while row_id_ptr < len_row_ids && leaf_page_ptr < len_page {
        if row_ids[row_id_ptr].eq(&leaf_page_records[leaf_page_ptr].row_id) {
            filtered_record_labels.push(leaf_page_records[leaf_page_ptr].clone());
            row_id_ptr += 1;
        }

        leaf_page_ptr += 1;
    }

    filtered_record_labels
}

#[allow(unused)]
pub fn filter_table_records(
    leaf_page_records: Vec<TableRecordLabel>,
    row_ids: &[IndexRowRef],
) -> Vec<TableRecordLabel> {
    // dispatch to the right method based on the number of row ids
    // binary search when the number of row ids on the page is > 15%
    let row_id_coverage = (row_ids.len() as f32) / (leaf_page_records.len() as f32);

    if row_id_coverage > 0.15_32 {
        merge_join_leaf_page_records(leaf_page_records, row_ids)
    } else {
        binary_search_leaf_page_records(leaf_page_records, row_ids)
    }
}
