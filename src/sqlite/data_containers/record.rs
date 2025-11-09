use super::serial_types::SerialType;
use crate::sqlite::data_containers::schema::{IndexMetadata, IndexRowRef};
use crate::sqlite::utils::{SqliteVarint, parse_sqlite_varint};
use anyhow::{Result, bail};
use std::cmp::Ordering;
use std::collections::HashMap;

pub trait SQLiteRecord {
    fn record_values(&self) -> &HashMap<String, Value>;
}

#[derive(Debug, Clone)]
pub struct TableRecordLabel {
    pub row_id: IndexRowRef,
    pub page_offset: u16,
}

#[derive(Debug)]
pub struct InteriorTablePageRecord {
    pub child_page_num: u64,
    pub row_id: u64,
}

impl InteriorTablePageRecord {
    pub fn new(data_buffer: &[u8], offset: usize) -> InteriorTablePageRecord {
        let child_page_num = u32::from_be_bytes([
            data_buffer[offset],
            data_buffer[offset + 1],
            data_buffer[offset + 2],
            data_buffer[offset + 3],
        ]) as u64;
        let SqliteVarint { varint: row_id, .. } = parse_sqlite_varint(data_buffer, offset + 4);
        InteriorTablePageRecord {
            child_page_num,
            row_id: row_id as u64,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Null,
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self {
            Self::Null => None,
            Self::Int(my_val) => match other {
                Self::Int(other_val) => my_val.partial_cmp(other_val),
                _ => None,
            },
            Self::Float(my_val) => match other {
                Self::Float(other_val) => my_val.partial_cmp(other_val),
                _ => None,
            },
            Self::String(my_val) => match other {
                Self::String(other_val) => my_val.partial_cmp(other_val),
                _ => None,
            },
            Self::Bool(my_val) => match other {
                Self::Bool(other_val) => my_val.partial_cmp(other_val),
                _ => None,
            },
        }
    }
}

impl Value {
    fn from_serial_type(
        buffer: &[u8],
        offset: usize,
        serial_type: SerialType,
    ) -> Result<(usize, Value)> {
        let offset_and_value = match serial_type {
            SerialType::Null => (0_usize, Value::Null),
            SerialType::Int8 => {
                let v = Value::Int(u8::from_be_bytes([buffer[offset]]) as i64);
                (1_usize, v)
            }
            SerialType::Int16 => {
                let v = Value::Int(u16::from_be_bytes([buffer[offset], buffer[offset + 1]]) as i64);
                (2_usize, v)
            }
            SerialType::Int24 => {
                let v = Value::Int(u32::from_be_bytes([
                    0_u8,
                    buffer[offset],
                    buffer[offset + 1],
                    buffer[offset + 2],
                ]) as i64);
                (3_usize, v)
            }
            SerialType::Int32 => {
                let v = Value::Int(u32::from_be_bytes([
                    buffer[offset],
                    buffer[offset + 1],
                    buffer[offset + 2],
                    buffer[offset + 3],
                ]) as i64);
                (4_usize, v)
            }
            SerialType::Int48 => {
                let v = Value::Int(u64::from_be_bytes([
                    0_u8,
                    0_u8,
                    buffer[offset],
                    buffer[offset + 1],
                    buffer[offset + 2],
                    buffer[offset + 3],
                    buffer[offset + 4],
                    buffer[offset + 5],
                ]) as i64);
                (6_usize, v)
            }
            SerialType::Int64 => {
                let v = Value::Int(u64::from_be_bytes([
                    buffer[offset],
                    buffer[offset + 1],
                    buffer[offset + 2],
                    buffer[offset + 3],
                    buffer[offset + 4],
                    buffer[offset + 5],
                    buffer[offset + 6],
                    buffer[offset + 7],
                ]) as i64);
                (8_usize, v)
            }
            SerialType::Float64 => {
                let v = Value::Float(f64::from_be_bytes([
                    buffer[offset],
                    buffer[offset + 1],
                    buffer[offset + 2],
                    buffer[offset + 3],
                    buffer[offset + 4],
                    buffer[offset + 5],
                    buffer[offset + 6],
                    buffer[offset + 7],
                ]));
                (8_usize, v)
            }
            SerialType::BoolFalse => (0_usize, Value::Bool(false)),
            SerialType::BoolTrue => (0_usize, Value::Bool(true)),
            SerialType::Blob(_v) => {
                todo!()
            }
            SerialType::String(size) => {
                let string_buffer = buffer[offset..offset + size as usize].to_vec();
                (
                    size as usize,
                    Value::String(
                        String::from_utf8(string_buffer)
                            .expect("unable to read string serial type data"),
                    ),
                )
            }
        };

        Ok(offset_and_value)
    }
}

#[derive(Debug)]
pub struct IndexRecord {
    pub record_values: HashMap<String, Value>,
}

impl SQLiteRecord for IndexRecord {
    fn record_values(&self) -> &HashMap<String, Value> {
        &self.record_values
    }
}

impl IndexRecord {
    pub fn new(
        payload_size: usize,
        payload_buffer: &[u8],
        index_meta: &IndexMetadata,
    ) -> Result<IndexRecord> {
        let pk_len = index_meta.table_primary_key.len();
        let SqliteVarint {
            byte_size: header_varint_len,
            varint: header_end,
        } = parse_sqlite_varint(payload_buffer, 0_usize);
        let mut payload_offset = header_varint_len;
        let index_columns = index_meta.available_columns();
        let mut record_values: HashMap<String, Value> = HashMap::with_capacity(index_columns.len());
        let mut serial_types: Vec<SerialType> = Vec::with_capacity(index_columns.len());

        while payload_offset < header_end {
            let SqliteVarint {
                byte_size: serial_type_offset_len,
                varint: serial_type_value,
            } = parse_sqlite_varint(&payload_buffer, payload_offset);

            payload_offset += serial_type_offset_len;
            serial_types.push(SerialType::new(serial_type_value)?);
        }

        let mut value_offset = header_end;
        // row id value comes last in the index record
        // decode index_columns
        let mut serial_type_iter = serial_types.into_iter();
        for i in pk_len..index_columns.len() {
            let Some(stype) = serial_type_iter.next() else {
                bail!("Parsed serail types do not match column size")
            };
            let (offset_delta, value) =
                Value::from_serial_type(payload_buffer, value_offset, stype)?;

            value_offset += offset_delta;
            record_values.insert(index_columns[i].to_uppercase(), value);
        }

        for i in 0..pk_len {
            let Some(stype) = serial_type_iter.next() else {
                bail!("Parsed serail types do not match column size")
            };
            let (offset_delta, value) =
                Value::from_serial_type(payload_buffer, value_offset, stype)?;

            value_offset += offset_delta;
            record_values.insert(index_columns[i].to_uppercase(), value);
        }

        debug_assert!(payload_offset <= payload_size);

        Ok(IndexRecord { record_values })
    }

    pub fn fetch_row_id(&self, row_id_col_name: &str) -> Option<&Value> {
        self.record_values.get(row_id_col_name)
    }
}

#[derive(Debug)]
pub struct TableRecord {
    #[allow(unused)]
    pub record_id: i64,
    pub row_values: HashMap<String, Value>,
}

impl SQLiteRecord for TableRecord {
    fn record_values(&self) -> &HashMap<String, Value> {
        &self.row_values
    }
}

impl TableRecord {
    pub fn new(
        header_buffer: &[u8],
        record_buffer: &[u8],
        header_size: usize,
        record_id: i64,
        col_names: &[&str],
    ) -> Result<TableRecord> {
        let mut record_i = 0_usize;

        let record_header_end = record_i + header_size as usize;

        let n_cols = col_names.len();

        let mut serial_types: Vec<SerialType> = Vec::with_capacity(n_cols);
        while record_i < record_header_end {
            let SqliteVarint {
                varint: value_size,
                byte_size,
            } = parse_sqlite_varint(&header_buffer, record_i);
            record_i += byte_size;

            serial_types.push(SerialType::new(value_size).expect("Invalid serial type"));
        }

        record_i = 0;
        let mut row_values: HashMap<String, Value> = HashMap::new();

        for (stype, col_name) in serial_types.into_iter().zip(col_names) {
            if *col_name == "ID" {
                row_values.insert(col_name.to_uppercase(), Value::Int(record_id.into()));
                continue;
            };

            let (offset_shift, record_v) = Value::from_serial_type(record_buffer, record_i, stype)?;
            record_i += offset_shift;

            row_values.insert(col_name.to_string(), record_v);
        }
        Ok(TableRecord {
            record_id,
            row_values,
        })
    }
}
