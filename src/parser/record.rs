use super::schema::TableSchema;
use super::serial_types::SerialType;
use crate::utils::{parse_varint, traverse_record};
use anyhow::Result;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Null,
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

#[derive(Debug)]
pub struct Record {
    #[allow(unused)]
    pub record_id: i32,
    pub row_values: HashMap<String, Value>,
}

impl Record {
    pub fn new(
        header_buffer: &[u8],
        record_buffer: &[u8],
        header_size: usize,
        _record_size: usize,
        record_id: i32,
        table_schema: &TableSchema,
    ) -> Result<Record> {
        // start
        let mut record_i = 0_usize;

        let record_header_end = record_i + header_size as usize;

        let n_cols = table_schema.num_columns();

        let mut value_sizes: Vec<SerialType> = Vec::with_capacity(n_cols);
        while record_i < record_header_end {
            let value_size_buffer = traverse_record(&header_buffer, record_i);

            record_i += value_size_buffer.len();
            let value_size = parse_varint(value_size_buffer);
            value_sizes.push(SerialType::new(value_size).expect("Invalid serial type"));
        }

        record_i = 0;
        let mut row_values: HashMap<String, Value> = HashMap::new();

        for (value, col_name) in value_sizes.into_iter().zip(table_schema.columns()) {
            let record_v = match value {
                SerialType::Null => Value::Null,
                SerialType::Int8 => {
                    let v = Value::Int(u8::from_be_bytes([record_buffer[record_i]]) as i64);
                    record_i += 1;
                    v
                }
                SerialType::Int16 => {
                    let v = Value::Int(u16::from_be_bytes([
                        record_buffer[record_i],
                        record_buffer[record_i + 1],
                    ]) as i64);
                    record_i += 2;
                    v
                }
                SerialType::Int24 => {
                    let v = Value::Int(u32::from_be_bytes([
                        0_u8,
                        record_buffer[record_i],
                        record_buffer[record_i + 1],
                        record_buffer[record_i + 2],
                    ]) as i64);
                    record_i += 3;
                    v
                }
                SerialType::Int32 => {
                    let v = Value::Int(u32::from_be_bytes([
                        record_buffer[record_i],
                        record_buffer[record_i + 1],
                        record_buffer[record_i + 2],
                        record_buffer[record_i + 3],
                    ]) as i64);
                    record_i += 4;
                    v
                }
                SerialType::Int48 => {
                    let v = Value::Int(u64::from_be_bytes([
                        0_u8,
                        0_u8,
                        record_buffer[record_i],
                        record_buffer[record_i + 1],
                        record_buffer[record_i + 2],
                        record_buffer[record_i + 3],
                        record_buffer[record_i + 4],
                        record_buffer[record_i + 5],
                    ]) as i64);
                    record_i += 6;
                    v
                }
                SerialType::Int64 => {
                    let v = Value::Int(u64::from_be_bytes([
                        record_buffer[record_i],
                        record_buffer[record_i + 1],
                        record_buffer[record_i + 2],
                        record_buffer[record_i + 3],
                        record_buffer[record_i + 4],
                        record_buffer[record_i + 5],
                        record_buffer[record_i + 6],
                        record_buffer[record_i + 7],
                    ]) as i64);
                    record_i += 8;
                    v
                }
                SerialType::Float64 => {
                    let v = Value::Float(f64::from_be_bytes([
                        record_buffer[record_i],
                        record_buffer[record_i + 1],
                        record_buffer[record_i + 2],
                        record_buffer[record_i + 3],
                        record_buffer[record_i + 4],
                        record_buffer[record_i + 5],
                        record_buffer[record_i + 6],
                        record_buffer[record_i + 7],
                    ]));
                    record_i += 8;
                    v
                }
                SerialType::BoolFalse => Value::Bool(false),
                SerialType::BoolTrue => Value::Bool(true),
                SerialType::Blob(_v) => {
                    todo!()
                }
                SerialType::String(size) => {
                    let string_buffer = record_buffer[record_i..record_i + size as usize].to_vec();
                    record_i += size as usize;
                    Value::String(
                        String::from_utf8(string_buffer)
                            .expect("unable to read string serial type data"),
                    )
                }
            };
            row_values.insert(col_name.clone(), record_v);
        }
        Ok(Record {
            record_id,
            row_values,
        })
    }
}
