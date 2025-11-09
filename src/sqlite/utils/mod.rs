use anyhow::Result;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

const SINGLE_BYTE_BIT_MAP: u8 = 0x80;

pub fn construct_base_column_bit_set(columns: &[&str]) -> HashMap<String, u64> {
    columns
        .iter()
        .enumerate()
        .map(|(i, c)| (c.to_uppercase(), 1_u64 << i))
        .collect()
}

pub fn generate_column_bitmask(col_map: &HashMap<String, u64>, columns: &[String]) -> u64 {
    // base table column map
    // and the set of columns related to some specific
    // COLUMNS ARE CANONOCALIZED AT THIS POINT
    columns
        .iter()
        .map(|c| col_map.get(c).copied().unwrap_or(0))
        .fold(0, |acc, bit| acc | bit)
}

pub fn col_set_covers(required: u64, available: u64) -> bool {
    // flip all bits in the available set
    // check bitwise AND to see if any required columns are not available
    (required & !available) == 0
}

pub fn read_varint_from_file(file: &mut File) -> Result<SqliteVarint> {
    let mut varint = 0_usize;
    let mut byte_size = 1_usize;
    for i in 0..9 {
        let mut buf = [0_u8; 1];
        file.read_exact(&mut buf)?;

        if i < 8 {
            varint = (varint << 7) | ((buf[0] & 0x7F) as usize);
            if buf[0] & SINGLE_BYTE_BIT_MAP == 0 {
                break;
            }
        } else {
            varint = (varint << 8) | (buf[0] as usize);
            break;
        }
        byte_size += 1;
    }

    Ok(SqliteVarint { byte_size, varint })
}

pub fn read_varint(file: &mut File) -> Result<SqliteVarint> {
    let mut offset = 1_usize;
    let mut byte_buff: Vec<u8> = Vec::new();
    let mut read_buff: [u8; 1] = [0_u8];
    file.read_exact(&mut read_buff)?;

    byte_buff.push(read_buff[0]);
    while read_buff[0] & SINGLE_BYTE_BIT_MAP != 0 {
        offset += 1;

        file.read_exact(&mut read_buff)?;

        byte_buff.push(read_buff[0]);
    }

    let varint_val = parse_varint(byte_buff);

    Ok(SqliteVarint {
        byte_size: offset,
        varint: varint_val,
    })
}

pub fn evaluate_varint(val: usize) -> usize {
    // this really should be only for size > 13
    let odd = val % 2 == 0;
    if odd && val >= 13 {
        return (val - 13) / 2;
    }

    if !odd && val >= 12 {
        return (val - 12) / 2;
    }

    return val;
}

pub fn parse_varint(bytes: Vec<u8>) -> usize {
    let mut result = 0_usize;
    for (i, b) in bytes.into_iter().enumerate() {
        if i < 8 {
            result = (result << 7) | ((b & 0x7F) as usize);
        } else {
            result = (result << 8) | (b as usize);
        }
    }

    result
}

pub fn traverse_record(record: &[u8], mut record_i: usize) -> Vec<u8> {
    let mut buffer: Vec<u8> = Vec::with_capacity(9_usize);
    let mut f = record[record_i];
    buffer.push(f);
    let mut res = f & SINGLE_BYTE_BIT_MAP;

    record_i += 1;
    while res != 0 && buffer.len() < 9 {
        f = record[record_i];
        res = f & SINGLE_BYTE_BIT_MAP;
        buffer.push(f);
        record_i += 1;
    }
    buffer
}

// type to represent a decoded varint and identify how many bytes where read
pub struct SqliteVarint {
    pub byte_size: usize,
    pub varint: usize,
}

// offset shift, evaluated_varint
pub fn parse_sqlite_varint(buffer: &[u8], start_offset: usize) -> SqliteVarint {
    let mut offset = start_offset;
    let mut varint = 0_usize;
    let max_varint_offset = start_offset + 9;

    while offset < max_varint_offset {
        let curr_byte = buffer[offset];
        if (offset - start_offset) == 8 {
            varint = (varint << 8) | (curr_byte as usize);
            break;
        }
        varint = (varint << 7) | ((curr_byte & 0x7F) as usize);
        offset += 1;
        if curr_byte & SINGLE_BYTE_BIT_MAP == 0 {
            break;
        }
    }

    SqliteVarint {
        byte_size: offset - start_offset,
        varint,
    }
}
