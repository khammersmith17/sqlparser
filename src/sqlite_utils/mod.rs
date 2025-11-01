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

// offset shift, evaluated_varint
#[allow(unused)]
pub fn parse_sqlite_varint(buffer: &[u8], start_offset: usize) -> (usize, usize) {
    let mut offset = start_offset;
    let mut varint = 0_usize;

    while offset < (start_offset + 9) {
        if (offset - start_offset) == 8 {
            varint = (varint << 8) | (buffer[offset] as usize);
        } else {
            varint = (varint << 7) | ((buffer[offset] & 0x7F) as usize);
        }

        if buffer[offset] & SINGLE_BYTE_BIT_MAP == 0 {
            break;
        }

        offset += 1;
    }

    (offset - start_offset, varint)
}
