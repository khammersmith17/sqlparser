use anyhow::{bail, Result};

#[derive(Debug)]
pub enum DbTextEncoding {
    Utf8,
    Utf16le,
    Utf16be,
}

impl TryFrom<u32> for DbTextEncoding {
    type Error = String;
    fn try_from(decoded_value: u32) -> Result<DbTextEncoding, String> {
        match decoded_value {
            1_u32 => Ok(DbTextEncoding::Utf8),
            2_u32 => Ok(DbTextEncoding::Utf16le),
            3_u32 => Ok(DbTextEncoding::Utf16be),
            _ => Err("Invalid encoding".into()),
        }
    }
}

// container for all the relevant database header attributes
#[derive(Debug)]
pub struct DatabaseHeader {
    #[allow(dead_code)]
    pub page_size: u16,
    #[allow(dead_code)]
    file_format_write_version: u8,
    #[allow(dead_code)]
    file_format_read_version: u8,
    #[allow(dead_code)]
    reserved_space: u8,
    #[allow(dead_code)]
    max_payload_fraction: u8,
    #[allow(dead_code)]
    min_payload_fraction: u8,
    #[allow(dead_code)]
    leaf_payload_fraction: u8,
    #[allow(dead_code)]
    file_change_counter: u32,
    #[allow(dead_code)]
    size_of_file: u32,
    #[allow(dead_code)]
    freelist_trunk_page_number: u32,
    #[allow(dead_code)]
    num_freelist_pages: u32,
    #[allow(dead_code)]
    schema_cookie: u32,
    #[allow(dead_code)]
    schema_format_number: u32,
    #[allow(dead_code)]
    default_page_cache_size: u32,
    #[allow(dead_code)]
    largest_root_b_tree_number: u32,
    #[allow(dead_code)]
    text_encoding: DbTextEncoding,
    #[allow(dead_code)]
    user_version: u32,
    #[allow(dead_code)]
    incremental_vaccum_mode: bool,
    #[allow(dead_code)]
    sqlite_version_number: u32,
}

impl DatabaseHeader {
    pub fn new(buffer: &[u8]) -> Result<DatabaseHeader> {
        let page_size = u16::from_be_bytes([buffer[16], buffer[17]]);
        let file_format_write_version = buffer[18];
        let file_format_read_version = buffer[19];
        let reserved_space = buffer[20];
        let max_payload_fraction = buffer[21];
        let min_payload_fraction = buffer[22];
        let leaf_payload_fraction = buffer[23];
        let file_change_counter =
            u32::from_be_bytes([buffer[24], buffer[25], buffer[26], buffer[27]]);
        let size_of_file = u32::from_be_bytes([buffer[28], buffer[29], buffer[30], buffer[31]]);
        let freelist_trunk_page_number =
            u32::from_be_bytes([buffer[32], buffer[33], buffer[34], buffer[35]]);
        let num_freelist_pages =
            u32::from_be_bytes([buffer[36], buffer[37], buffer[38], buffer[39]]);
        let schema_cookie = u32::from_be_bytes([buffer[40], buffer[41], buffer[42], buffer[43]]);
        let schema_format_number =
            u32::from_be_bytes([buffer[44], buffer[45], buffer[46], buffer[47]]);
        let default_page_cache_size =
            u32::from_be_bytes([buffer[48], buffer[49], buffer[50], buffer[51]]);
        let largest_root_b_tree_number =
            u32::from_be_bytes([buffer[52], buffer[53], buffer[54], buffer[55]]);
        let text_encoding = match DbTextEncoding::try_from(u32::from_be_bytes([
            buffer[56], buffer[57], buffer[58], buffer[59],
        ])) {
            Ok(enc) => enc,
            Err(e) => bail!(e),
        };
        let user_version = u32::from_be_bytes([buffer[60], buffer[61], buffer[62], buffer[63]]);
        let incremental_vaccum_mode =
            match u32::from_be_bytes([buffer[64], buffer[65], buffer[66], buffer[67]]) {
                0_u32 => false,
                _ => true,
            };
        let sqlite_version_number =
            u32::from_be_bytes([buffer[96], buffer[97], buffer[98], buffer[99]]);

        Ok(DatabaseHeader {
            page_size,
            file_format_write_version,
            file_format_read_version,
            reserved_space,
            max_payload_fraction,
            min_payload_fraction,
            leaf_payload_fraction,
            file_change_counter,
            size_of_file,
            freelist_trunk_page_number,
            num_freelist_pages,
            schema_cookie,
            schema_format_number,
            default_page_cache_size,
            largest_root_b_tree_number,
            text_encoding,
            user_version,
            incremental_vaccum_mode,
            sqlite_version_number,
        })
    }
}
