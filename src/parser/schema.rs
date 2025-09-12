use super::page::PageHeader;
use super::serial_types::SerialType;
use crate::sql::{parse_index_schema, parse_table_schema, Condition, Parser};
use crate::utils::{construct_base_column_bit_set, generate_column_bitmask, read_varint};
use anyhow::{bail, Result};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek};

#[derive(Debug)]
pub struct QueryTargetMetadata {
    pub root_page: i64,
    pub columns: Vec<String>,
}

impl QueryTargetMetadata {
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn column_names(&self) -> &[String] {
        &self.columns
    }
}

#[derive(Debug)]
pub enum PrimaryKey {
    RowId(usize),
    Simple(usize),
    Composite(Vec<usize>),
    None,
}

impl PrimaryKey {
    pub fn from_table_columns(columns: &[TableColumn]) -> PrimaryKey {
        let mut primary_keys: Vec<usize> = Vec::new();

        for (i, col) in columns.iter().enumerate() {
            for cond in &col.constraints {
                if matches!(cond, SQLiteColumnConstraints::PrimaryKey) {
                    primary_keys.push(i);
                }
            }
        }

        match primary_keys.len() {
            0_usize => PrimaryKey::None,
            1_usize => {
                let table_i = primary_keys[0];
                let table = &columns[table_i];
                if matches!(table.data_type, Some(SQLiteColumnType::Integer)) {
                    PrimaryKey::RowId(table_i)
                } else {
                    PrimaryKey::Simple(table_i)
                }
            }
            _ => PrimaryKey::Composite(primary_keys),
        }
    }
}

#[derive(Debug)]
pub struct TableMetadata {
    #[allow(dead_code)]
    pub cell_offset: u16,
    pub root_page: i64,
    #[allow(dead_code)]
    schema_name: String,
    #[allow(dead_code)]
    pub table_name: String,
    pub column_bitmap: HashMap<String, u64>,
    #[allow(dead_code)]
    pub table_schema: TableSchema,
}

#[derive(Debug)]
pub struct IndexMetadataTemp {
    #[allow(dead_code)]
    cell_offset: u16,
    #[allow(dead_code)]
    root_page: i64,
    #[allow(dead_code)]
    schema_name: String,
    #[allow(dead_code)]
    pub index_name: String,
    #[allow(dead_code)]
    pub index_schema: IndexSchema,
}

#[derive(Debug)]
pub struct IndexMetadata {
    #[allow(dead_code)]
    cell_offset: u16,
    #[allow(dead_code)]
    bit_mask: u64,
    #[allow(dead_code)]
    pub root_page: i64,
    #[allow(dead_code)]
    schema_name: String,
    #[allow(dead_code)]
    pub index_name: String,
    #[allow(dead_code)]
    pub available_columns: Vec<String>,
    #[allow(dead_code)]
    pub index_schema: IndexSchema,
}

impl IndexMetadata {
    pub fn contains_all_query_columns(&self, required_bitmask: u64) -> bool {
        (required_bitmask & !self.bit_mask) == 0
    }
}

#[derive(Debug, Clone)]
pub enum SQLiteColumnType {
    Null,
    Integer,
    Real,
    Text,
    Blob,
}

impl TryFrom<&str> for SQLiteColumnType {
    type Error = String;
    fn try_from(type_value: &str) -> Result<Self, Self::Error> {
        let upper_v = type_value.to_uppercase();

        match upper_v.as_str() {
            "INTEGER" => Ok(Self::Integer),
            "TEXT" => Ok(Self::Text),
            "REAL" => Ok(Self::Real),
            "NULL" => Ok(Self::Null),
            "BLOB" => Ok(Self::Blob),
            _ => Err("Invalid type".into()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SQLiteColumnConstraints {
    PrimaryKey,
    AutoIncrement,
    Ascending,
    Descending,
    NotNull,
    Unique,
    Default(String),
    Check(Condition),
    Collate(String),
    ForiegnKey {
        table: String,
        column: String,
        on_delete: Option<String>,
        on_update: Option<String>,
    },
}

#[derive(Debug, Clone)]
pub struct TableColumn {
    pub name: String,
    pub data_type: Option<SQLiteColumnType>,
    pub constraints: Vec<SQLiteColumnConstraints>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct TableSchema {
    columns: Vec<TableColumn>,
    primary_key: PrimaryKey,
    without_rowid: bool,
}

#[derive(Debug)]
pub struct IndexColumn {
    pub name: String,
    pub constraints: Vec<SQLiteColumnConstraints>,
}

// TODO:
#[derive(Debug)]
pub struct IndexSchema {
    #[allow(dead_code)]
    unique: bool,
    base_table_name: String,
    index_columns: Vec<IndexColumn>,
}

impl IndexSchema {
    fn new(schema: String) -> Result<(String, IndexSchema)> {
        println!("index schema: {schema}");
        let parser = parse_index_schema();
        let Ok((_, (index_name, base_table_name, unique, index_columns))) = parser.parse(&schema)
        else {
            bail!("Invalid index schema")
        };

        Ok((
            index_name,
            IndexSchema {
                base_table_name: base_table_name.to_uppercase(),
                unique,
                index_columns,
            },
        ))
    }
}

#[derive(Debug)]
pub enum SchemaType {
    Table,
    Index,
    Trigger,
    View,
}

impl TryFrom<String> for SchemaType {
    type Error = String;
    fn try_from(value: String) -> Result<SchemaType, Self::Error> {
        match value.to_lowercase().as_str() {
            "table" => Ok(Self::Table),
            "index" => Ok(Self::Index),
            "trigger" => Ok(Self::Trigger),
            "view" => Ok(Self::View),
            _ => Err("invalid schema type".into()),
        }
    }
}

impl TableSchema {
    fn new(schema: String) -> Result<(String, TableSchema)> {
        let table_schema_parser = parse_table_schema();
        if let Ok((_, (name, columns, primary_key, without_rowid))) =
            table_schema_parser.parse(&schema)
        {
            Ok((
                name,
                TableSchema {
                    columns,
                    primary_key,
                    without_rowid,
                },
            ))
        } else {
            bail!("Invalid schema")
        }
    }

    pub fn contains_all_index_columns(&self, index_columns: &[String]) -> bool {
        let columns_set = self
            .columns
            .clone()
            .into_iter()
            .map(|c| c.name)
            .collect::<HashSet<String>>();

        for column in index_columns {
            if !columns_set.contains(column) {
                return false;
            }
        }
        true
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn column_names(&self) -> Vec<&str> {
        self.columns
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<&str>>()
    }
}

#[derive(Debug)]
pub struct SqliteSchema {
    tables: Vec<TableMetadata>,
    indexes: Vec<IndexMetadata>,
    // indexes of each table in table vec
    table_map: HashMap<String, usize>,
    // offsets into the index array for all indexes on a table
    table_indexes: HashMap<String, Vec<usize>>,
    pub page_header: PageHeader,
}

// parse the first page to get the schema
// the first page sits after the database header and contains the schema
// TODO:
//  if the table is withrowid, ie the WITHOUT ROWID qualifier is not set
//      && the primary key is an integer, the primary key is the row id
//  if the table is withrow id, but the primary key is not an integer, the primary key is
impl SqliteSchema {
    pub fn new(file: &mut File) -> Result<SqliteSchema> {
        file.seek(std::io::SeekFrom::Start(100))?;
        // buffer to read in the btree header
        let mut bheader_buffer: [u8; 12] = [0; 12];

        file.read_exact(&mut bheader_buffer)?;
        let page_header = PageHeader::new(&bheader_buffer)?;
        // this will be the size of the cell pointer array

        let mut bcell_pointer: [u8; 2] = [0; 2];
        file.seek(std::io::SeekFrom::Start((100 + page_header.len()).into()))?;

        let mut cell_offsets: Vec<u16> = Vec::with_capacity(page_header.num_cells.into());
        for _ in 0..page_header.num_cells {
            file.read_exact(&mut bcell_pointer)?;
            cell_offsets.push(u16::from_be_bytes([bcell_pointer[0], bcell_pointer[1]]));
        }

        let mut tables: Vec<TableMetadata> = Vec::with_capacity(cell_offsets.len());
        let mut table_map: HashMap<String, usize> = HashMap::with_capacity(cell_offsets.len());
        let mut table_indexes: HashMap<String, Vec<usize>> =
            HashMap::with_capacity(cell_offsets.len());
        let mut temp_indexes: Vec<IndexMetadataTemp> = Vec::with_capacity(cell_offsets.len());

        // TODO:
        // in the first pass, define the tables and indexes
        // take another pass over the indexes to retrieve the primary keys from the table
        // then compute the bitmap for the column names
        for offset in cell_offsets {
            // read record in
            // get record size
            // record size is a varint, not a single byte
            // TODO: read the varint, rather than reading a single byte

            let (header_bytes_read, schema_record_header) =
                SchemaRecordHeader::new(offset.into(), file)?;

            let SchemaRecordHeader {
                record_header_size,
                schema_type_serial_type,
                schema_name_serial_type,
                table_name_serial_type,
                root_page_serial_type,
                schema_serial_type,
                ..
            } = schema_record_header;

            if header_bytes_read < record_header_size {
                file.seek(std::io::SeekFrom::Current(
                    record_header_size - header_bytes_read,
                ))?;
            }

            let mut schema_type_buffer = vec![0_u8; schema_type_serial_type.size()];
            file.read_exact(&mut schema_type_buffer)
                .expect("reading schema type buffer failed");
            let Ok(schema_type) = SchemaType::try_from(String::from_utf8(schema_type_buffer)?)
            else {
                bail!("Invalid schema type");
            };

            let mut schema_name_buffer = vec![0_u8; schema_name_serial_type.size()];
            file.read_exact(&mut schema_name_buffer)
                .expect("reading schema name buffer failed");
            let schema_name = String::from_utf8(schema_name_buffer)?.to_uppercase();

            let mut table_name_buffer = vec![0_u8; table_name_serial_type.size()];
            file.read_exact(&mut table_name_buffer)
                .expect("reading table name buffer failed");
            let table_name = String::from_utf8(table_name_buffer)?.to_uppercase();

            let mut root_page_buffer = vec![0_u8; root_page_serial_type.size()];
            file.read_exact(&mut root_page_buffer)
                .expect("reading table name buffer failed");
            let root_page_value = match root_page_serial_type {
                SerialType::Int8 => root_page_buffer[0] as i64,
                SerialType::Int16 => {
                    u16::from_be_bytes([root_page_buffer[0], root_page_buffer[1]]) as i64
                }
                SerialType::Int24 => u32::from_be_bytes([
                    0_u8,
                    root_page_buffer[0],
                    root_page_buffer[1],
                    root_page_buffer[2],
                ]) as i64,
                SerialType::Int32 => u32::from_be_bytes([
                    root_page_buffer[0],
                    root_page_buffer[1],
                    root_page_buffer[2],
                    root_page_buffer[3],
                ]) as i64,
                _ => bail!("Invalid serial type for root page value"),
            };

            let mut schema_buffer = vec![0_u8; schema_serial_type.size()];
            file.read_exact(&mut schema_buffer)
                .expect("reading schema failed");
            let schema_str = String::from_utf8(schema_buffer)?;
            match schema_type {
                SchemaType::Table => {
                    let (table_name_str, table_schema) = TableSchema::new(schema_str)?;

                    let column_name_refs = table_schema
                        .columns
                        .iter()
                        .map(|s| s.name.as_str())
                        .collect::<Vec<&str>>();

                    let column_bitmap = construct_base_column_bit_set(&column_name_refs);
                    table_map.insert(table_name_str.to_uppercase(), tables.len());
                    tables.push(TableMetadata {
                        root_page: root_page_value,
                        cell_offset: offset,
                        column_bitmap,
                        schema_name,
                        table_name,
                        table_schema,
                    });
                }
                SchemaType::Index => {
                    let (index_name, index_schema) = IndexSchema::new(schema_str)?;

                    let lookup_key = index_schema.base_table_name.to_uppercase();
                    if let Some(table_indexes) = table_indexes.get_mut(&lookup_key) {
                        table_indexes.push(temp_indexes.len());
                    } else {
                        let new_index_ref: Vec<usize> = vec![temp_indexes.len()];
                        table_indexes.insert(lookup_key, new_index_ref);
                    }
                    temp_indexes.push(IndexMetadataTemp {
                        root_page: root_page_value,
                        cell_offset: offset,
                        schema_name,
                        index_name,
                        index_schema,
                    })
                }
                SchemaType::View => todo!(),
                SchemaType::Trigger => todo!(),
            }
        }

        let mut indexes: Vec<IndexMetadata> = Vec::with_capacity(temp_indexes.len());
        for index in temp_indexes.into_iter() {
            // TODO:
            //  grab the base table
            //  add the primary keys
            //  push the ready index
            let IndexMetadataTemp {
                cell_offset,
                root_page,
                schema_name,
                index_name,
                index_schema,
            } = index;
            let base_table_name = &index_schema.base_table_name;
            let base_table_ref = &tables[table_map[base_table_name]];
            let (bit_mask, available_columns) = match base_table_ref.table_schema.primary_key {
                PrimaryKey::None => {
                    // geenerate bitset based on just columns in the index
                    (
                        0_u64,
                        index_schema
                            .index_columns
                            .iter()
                            .map(|c| c.name.clone())
                            .collect(),
                    )
                }
                PrimaryKey::RowId(i) => {
                    let mut cols_w_pk: Vec<String> =
                        Vec::with_capacity(index_schema.index_columns.len());
                    cols_w_pk.push(base_table_ref.table_schema.columns[i].name.to_uppercase());
                    for col in &index_schema.index_columns {
                        cols_w_pk.push(col.name.to_uppercase())
                    }
                    // here the primary key is an alias for row id
                    // fetch the column name and leverage the row id when parsing records
                    // TODO: compute col bitset
                    (
                        generate_column_bitmask(&base_table_ref.column_bitmap, &cols_w_pk),
                        cols_w_pk,
                    )
                }
                PrimaryKey::Simple(i) => {
                    // a non integer single primary key
                    let mut cols_w_pk: Vec<String> =
                        Vec::with_capacity(index_schema.index_columns.len());
                    cols_w_pk.push(base_table_ref.table_schema.columns[i].name.to_uppercase());
                    for col in &index_schema.index_columns {
                        cols_w_pk.push(col.name.to_uppercase())
                    }

                    (
                        generate_column_bitmask(&base_table_ref.column_bitmap, &cols_w_pk),
                        cols_w_pk,
                    )
                }
                PrimaryKey::Composite(ref pk_set) => {
                    let mut cols_w_pk: Vec<String> =
                        Vec::with_capacity(index_schema.index_columns.len() + pk_set.len());
                    for i in pk_set.iter() {
                        cols_w_pk.push(base_table_ref.table_schema.columns[*i].name.to_uppercase());
                    }
                    for col in &index_schema.index_columns {
                        cols_w_pk.push(col.name.to_uppercase())
                    }
                    // composite primary key
                    (
                        generate_column_bitmask(&base_table_ref.column_bitmap, &cols_w_pk),
                        cols_w_pk,
                    )
                }
            };
            indexes.push(IndexMetadata {
                cell_offset,
                root_page,
                index_schema,
                available_columns,
                index_name,
                bit_mask,
                schema_name,
            })
        }
        Ok(SqliteSchema {
            table_map,
            table_indexes,
            tables,
            indexes,
            page_header,
        })
    }

    pub fn resolve_query_metadata(
        &self,
        table_name: &str,
        query_columns: Option<&[String]>,
    ) -> Option<&IndexMetadata> {
        // if an index can service the query, add the table primary key to the columns
        // if the index cannot, pass all table columns
        // then pass back the root page number and the columns in a QueryMetadata
        //
        let Some(table_i) = self.table_map.get(table_name) else {
            return None;
        };

        let Some(q_cols) = query_columns else {
            return None;
        };

        let base_table = &self.tables[*table_i];
        let required_bitmask = generate_column_bitmask(&base_table.column_bitmap, q_cols);
        self.fetch_index_to_service_query(table_name, required_bitmask)
    }

    pub fn get_tables(&self) -> &[TableMetadata] {
        &self.tables
    }

    fn index_can_serve_query<'a>(&self, index_i: usize, bitset: u64) -> bool {
        let Some(index) = self.indexes.get(index_i) else {
            return false;
        };

        if index.contains_all_query_columns(bitset) {
            true
        } else {
            false
        }
    }

    fn fetch_index_to_service_query(
        &self,
        table_name: &str,
        required_bitset: u64,
    ) -> Option<&IndexMetadata> {
        let Some(table_index_offsets) = self.table_indexes.get(table_name) else {
            println!("no indexes for table");
            return None;
        };
        for i in table_index_offsets {
            let j = *i;
            if self.index_can_serve_query(j, required_bitset) {
                return self.indexes.get(j);
            }
        }
        None
    }

    #[allow(dead_code)]
    pub fn fetch_indexes(&self, table_name: &str) -> Option<Vec<&IndexMetadata>> {
        if let Some(table_index_offsets) = self.table_indexes.get(table_name) {
            let mut rel_indexes: Vec<&IndexMetadata> =
                Vec::with_capacity(table_index_offsets.len());
            for i in table_index_offsets {
                let Some(index_ref) = self.indexes.get(*i) else {
                    panic!("Index vec is invalid");
                };
                rel_indexes.push(index_ref);
            }
            Some(rel_indexes)
        } else {
            return None;
        }
    }

    pub fn fetch_table(&self, table_name: &str) -> Option<&TableMetadata> {
        if let Some(i) = self.table_map.get(table_name) {
            self.tables.get(*i)
        } else {
            None
        }
    }
}

struct SchemaRecordHeader {
    record_header_size: i64,
    schema_type_serial_type: SerialType,
    schema_name_serial_type: SerialType,
    table_name_serial_type: SerialType,
    root_page_serial_type: SerialType,
    schema_serial_type: SerialType,
}

impl SchemaRecordHeader {
    fn new(offset: u64, file: &mut File) -> Result<(i64, SchemaRecordHeader)> {
        file.seek(std::io::SeekFrom::Start(offset))?;

        let (_record_size, _) = read_varint(file)?;

        let (_row_id, _) = read_varint(file)?;

        let mut header_bytes_read = 0_usize;
        let (record_header_size, header_size_bytes_read) = read_varint(file)?;
        header_bytes_read += header_size_bytes_read;

        let (schema_type_size, schema_type_size_bytes_read) = read_varint(file)?;
        header_bytes_read += schema_type_size_bytes_read;
        let schema_type_serial_type = SerialType::new(schema_type_size as usize)?;

        let (schema_name_size, schema_name_size_bytes_read) = read_varint(file)?;
        header_bytes_read += schema_name_size_bytes_read;
        let schema_name_serial_type = SerialType::new(schema_name_size as usize)?;

        let (table_name_size, table_name_size_bytes_read) = read_varint(file)?;
        header_bytes_read += table_name_size_bytes_read;
        let table_name_serial_type = SerialType::new(table_name_size as usize)?;

        let (root_page_size, root_page_size_bytes_read) = read_varint(file)?;
        header_bytes_read += root_page_size_bytes_read;
        let root_page_serial_type = SerialType::new(root_page_size as usize)?;

        let (schema_size, schema_size_bytes_read) = read_varint(file)?;
        header_bytes_read += schema_size_bytes_read;
        let schema_serial_type = SerialType::new(schema_size as usize)?;

        Ok((
            header_bytes_read as i64,
            SchemaRecordHeader {
                record_header_size,
                schema_type_serial_type,
                schema_name_serial_type,
                table_name_serial_type,
                root_page_serial_type,
                schema_serial_type,
            },
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse_schema() {
        let schema: String = r#"
            CREATE TABLE "superheroes" (
                id integer primary key autoincrement,
                name text not null,
                eye_color text,
                hair_color text,
                appearance_count integer,
                first_appearance text,
                first_appearance_year text
            )"#
        .into();
        let _table_schema = TableSchema::new(schema).unwrap();
    }

    #[test]
    fn test_bit_map() {
        let columns = vec![
            "ID",
            "NAME",
            "EYE_COLOR",
            "HAIR_COLOR",
            "APPEARANCE_COUNT",
            "FIRST_APPEARANCE",
        ];

        let _bitset = construct_base_column_bit_set(&columns);

        // let col_res =
    }
}
