use anyhow::{bail, Result};

#[derive(Debug)]
pub enum SerialType {
    Null,
    Int8,
    Int16,
    Int24,
    Int32,
    Int48,
    Int64,
    Float64,
    BoolFalse,
    BoolTrue,
    Blob(i64),
    String(i64),
}

impl SerialType {
    pub fn new(size: usize) -> Result<Self> {
        match size {
            0_usize => Ok(Self::Null),
            1_usize => Ok(Self::Int8),
            2_usize => Ok(Self::Int16),
            3_usize => Ok(Self::Int24),
            4_usize => Ok(Self::Int32),
            5_usize => Ok(Self::Int48),
            6_usize => Ok(Self::Int64),
            7_usize => Ok(Self::Float64),
            8_usize => Ok(Self::BoolFalse),
            9_usize => Ok(Self::BoolTrue),
            10_usize | 11_usize => bail!("reserved size"),
            _ if size % 2 == 0 => Ok(Self::Blob(((size - 12) / 2) as i64)),
            _ => Ok(Self::String(((size - 13) / 2) as i64)),
        }
    }

    pub fn size(&self) -> usize {
        match *self {
            Self::Null => 0_usize,
            Self::Int8 => 1_usize,
            Self::Int16 => 2_usize,
            Self::Int24 => 3_usize,
            Self::Int32 => 4_usize,
            Self::Int48 => 6_usize,
            Self::Int64 => 8_usize,
            Self::Float64 => 8_usize,
            Self::BoolFalse => 0_usize,
            Self::BoolTrue => 0_usize,
            Self::Blob(size) => size as usize,
            Self::String(size) => size as usize,
        }
    }
}
