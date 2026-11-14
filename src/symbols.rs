//! Function names for kernel image addresses, from an analysis database
//! export.
//!
//! The accepted file is a JSON object keyed by anything, whose values carry
//! `"name"`, `"start"` and `"end"` (hex strings, end exclusive):
//!
//! ```json
//! { "0x800ab9f0": { "name": "Kernel_Main", "start": "0x800ab9f0", "end": "0x800ac270" } }
//! ```
//!
//! Other fields are ignored. Addresses are link-time image addresses.

use std::{fmt, io, path::Path};

use serde::Deserialize;

#[derive(Debug)]
pub enum SymbolError {
    Io(io::Error),
    Json(serde_json::Error),
    /// An entry's `start` or `end` is not a hex address.
    Address(String),
}

impl fmt::Display for SymbolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "cannot read symbols: {error}"),
            Self::Json(error) => write!(f, "malformed symbol file: {error}"),
            Self::Address(text) => {
                write!(f, "malformed symbol file: {text:?} is not a hex address")
            }
        }
    }
}

impl std::error::Error for SymbolError {}

#[derive(Deserialize)]
struct Entry {
    name: String,
    start: String,
    end: String,
}

#[derive(Debug)]
struct Function {
    start: u64,
    end: u64,
    name: String,
}

/// A function containing an address, and the address's offset into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Symbol<'a> {
    pub name: &'a str,
    pub offset: u64,
}

impl fmt::Display for Symbol<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.offset {
            0 => f.write_str(self.name),
            offset => write!(f, "{}+{offset:#x}", self.name),
        }
    }
}

#[derive(Debug, Default)]
pub struct SymbolTable {
    /// Sorted by start address.
    functions: Vec<Function>,
}

impl SymbolTable {
    pub fn load(path: &Path) -> Result<Self, SymbolError> {
        Self::parse(&std::fs::read_to_string(path).map_err(SymbolError::Io)?)
    }

    pub fn parse(json: &str) -> Result<Self, SymbolError> {
        let entries: std::collections::HashMap<String, Entry> =
            serde_json::from_str(json).map_err(SymbolError::Json)?;
        let address = |text: &str| {
            u64::from_str_radix(text.trim_start_matches("0x"), 16)
                .map_err(|_| SymbolError::Address(text.into()))
        };
        let mut functions = entries
            .into_values()
            .map(|entry| {
                Ok(Function {
                    start: address(&entry.start)?,
                    end: address(&entry.end)?,
                    name: entry.name,
                })
            })
            .collect::<Result<Vec<_>, SymbolError>>()?;
        functions.sort_by_key(|function| function.start);
        Ok(Self { functions })
    }

    /// The function whose range contains `address`.
    pub fn resolve(&self, address: u64) -> Option<Symbol<'_>> {
        let index = self
            .functions
            .partition_point(|function| function.start <= address)
            .checked_sub(1)?;
        let function = &self.functions[index];
        (address < function.end).then(|| Symbol {
            name: &function.name,
            offset: address - function.start,
        })
    }

    /// Start address of the function called `name`.
    pub fn address_of(&self, name: &str) -> Option<u64> {
        self.functions
            .iter()
            .find(|function| function.name == name)
            .map(|function| function.start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_addresses_inside_functions_and_not_in_the_gaps_between_them() {
        let table = SymbolTable::parse(
            r#"{
                "b": { "name": "Second", "start": "0x2000", "end": "0x2010", "callers": [] },
                "a": { "name": "First", "start": "0x1000", "end": "0x1008" }
            }"#,
        )
        .unwrap();
        assert_eq!(
            table.resolve(0x1000),
            Some(Symbol {
                name: "First",
                offset: 0
            })
        );
        assert_eq!(table.resolve(0x200c).unwrap().to_string(), "Second+0xc");
        for outside in [0xfff, 0x1008, 0x1fff, 0x2010] {
            assert_eq!(table.resolve(outside), None, "{outside:#x}");
        }
        assert_eq!(table.address_of("Second"), Some(0x2000));
        assert_eq!(table.address_of("Third"), None);
    }
}
