use std::path::Path;
use crate::output::FileParseResult;

pub mod python;
pub mod rust_lang;

pub trait LanguageParser {
    fn parse_file(&self, path: &Path, source: &str) -> FileParseResult;
}

pub fn get_parser_for_extension(ext: &str) -> Option<Box<dyn LanguageParser>> {
    match ext {
        "py" => Some(Box::new(python::PythonParser)),
        "rs" => Some(Box::new(rust_lang::RustParser)),
        _ => None,
    }
}
