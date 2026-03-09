use serde::{Deserialize, Serialize};

/// A parsed code node (class, function, method).
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct CodeNode {
    pub kind: String,
    pub name: String,
    pub qualified_name: String,
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub docstring: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub bases: Vec<String>,
    #[serde(default)]
    pub decorators: Vec<String>,
}

/// A call or import edge between nodes.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CodeEdge {
    pub source_qname: String,
    pub target_name: String,
    pub kind: String,
}

/// Full parse result for one file — matches Python FileParseResult schema.
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct FileParseResult {
    pub file_path: String,
    #[serde(default)]
    pub nodes: Vec<CodeNode>,
    #[serde(default)]
    pub edges: Vec<CodeEdge>,
}
