use std::path::Path;
use tree_sitter::{Parser, Node};
use crate::output::{FileParseResult, CodeNode};
use crate::parsers::{LanguageParser, node_text};

pub struct RustParser;

impl LanguageParser for RustParser {
    fn parse_file(&self, path: &Path, source: &str) -> FileParseResult {
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_rust::LANGUAGE.into()).unwrap();

        let tree = match parser.parse(source, None) {
            Some(t) => t,
            None => return FileParseResult { file_path: path.display().to_string(), ..Default::default() },
        };

        let mut result = FileParseResult {
            file_path: path.display().to_string(),
            ..Default::default()
        };

        let module = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
        collect_rust_nodes(tree.root_node(), source, module, &path.display().to_string(), &mut result);
        result
    }
}

fn collect_rust_nodes(node: Node, source: &str, module: &str, file_path: &str, result: &mut FileParseResult) {
    match node.kind() {
        "struct_item" | "enum_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(name_node, source).to_string();
                result.nodes.push(CodeNode {
                    kind: "class".to_string(),
                    name: name.clone(),
                    qualified_name: format!("{}::{}", module, name),
                    file_path: file_path.to_string(),
                    line_start: node.start_position().row + 1,
                    line_end: node.end_position().row + 1,
                    ..Default::default()
                });
            }
        }
        "function_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(name_node, source).to_string();
                result.nodes.push(CodeNode {
                    kind: "function".to_string(),
                    name: name.clone(),
                    qualified_name: format!("{}::{}", module, name),
                    file_path: file_path.to_string(),
                    line_start: node.start_position().row + 1,
                    line_end: node.end_position().row + 1,
                    ..Default::default()
                });
            }
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_rust_nodes(child, source, module, file_path, result);
    }
}
