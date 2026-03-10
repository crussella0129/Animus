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
        collect_rust_nodes(tree.root_node(), source, module, &path.display().to_string(), None, &mut result);
        result
    }
}

fn collect_rust_nodes<'a>(
    node: Node<'a>,
    source: &str,
    module: &str,
    file_path: &str,
    impl_name: Option<&str>,
    result: &mut FileParseResult,
) {
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
        "impl_item" => {
            // Extract the type being implemented to pass as context to child functions
            if let Some(type_node) = node.child_by_field_name("type") {
                let name = node_text(type_node, source).to_string();
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    collect_rust_nodes(child, source, module, file_path, Some(&name), result);
                }
                return;
            }
        }
        "function_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(name_node, source).to_string();
                let (kind, qualified_name) = if let Some(impl_type) = impl_name {
                    (
                        "method".to_string(),
                        format!("{}::{}::{}", module, impl_type, name),
                    )
                } else {
                    (
                        "function".to_string(),
                        format!("{}::{}", module, name),
                    )
                };
                result.nodes.push(CodeNode {
                    kind,
                    name: name.clone(),
                    qualified_name,
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
        collect_rust_nodes(child, source, module, file_path, impl_name, result);
    }
}
