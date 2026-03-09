use std::path::Path;
use tree_sitter::{Parser, Node};
use crate::output::{FileParseResult, CodeNode};
use crate::parsers::LanguageParser;

pub struct PythonParser;

impl LanguageParser for PythonParser {
    fn parse_file(&self, path: &Path, source: &str) -> FileParseResult {
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();

        let tree = match parser.parse(source, None) {
            Some(t) => t,
            None => return FileParseResult { file_path: path.display().to_string(), ..Default::default() },
        };

        let mut result = FileParseResult {
            file_path: path.display().to_string(),
            ..Default::default()
        };

        let module = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        collect_python_nodes(tree.root_node(), source, module, &path.display().to_string(), &mut result);
        result
    }
}

fn get_node_text<'a>(node: Node, source: &'a str) -> &'a str {
    &source[node.start_byte()..node.end_byte()]
}

fn collect_python_nodes(
    node: Node,
    source: &str,
    module: &str,
    file_path: &str,
    result: &mut FileParseResult,
) {
    match node.kind() {
        "class_definition" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = get_node_text(name_node, source).to_string();
                let qualified = format!("{}.{}", module, name);
                let docstring = extract_docstring(node, source);

                let bases = if let Some(args_node) = node.child_by_field_name("superclasses") {
                    collect_identifiers(args_node, source)
                } else {
                    Vec::new()
                };

                result.nodes.push(CodeNode {
                    kind: "class".to_string(),
                    name: name.clone(),
                    qualified_name: qualified,
                    file_path: file_path.to_string(),
                    line_start: node.start_position().row + 1,
                    line_end: node.end_position().row + 1,
                    docstring,
                    bases,
                    ..Default::default()
                });
            }
        }
        "function_definition" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = get_node_text(name_node, source).to_string();
                let qualified = format!("{}.{}", module, name);
                let docstring = extract_docstring(node, source);

                result.nodes.push(CodeNode {
                    kind: "function".to_string(),
                    name: name.clone(),
                    qualified_name: qualified,
                    file_path: file_path.to_string(),
                    line_start: node.start_position().row + 1,
                    line_end: node.end_position().row + 1,
                    docstring,
                    ..Default::default()
                });
            }
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_python_nodes(child, source, module, file_path, result);
    }
}

fn extract_docstring(node: Node, source: &str) -> Option<String> {
    if let Some(body) = node.child_by_field_name("body") {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() == "expression_statement" {
                if let Some(string_node) = child.child(0) {
                    if string_node.kind() == "string" {
                        let text = get_node_text(string_node, source);
                        return Some(text.trim_matches(|c| c == '"' || c == '\'').to_string());
                    }
                }
            }
            break;
        }
    }
    None
}

fn collect_identifiers(node: Node, source: &str) -> Vec<String> {
    let mut ids = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" {
            ids.push(get_node_text(child, source).to_string());
        }
    }
    ids
}
