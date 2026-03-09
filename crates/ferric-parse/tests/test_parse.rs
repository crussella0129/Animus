use std::process::Command;
use std::path::PathBuf;

fn ferric_parse_binary() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.pop(); // test binary dir
    path.pop(); // deps/
    path.push("ferric-parse");
    #[cfg(windows)]
    path.set_extension("exe");
    path
}

#[test]
fn test_parse_python_file_produces_nodes() {
    let binary = ferric_parse_binary();
    if !binary.exists() {
        println!("Skipping: ferric-parse binary not found at {:?}", binary);
        return;
    }

    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sample.py");

    let output = Command::new(&binary)
        .arg(fixture.to_str().unwrap())
        .output()
        .expect("Failed to run ferric-parse");

    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)
        .expect("Output should be valid JSON");

    assert!(json["nodes"].is_array());
    let nodes = json["nodes"].as_array().unwrap();
    assert!(!nodes.is_empty(), "Should have found at least one node");

    let names: Vec<&str> = nodes.iter()
        .filter_map(|n| n["name"].as_str())
        .collect();
    assert!(names.contains(&"Calculator"), "Expected to find Calculator class");
    assert!(names.contains(&"standalone_function"), "Expected to find standalone_function");
}
