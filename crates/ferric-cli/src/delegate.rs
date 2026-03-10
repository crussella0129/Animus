use std::process::ExitCode;

/// Find the Python interpreter on PATH.
fn find_python() -> String {
    for candidate in &["python3", "python"] {
        if std::process::Command::new(candidate)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
        {
            return candidate.to_string();
        }
    }
    "python".to_string()
}

/// Delegate a command to the Python CLI (python -m src.main).
/// All original args are passed through. Returns the Python process exit code.
pub fn delegate_to_python(args: &[String]) -> ExitCode {
    let python = find_python();
    let status = std::process::Command::new(&python)
        .arg("-m")
        .arg("src.main")
        .args(args)
        .status();

    match status {
        Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(e) => {
            eprintln!("ferric-cli: failed to launch Python runtime: {}", e);
            ExitCode::FAILURE
        }
    }
}
