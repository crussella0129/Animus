use clap::Parser as ClapParser;
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::process::Command;
use std::time::Instant;

#[derive(ClapParser)]
#[command(name = "ferric-sandbox", about = "Process isolation for Animus tool execution (internal)")]
struct Cli {
    #[arg(long, default_value = "512")]
    memory: u64,
    #[arg(long, default_value = "30")]
    timeout: u64,
    #[arg(long)]
    no_network: bool,
    #[arg(long)]
    read_only: bool,
    #[arg(long)]
    smough: bool,
}

#[derive(Deserialize)]
struct SandboxRequest {
    args: Vec<String>,
}

#[derive(Serialize)]
struct IsolationInfo {
    ornstein: bool,
    smough: bool,
    reason: String,
}

#[derive(Serialize)]
struct ResourceUsage {
    wall_time_ms: u128,
    cpu_time_ms: u64,
    peak_memory_kb: u64,
}

#[derive(Serialize)]
struct SandboxResult {
    success: bool,
    output: String,
    error: Option<String>,
    exit_code: i32,
    isolation: IsolationInfo,
    resource_usage: ResourceUsage,
}

fn main() {
    let cli = Cli::parse();

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap_or(0);

    let req: SandboxRequest = match serde_json::from_str(&input) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ferric-sandbox: invalid input JSON: {}", e);
            std::process::exit(1);
        }
    };

    let start = Instant::now();

    // Platform detection — kernel features only on Linux (future work)
    let (ornstein_active, ornstein_reason) = {
        #[cfg(target_os = "linux")]
        {
            (false, "Linux detected — kernel isolation not yet implemented, using process isolation".to_string())
        }
        #[cfg(not(target_os = "linux"))]
        {
            (false, format!("Platform {} does not support kernel isolation — using process isolation", std::env::consts::OS))
        }
    };

    // Run the command using the pre-split args list (preserves arguments with spaces)
    if req.args.is_empty() {
        eprintln!("ferric-sandbox: empty args");
        std::process::exit(1);
    }

    let result = Command::new(&req.args[0])
        .args(&req.args[1..])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();

    let wall_time = start.elapsed().as_millis();

    // Suppress unused variable warning for cli flags (reserved for future kernel features)
    let _ = (cli.memory, cli.timeout, cli.no_network, cli.read_only, cli.smough);

    let sandbox_result = match result {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            let exit_code = out.status.code().unwrap_or(-1);
            SandboxResult {
                success: out.status.success(),
                output: stdout,
                error: if stderr.is_empty() { None } else { Some(stderr) },
                exit_code,
                isolation: IsolationInfo {
                    ornstein: ornstein_active,
                    smough: false,
                    reason: ornstein_reason,
                },
                resource_usage: ResourceUsage {
                    wall_time_ms: wall_time,
                    cpu_time_ms: 0,
                    peak_memory_kb: 0,
                },
            }
        }
        Err(e) => SandboxResult {
            success: false,
            output: String::new(),
            error: Some(e.to_string()),
            exit_code: -1,
            isolation: IsolationInfo {
                ornstein: ornstein_active,
                smough: false,
                reason: ornstein_reason,
            },
            resource_usage: ResourceUsage {
                wall_time_ms: wall_time,
                cpu_time_ms: 0,
                peak_memory_kb: 0,
            },
        },
    };

    println!("{}", serde_json::to_string(&sandbox_result).unwrap());
}
