//! ferric-cli — fast Animus entrypoint.
//! Handles detect/config/status in Rust, delegates rise/ingest/search to Python.

mod commands {
    pub mod detect;
    pub mod config_cmd;
}
mod delegate;

use clap::{Parser, Subcommand};
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "animus", about = "Animus — local-first LLM agent", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Detect hardware and check system capabilities
    Detect,
    /// Show or manage configuration
    Config {
        /// Show full config content
        #[arg(long)]
        show: bool,
        /// Show path to config file
        #[arg(long)]
        path: bool,
    },
    /// Show system status (combines detect + config check)
    Status,
    /// Start an agent session (delegates to Python)
    Rise {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Ingest a codebase (delegates to Python)
    Ingest {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Search the knowledge graph (delegates to Python)
    Search {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Build knowledge graph from a path (delegates to Python)
    Graph {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Download a model (delegates to Python)
    Pull {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Initialize a new workspace (delegates to Python)
    Init {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// List available models (delegates to Python)
    Models {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Manage sessions (delegates to Python)
    Sessions {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Show routing statistics (delegates to Python)
    RoutingStats {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

fn main() -> ExitCode {
    let original_args: Vec<String> = std::env::args().skip(1).collect();
    let cli = Cli::parse();

    match cli.command {
        Commands::Detect => {
            commands::detect::run();
            ExitCode::SUCCESS
        }
        Commands::Config { show: _, path } => {
            if path {
                commands::config_cmd::show_config_path();
            } else {
                commands::config_cmd::show_config();
            }
            ExitCode::SUCCESS
        }
        Commands::Status => {
            commands::detect::run();
            ExitCode::SUCCESS
        }
        // All other commands delegate to Python
        _ => delegate::delegate_to_python(&original_args),
    }
}
