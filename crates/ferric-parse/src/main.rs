mod output;
mod parsers;

use clap::Parser as ClapParser;
use std::path::Path;

#[derive(ClapParser)]
#[command(name = "ferric-parse", about = "Multi-language code parser for Animus (internal tool)")]
struct Cli {
    /// File to parse
    file: String,
    /// Output format
    #[arg(long, default_value = "json")]
    format: String,
}

fn main() {
    let cli = Cli::parse();
    let path = Path::new(&cli.file);

    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ferric-parse: failed to read {}: {}", cli.file, e);
            std::process::exit(1);
        }
    };

    let result = match parsers::get_parser_for_extension(ext) {
        Some(parser) => parser.parse_file(path, &source),
        None => {
            // Unsupported extension — emit empty result, don't error
            output::FileParseResult {
                file_path: cli.file.clone(),
                ..Default::default()
            }
        }
    };

    println!("{}", serde_json::to_string(&result).unwrap());
}
