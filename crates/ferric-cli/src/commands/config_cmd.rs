use std::path::PathBuf;

pub fn show_config() {
    let path = find_config_path();
    match path {
        Some(p) => {
            println!("Config: {}", p.display());
            match std::fs::read_to_string(&p) {
                Ok(contents) => print!("{}", contents),
                Err(e) => eprintln!("Error reading config: {}", e),
            }
        }
        None => println!("No config file found. Run 'animus init' to create one."),
    }
}

pub fn show_config_path() {
    match find_config_path() {
        Some(p) => println!("{}", p.display()),
        None => println!("(none)"),
    }
}

fn find_config_path() -> Option<PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()
        .map(PathBuf::from);

    let candidates = [
        Some(PathBuf::from("config.yaml")),
        home.as_ref().map(|h| h.join(".animus").join("config.yaml")),
    ];

    candidates.into_iter().flatten().find(|p| p.exists())
}
