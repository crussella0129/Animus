use super::{detect, config_cmd};

pub fn run() {
    detect::run();
    println!();
    println!("Configuration");
    println!("=============");
    config_cmd::show_config();
}
