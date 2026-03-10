use sysinfo::System;

pub fn run() {
    let mut sys = System::new_all();
    sys.refresh_all();

    println!("System Information");
    println!("==================");
    println!("OS:          {} {}", System::name().unwrap_or_default(),
             System::os_version().unwrap_or_default());
    println!("Kernel:      {}", System::kernel_version().unwrap_or_default());
    println!("Hostname:    {}", System::host_name().unwrap_or_default());
    println!("CPU cores:   {}", sys.cpus().len());
    println!("Total RAM:   {} MB", sys.total_memory() / 1024 / 1024);
    println!("Free RAM:    {} MB", sys.available_memory() / 1024 / 1024);

    println!();
    println!("Ferric Layer:");
    println!("  ferric-parse:   {}", check_binary("ferric-parse"));
    println!("  ferric-sandbox: {}", check_binary("ferric-sandbox"));
}

fn check_binary(name: &str) -> &'static str {
    if which_exists(name) { "available" } else { "not found" }
}

fn which_exists(name: &str) -> bool {
    let separator = if cfg!(windows) { ';' } else { ':' };
    std::env::var("PATH")
        .unwrap_or_default()
        .split(separator)
        .any(|dir| {
            let mut p = std::path::PathBuf::from(dir);
            p.push(name);
            if cfg!(windows) { p.set_extension("exe"); }
            p.exists()
        })
}
