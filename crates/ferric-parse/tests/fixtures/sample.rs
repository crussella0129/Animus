//! Sample Rust file for ferric-parse integration tests.

pub struct Calculator {
    value: f64,
}

pub enum Operation {
    Add,
    Multiply,
}

pub fn standalone_fn(x: f64) -> f64 {
    x * 2.0
}

impl Calculator {
    pub fn new(value: f64) -> Self {
        Self { value }
    }

    pub fn add(&self, other: f64) -> f64 {
        self.value + other
    }
}
