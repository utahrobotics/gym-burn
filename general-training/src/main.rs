#[cfg(feature = "app")]
use general_training::app::main;

#[cfg(not(feature = "app"))]
fn main() {
    println!("The `app` feature needs to be enabled")
}
