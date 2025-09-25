#[cfg(feature = "app")]
fn main() {}

#[cfg(not(feature = "app"))]
fn main() {
    println!("The `app` feature needs to be enabled")
}
