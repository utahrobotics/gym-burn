#![recursion_limit = "256"]

#[cfg(feature = "app")]
pub mod app;
// pub mod batches;
// pub mod dataset;
pub mod regression;
pub mod training_loop;
