#![recursion_limit = "256"]

#[cfg(feature = "app")]
pub mod app;

pub mod trainable_models;
pub mod training_loop;
