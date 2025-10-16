use burn::backend::{Autodiff, Wgpu};

mod implementations;

pub type InnerBackend = Autodiff<Wgpu<f32, i32, u32>>;

#[derive(Default, Debug, Clone)]
#[repr(transparent)]
pub struct TrackingBackend(InnerBackend);
