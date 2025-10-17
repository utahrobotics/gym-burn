use burn::backend::{Autodiff, wgpu::CubeBackend};
use burn::cubecl::wgpu::WgpuRuntime;

mod implementations;
mod tracking;

pub type InnerBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
pub type InnerAutodiffBackend = Autodiff<InnerBackend>;

#[derive(Default, Debug, Clone)]
#[repr(transparent)]
pub struct TrackingBackend(InnerAutodiffBackend);

pub use tracking::{set_artifact_dir, wait_until_paused};
