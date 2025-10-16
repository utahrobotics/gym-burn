use burn::backend::wgpu::CubeBackend;
use burn::cubecl::wgpu::WgpuRuntime;

mod implementations;

pub type InnerBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

#[derive(Default, Debug, Clone)]
#[repr(transparent)]
pub struct TrackingBackend(InnerBackend);
