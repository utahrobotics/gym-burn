use std::sync::LazyLock;

use burn::backend::{Wgpu, wgpu::WgpuDevice};

pub type WgpuBackend = Wgpu<f32, i32>;

static DEVICE: LazyLock<WgpuDevice> = LazyLock::new(Default::default);

pub fn get_device() -> &'static WgpuDevice {
    &DEVICE
}
