use burn::{Tensor, module::Module, prelude::Backend};

pub mod autoencoder;
pub mod serde_fix;
pub trait FromConfig<B: Backend> {
    type Config;

    fn init(config: Self::Config, device: &B::Device) -> Self;
}

pub trait SimpleForwardable<B: Backend, const N_I: usize, const N_O: usize>:
    Module<B> + FromConfig<B>
{
    fn forward(&self, images: Tensor<B, N_I>) -> Tensor<B, N_O>;
}

#[cfg(feature = "wgpu")]
pub mod wgpu {
    use std::sync::LazyLock;

    use burn::backend::{Wgpu, wgpu::WgpuDevice};
    pub type WgpuBackend = Wgpu<f32, i32>;

    static DEVICE: LazyLock<WgpuDevice> = LazyLock::new(Default::default);

    pub fn get_device() -> &'static WgpuDevice {
        &DEVICE
    }
}

#[cfg(feature = "rocm")]
pub mod rocm {
    use std::sync::LazyLock;

    use burn::backend::{Rocm, rocm::RocmDevice};
    pub type RocmBackend = Rocm<f32, i32>;

    static DEVICE: LazyLock<RocmDevice> = LazyLock::new(Default::default);

    pub fn get_device() -> &'static RocmDevice {
        &DEVICE
    }
}

#[cfg(feature = "cuda")]
pub mod cuda {
    use std::sync::LazyLock;

    use burn::backend::{Cuda, cuda::CudaDevice};
    pub type CudaBackend = Cuda<f32, i32>;

    static DEVICE: LazyLock<CudaDevice> = LazyLock::new(Default::default);

    pub fn get_device() -> &'static CudaDevice {
        &DEVICE
    }
}
