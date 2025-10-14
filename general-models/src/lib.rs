use burn::{Tensor, prelude::Backend};

pub mod common;
pub mod composite;
pub mod conv;
pub mod error;
pub mod linear;

pub trait Init<B: Backend, T> {
    fn init(self, device: &B::Device) -> T;
}

pub trait SimpleInfer<B: Backend, const N_I: usize, const N_O: usize> {
    fn forward(&self, tensor: Tensor<B, N_I>) -> Tensor<B, N_O>;
    fn infer(&self, tensor: Tensor<B, N_I>) -> Tensor<B, N_O> {
        self.forward(tensor)
    }
}

pub trait SimpleTrain<B: Backend, const N_I: usize, const N_O: usize> {
    fn forward(&self, tensor: Tensor<B, N_I>) -> Tensor<B, N_O>;
    fn train(&self, tensor: Tensor<B, N_I>) -> Tensor<B, N_O> {
        self.forward(tensor)
    }
}

impl<B: Backend, const N_I: usize, const N_O: usize, T: SimpleInfer<B, N_I, N_O>> SimpleInfer<B, N_I, N_O> for &T {
    fn forward(&self, tensor: Tensor<B, N_I>) -> Tensor<B, N_O> {
        <T as SimpleInfer<B, N_I, N_O>>::forward(self, tensor)
    }
}

impl<B: Backend, const N_I: usize, const N_O: usize, T: SimpleTrain<B, N_I, N_O>> SimpleTrain<B, N_I, N_O> for &T {
    fn forward(&self, tensor: Tensor<B, N_I>) -> Tensor<B, N_O> {
        <T as SimpleTrain<B, N_I, N_O>>::forward(self, tensor)
    }
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

pub mod cpu {
    use std::sync::LazyLock;

    use burn::backend::{NdArray, ndarray::NdArrayDevice};
    pub type NdArrayBackend = NdArray<f32, i32>;

    static DEVICE: LazyLock<NdArrayDevice> = LazyLock::new(Default::default);

    pub fn get_device() -> &'static NdArrayDevice {
        &DEVICE
    }
}
