use burn::{Tensor, module::Module, prelude::Backend};

pub mod autoencoder;
pub mod serde_fix;

// #[derive(Debug, Clone)]
// struct TrainingShim<T> {
//     model: T,
// }

// struct TrainingShimRecord<T> {
//     model_record: T,
// }

// #[derive(Deserialize, Serialize)]
// struct TrainingShimRecordItem<T> {
//     model_item: T,
// }

// impl<B: Backend, T: Record<B>> Record<B> for TrainingShimRecord<T> {
//     type Item<S: burn::record::PrecisionSettings> = TrainingShimRecordItem<T::Item<S>>;

//     fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
//         TrainingShimRecordItem {
//             model_item: self.model_record.into_item(),
//         }
//     }

//     fn from_item<S: burn::record::PrecisionSettings>(
//         item: Self::Item<S>,
//         device: &<B as Backend>::Device,
//     ) -> Self {
//         Self {
//             model_record: Record::from_item(item.model_item, device),
//         }
//     }
// }

// impl<B: Backend, T: Module<B>> Module<B> for TrainingShim<T> {
//     type Record = TrainingShimRecord<T::Record>;

//     fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
//         self.model.collect_devices(devices)
//     }

//     fn fork(self, device: &<B as Backend>::Device) -> Self {
//         Self {
//             model: self.model.fork(device),
//         }
//     }

//     fn to_device(self, device: &<B as Backend>::Device) -> Self {
//         Self {
//             model: self.model.to_device(device),
//         }
//     }

//     fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
//         self.model.visit(visitor);
//     }

//     fn map<Mapper: burn::module::ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
//         Self {
//             model: self.model.map(mapper),
//         }
//     }

//     fn load_record(self, record: Self::Record) -> Self {
//         Self {
//             model: self.model.load_record(record.model_record),
//         }
//     }

//     fn into_record(self) -> Self::Record {
//         TrainingShimRecord {
//             model_record: self.model.into_record(),
//         }
//     }
// }

// pub trait LoadTraining<B: Backend>: Module<B> {
//     fn load_compact_training(
//         &mut self,
//         path: impl AsRef<Path>,
//         device: &B::Device,
//     ) -> Result<(), RecorderError> {
//         let record = CompactRecorder::new().load(path.as_ref().into(), device)?;
//         unsafe {
//             let tmp = TrainingShim {
//                 model: std::ptr::read(self),
//             }
//             .load_record(record);
//             std::ptr::write(self, tmp.model);
//         }
//         Ok(())
//     }

//     #[cfg(feature = "wgpu")]
//     fn load_compact_training_wgpu(&mut self, path: impl AsRef<Path>) -> Result<(), RecorderError> {
//         self.load_compact_training(path, wgpu::get_device())
//     }
// }

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
