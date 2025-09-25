use burn::{module::Module, nn::loss::{MseLoss, Reduction}, prelude::Backend, record::Record, train::RegressionOutput, Tensor};
use general_models::{autoencoder::SimpleAutoEncoder, SimpleForwardable};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct TrainableModel<T> {
    pub model: T
}

#[derive(Debug, Clone)]
pub struct TrainableModelRecord<T> {
    model_record: T
}


#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TrainableModelRecordItem<T> {
    model_item: T
}


impl<B: Backend, T: Record<B>> Record<B> for TrainableModelRecord<T> {
    type Item<S: burn::record::PrecisionSettings> = TrainableModelRecordItem<T::Item<S>>;

    fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
        TrainableModelRecordItem {
            model_item: self.model_record.into_item()
        }
    }

    fn from_item<S: burn::record::PrecisionSettings>(item: Self::Item<S>, device: &<B as Backend>::Device) -> Self {
        Self {
            model_record: Record::from_item(item.model_item, device)
        }
    }
}


impl<B: Backend, T: Module<B>> Module<B> for TrainableModel<T> {
    type Record = TrainableModelRecord<T::Record>;

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        self.model.collect_devices(devices)
    }

    fn fork(self, device: &<B as Backend>::Device) -> Self {
        Self {
            model: self.model.fork(device)
        }
    }

    fn to_device(self, device: &<B as Backend>::Device) -> Self {
        Self {
            model: self.model.to_device(device)
        }
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
        self.model.visit(visitor);
    }

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
        Self {
            model: self.model.map(mapper)
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            model: self.model.load_record(record.model_record)
        }
    }

    fn into_record(self) -> Self::Record {
        TrainableModelRecord {
            model_record: self.model.into_record()
        }
    }
}


pub trait RegressionTrainable<B: Backend, const N_I: usize, const N_O: usize>: Module<B> {
    fn forward_regression(&self, input: Tensor<B, N_I>, expected: Tensor<B, N_O>) -> RegressionOutput<B>;
}


impl<B, const N_I: usize, const N_D: usize, E, D> RegressionTrainable<B, N_I, N_I> for SimpleAutoEncoder<B, E, D, N_I, N_D>
where
    B: Backend,
    Self: SimpleForwardable<B, N_I, N_I>
{
    fn forward_regression(&self, input: Tensor<B, N_I>, expected: Tensor<B, N_I>) -> RegressionOutput<B> {
        let batch_size = input.dims()[0];
        let actual = self.forward(input.clone());
        let loss = MseLoss::new().forward(actual.clone(), expected.clone(), Reduction::Mean);
    
        RegressionOutput::new(
            loss,
            actual.reshape([batch_size as i32, -1]),
            expected.reshape([batch_size as i32, -1]),
        )
    }
}
