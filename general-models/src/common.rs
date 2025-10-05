use burn::{Tensor, config::Config, module::Module, nn::{BatchNorm, BatchNormConfig, GroupNorm, GroupNormConfig, InstanceNorm, InstanceNormConfig, LayerNorm, LayerNormConfig, RmsNorm, RmsNormConfig}, prelude::Backend, tensor::activation::{gelu, relu, sigmoid, tanh}};
use delegate::delegate;
use derive_more::From;
use serde::{Deserialize, Serialize};

use crate::{FromConfig, SimpleForwardable};

macro_rules! default_f {
    ($ident: ident, $ty: ty, $expr: expr) => {
        fn $ident() -> $ty {
            $expr
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Module)]
pub enum Activation {
    Relu,
    #[default]
    Gelu,
    Sigmoid,
    Tanh,
}

// impl<B: Backend> Module<B> for Activation {
//     type Record = ConstantRecord;

//     fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
//         devices
//     }

//     fn fork(self, _: &B::Device) -> Self {
//         self
//     }

//     fn to_device(self, _: &B::Device) -> Self {
//         self
//     }

//     fn visit<Visitor: ModuleVisitor<B>>(&self, _: &mut Visitor) {
//     }

//     fn map<Mapper: ModuleMapper<B>>(self, _: &mut Mapper) -> Self {
//         self
//     }

//     fn load_record(self, _: Self::Record) -> Self {
//         self
//     }

//     fn into_record(self) -> Self::Record {
//         ConstantRecord
//     }
// }

impl<B: Backend, const D: usize> SimpleForwardable<B, D, D> for Activation {
    fn forward(&self, tensor: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Activation::Relu => relu(tensor),
            Activation::Gelu => gelu(tensor),
            Activation::Sigmoid => sigmoid(tensor),
            Activation::Tanh => tanh(tensor),
        }
    }
}

impl<B: Backend> FromConfig<B> for Activation {
    type Config = ActivationConfig;

    fn init(config: Self::Config, _: &B::Device) -> Self {
        config.init()
    }
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq, Eq, Clone, Copy)]
pub enum ActivationConfig {
    Relu,
    #[default]
    Gelu,
    Sigmoid,
    Tanh
}

impl ActivationConfig {
    pub fn init(self) -> Activation {
        match self {
            ActivationConfig::Relu => Activation::Relu,
            ActivationConfig::Gelu => Activation::Gelu,
            ActivationConfig::Sigmoid => Activation::Sigmoid,
            ActivationConfig::Tanh => Activation::Tanh,
        }
    }
}

impl Config for ActivationConfig {}

#[derive(Debug, From, Module)]
pub enum Norm<B: Backend> {
    BatchNorm(BatchNorm<B>),
    RmsNorm(RmsNorm<B>),
    LayerNorm(LayerNorm<B>),
    GroupNorm(GroupNorm<B>),
    InstanceNorm(InstanceNorm<B>),
}

impl<B: Backend, const D: usize> SimpleForwardable<B, D, D> for Norm<B> {
    delegate! {
        to match self {
            Norm::BatchNorm(x) => x,
            Norm::RmsNorm(x) => x,
            Norm::LayerNorm(x) => x,
            Norm::GroupNorm(x) => x,
            Norm::InstanceNorm(x) => x,
        } {
            fn forward(&self, tensor: Tensor<B, D>) -> Tensor<B, D>;
        }
    }
}

// impl<B: Backend> Module<B> for Norm<B> {
//     type Record = NormRecord<B>;

//     delegate! {
//         to match self {
//             Norm::BatchNorm(x) => x,
//             Norm::RmsNorm(x) => x,
//             Norm::LayerNorm(x) => x,
//             Norm::GroupNorm(x) => x,
//             Norm::InstanceNorm(x) => x,
//         } {
//             fn collect_devices(&self, devices: Devices<B>) -> Devices<B>;
//             #[into]
//             fn fork(self, device: &<B as Backend>::Device) -> Self;
//             #[into]
//             fn to_device(self, device: &<B as Backend>::Device) -> Self;
//             fn visit<Visitor: ModuleVisitor<B>>(&self, visitor: &mut Visitor);
//             #[into]
//             fn map<Mapper: ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self;
//             #[into]
//             fn into_record(self) -> Self::Record;
//         }
//     }

//     fn load_record(self, record: Self::Record) -> Self {
//         macro_rules! unwrap {
//             ($ident: ident) => {{
//                 let NormRecord::$ident(x) = record else { panic!("Unexpected record type") };
//                 x
//             }};
//         }
//         match self {
//             Norm::BatchNorm(x) => x.load_record(unwrap!(BatchNormRecord)).into(),
//             Norm::RmsNorm(x) => x.load_record(unwrap!(RmsNormRecord)).into(),
//             Norm::LayerNorm(x) => x.load_record(unwrap!(LayerNormRecord)).into(),
//             Norm::GroupNorm(x) => x.load_record(unwrap!(GroupNormRecord)).into(),
//             Norm::InstanceNorm(x) => x.load_record(unwrap!(InstanceNormRecord)).into(),
//         }
//     }
// }

// #[derive(From, Debug, Clone)]
// pub enum NormInnerModule<B: AutodiffBackend> {
//     BatchNorm(<BatchNorm<B> as AutodiffModule<B>>::InnerModule),
//     RmsNorm(<RmsNorm<B> as AutodiffModule<B>>::InnerModule),
//     LayerNorm(<LayerNorm<B> as AutodiffModule<B>>::InnerModule),
//     GroupNorm(<GroupNorm<B> as AutodiffModule<B>>::InnerModule),
//     InstanceNorm(<InstanceNorm<B> as AutodiffModule<B>>::InnerModule),
// }

// impl<B: AutodiffBackend> Module<B::InnerBackend> for NormInnerModule<B> {
//     type Record = NormRecord<B::InnerBackend>;

//     delegate! {
//         to match self {
//             NormInnerModule::BatchNorm(x) => x,
//             NormInnerModule::RmsNorm(x) => x,
//             NormInnerModule::LayerNorm(x) => x,
//             NormInnerModule::GroupNorm(x) => x,
//             NormInnerModule::InstanceNorm(x) => x,
//         } {
//             fn collect_devices(&self, devices: Devices<B>) -> Devices<B>;
//             #[into]
//             fn fork(self, device: &<B as Backend>::Device) -> Self;
//             #[into]
//             fn to_device(self, device: &<B::InnerBackend as Backend>::Device) -> Self;
//             fn visit<Visitor: ModuleVisitor<B::InnerBackend>>(&self, visitor: &mut Visitor);
//             #[into]
//             fn map<Mapper: ModuleMapper<B::InnerBackend>>(self, mapper: &mut Mapper) -> Self;
//             #[into]
//             fn into_record(self) -> Self::Record;
//         }
//     }

//     fn load_record(self, record: Self::Record) -> Self {
//         macro_rules! unwrap {
//             ($ident: ident) => {{
//                 let NormRecord::$ident(x) = record else { panic!("Unexpected record type") };
//                 x
//             }};
//         }
//         match self {
//             NormInnerModule::BatchNorm(x) => x.load_record(unwrap!(BatchNormRecord)).into(),
//             NormInnerModule::RmsNorm(x) => x.load_record(unwrap!(RmsNormRecord)).into(),
//             NormInnerModule::LayerNorm(x) => x.load_record(unwrap!(LayerNormRecord)).into(),
//             NormInnerModule::GroupNorm(x) => x.load_record(unwrap!(GroupNormRecord)).into(),
//             NormInnerModule::InstanceNorm(x) => x.load_record(unwrap!(InstanceNormRecord)).into(),
//         }
//     }
// }

// impl<B: AutodiffBackend> AutodiffModule<B> for Norm<B> {
//     type InnerModule = NormInnerModule<B>;

//     fn valid(&self) -> Self::InnerModule {
//         todo!()
//     }
// }

// #[derive(From)]
// pub enum NormRecord<B: Backend> {
//     BatchNormRecord(BatchNormRecord<B>),
//     RmsNormRecord(RmsNormRecord<B>),
//     LayerNormRecord(LayerNormRecord<B>),
//     GroupNormRecord(GroupNormRecord<B>),
//     InstanceNormRecord(InstanceNormRecord<B>),
// }

// impl<B: Backend> Record<B> for NormRecord<B> {
//     type Item<S: PrecisionSettings> = NormRecordItem<B, S>;

//     delegate! {
//         to match self {
//             NormRecord::BatchNormRecord(x) => x,
//             NormRecord::RmsNormRecord(x) => x,
//             NormRecord::LayerNormRecord(x) => x,
//             NormRecord::GroupNormRecord(x) => x,
//             NormRecord::InstanceNormRecord(x) => x,
//         } {
//             #[into]
//             fn into_item<S: PrecisionSettings>(self) -> Self::Item<S>;
//         }
//     }
    
//     fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &<B as Backend>::Device) -> Self {
//         match item {
//             NormRecordItem::BatchNormRecordItem(x) => BatchNormRecord::from_item(x, device).into(),
//             NormRecordItem::RmsNormRecordItem(x) => RmsNormRecord::from_item(x, device).into(),
//             NormRecordItem::LayerNormRecordItem(x) => LayerNormRecord::from_item(x, device).into(),
//             NormRecordItem::GroupNormRecordItem(x) => GroupNormRecord::from_item(x, device).into(),
//             NormRecordItem::InstanceNormRecordItem(x) => InstanceNormRecord::from_item(x, device).into(),
//         }
//     }
// }

// #[derive(From, Serialize, Deserialize)]
// #[serde(bound = "")]
// pub enum NormRecordItem<B: Backend, S: PrecisionSettings> {
//     BatchNormRecordItem(BatchNormRecordItem<B, S>),
//     RmsNormRecordItem(RmsNormRecordItem<B, S>),
//     LayerNormRecordItem(LayerNormRecordItem<B, S>),
//     GroupNormRecordItem(GroupNormRecordItem<B, S>),
//     InstanceNormRecordItem(InstanceNormRecordItem<B, S>),
// }


#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum NormConfig {
    BatchNorm {
        #[serde(default = "default_epsilon")]
        epsilon: f64,
        #[serde(default = "default_momentum")]
        momentum: f64
    },
    RmsNorm {
        #[serde(default = "default_epsilon")]
        epsilon: f64,
    },
    LayerNorm {
        #[serde(default = "default_epsilon")]
        epsilon: f64,
    },
    GroupNorm {
        num_groups: usize,
        #[serde(default = "default_epsilon")]
        epsilon: f64,
        #[serde(default = "default_affine")]
        affine: bool
    },
    InstanceNorm {
        #[serde(default = "default_epsilon")]
        epsilon: f64,
        #[serde(default = "default_affine")]
        affine: bool
    },
}

impl NormConfig {
    pub fn init<B: Backend>(self, device: &B::Device, input_size: usize) -> Norm<B> {
        match self {
            NormConfig::BatchNorm { epsilon, momentum } => Norm::BatchNorm(BatchNormConfig { num_features: input_size, epsilon, momentum }.init(device)),
            NormConfig::RmsNorm { epsilon } => Norm::RmsNorm(RmsNormConfig { d_model: input_size, epsilon }.init(device)),
            NormConfig::LayerNorm { epsilon } => Norm::LayerNorm(LayerNormConfig { d_model: input_size, epsilon }.init(device)),
            NormConfig::GroupNorm { num_groups, epsilon, affine } => Norm::GroupNorm(GroupNormConfig { num_channels: input_size, num_groups, epsilon, affine }.init(device)),
            NormConfig::InstanceNorm { epsilon, affine } => Norm::InstanceNorm(InstanceNormConfig { num_channels: input_size, epsilon, affine }.init(device)),
        }
    }
}

impl Config for NormConfig {}

default_f!(default_epsilon, f64, 1e-5);
default_f!(default_momentum, f64, 0.1);
default_f!(default_affine, bool, true);