use std::{fmt::Debug, marker::PhantomData};

use burn::{
    Tensor,
    config::Config,
    module::{AutodiffModule, ConstantRecord, Module, ModuleDisplay, ModuleDisplayDefault},
    nn::{
        BatchNorm, BatchNormConfig, GroupNorm, GroupNormConfig, InstanceNorm, InstanceNormConfig,
        LayerNorm, LayerNormConfig, RmsNorm, RmsNormConfig,
    },
    prelude::Backend,
    tensor::backend::AutodiffBackend,
};
use delegate::delegate;
use derive_more::From;
use serde::{Deserialize, Serialize};

use crate::SimpleInfer;

#[macro_export]
macro_rules! default_f {
    ($ident: ident, $ty: ty, $expr: expr) => {
        fn $ident() -> $ty {
            $expr
        }
    };
}

#[derive(Debug, From, Module)]
pub enum Norm<B: Backend> {
    BatchNorm(BatchNorm<B>),
    RmsNorm(RmsNorm<B>),
    LayerNorm(LayerNorm<B>),
    GroupNorm(GroupNorm<B>),
    InstanceNorm(InstanceNorm<B>),
}

impl<B: Backend, const D: usize> SimpleInfer<B, D, D> for Norm<B> {
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum NormConfig {
    BatchNorm {
        #[serde(default = "default_epsilon")]
        epsilon: f64,
        #[serde(default = "default_momentum")]
        momentum: f64,
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
        affine: bool,
    },
    InstanceNorm {
        #[serde(default = "default_epsilon")]
        epsilon: f64,
        #[serde(default = "default_affine")]
        affine: bool,
    },
}

impl NormConfig {
    pub fn init<B: Backend>(self, device: &B::Device, input_size: usize) -> Norm<B> {
        match self {
            NormConfig::BatchNorm { epsilon, momentum } => Norm::BatchNorm(
                BatchNormConfig {
                    num_features: input_size,
                    epsilon,
                    momentum,
                }
                .init(device),
            ),
            NormConfig::RmsNorm { epsilon } => Norm::RmsNorm(
                RmsNormConfig {
                    d_model: input_size,
                    epsilon,
                }
                .init(device),
            ),
            NormConfig::LayerNorm { epsilon } => Norm::LayerNorm(
                LayerNormConfig {
                    d_model: input_size,
                    epsilon,
                }
                .init(device),
            ),
            NormConfig::GroupNorm {
                num_groups,
                epsilon,
                affine,
            } => Norm::GroupNorm(
                GroupNormConfig {
                    num_channels: input_size,
                    num_groups,
                    epsilon,
                    affine,
                }
                .init(device),
            ),
            NormConfig::InstanceNorm { epsilon, affine } => Norm::InstanceNorm(
                InstanceNormConfig {
                    num_channels: input_size,
                    epsilon,
                    affine,
                }
                .init(device),
            ),
        }
    }
}

impl Config for NormConfig {}

#[derive(Debug, Default, Clone, Copy)]
pub struct PhantomBackend<B: Backend>(PhantomData<fn() -> B>);

impl<B: Backend> Module<B> for PhantomBackend<B> {
    type Record = ConstantRecord;

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _: &<B as Backend>::Device) -> Self {
        self
    }

    fn to_device(self, _: &<B as Backend>::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {
        ConstantRecord
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for PhantomBackend<B> {
    type InnerModule = PhantomBackend<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        Default::default()
    }
}

impl<B: Backend> ModuleDisplay for PhantomBackend<B> {}

impl<B: Backend> ModuleDisplayDefault for PhantomBackend<B> {
    fn content(&self, _content: burn::module::Content) -> Option<burn::module::Content> {
        None
    }
}

// pub struct PhantomModule<B: Backend, T>(PhantomData<fn() -> (B, T)>);

// impl<B: Backend, T> Debug for PhantomModule<B, T> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_tuple("PhantomModule").field(&self.0).finish()
//     }
// }

// impl<B: Backend, T> Default for PhantomModule<B, T> {
//     fn default() -> Self {
//         Self(Default::default())
//     }
// }

// impl<B: Backend, T> Clone for PhantomModule<B, T> {
//     fn clone(&self) -> Self {
//         Self(self.0.clone())
//     }
// }

// impl<B: Backend, T> Copy for PhantomModule<B, T> {}

// impl<B: Backend, T> Module<B> for PhantomModule<B, T> {
//     type Record = ConstantRecord;

//     fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
//         devices
//     }

//     fn fork(self, _: &<B as Backend>::Device) -> Self {
//         self
//     }

//     fn to_device(self, _: &<B as Backend>::Device) -> Self {
//         self
//     }

//     fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _: &mut Visitor) {}

//     fn map<Mapper: burn::module::ModuleMapper<B>>(self, _: &mut Mapper) -> Self {
//         self
//     }

//     fn load_record(self, _: Self::Record) -> Self {
//         self
//     }

//     fn into_record(self) -> Self::Record {
//         ConstantRecord
//     }
// }

// impl<B: AutodiffBackend, T> AutodiffModule<B> for PhantomModule<B, T> {
//     type InnerModule = PhantomBackend<B::InnerBackend>;

//     fn valid(&self) -> Self::InnerModule {
//         Default::default()
//     }
// }

// impl<B: Backend, T> ModuleDisplay for PhantomModule<B, T> {}

// impl<B: Backend, T> ModuleDisplayDefault for PhantomModule<B, T> {
//     fn content(&self, _content: burn::module::Content) -> Option<burn::module::Content> {
//         None
//     }
// }

default_f!(default_epsilon, f64, 1e-5);
default_f!(default_momentum, f64, 0.1);
default_f!(default_affine, bool, true);
