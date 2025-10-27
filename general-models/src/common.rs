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
use serde::{Deserialize, Serialize};
use utils::default_f;

use crate::SimpleInfer;

#[derive(Debug, derive_more::From, Module)]
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

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "snake_case")]
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
    #[default]
    None,
}

impl NormConfig {
    pub fn init<B: Backend>(self, device: &B::Device, input_size: usize) -> Option<Norm<B>> {
        Some(match self {
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
            NormConfig::None => return None,
        })
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
    fn content(&self, content: burn::module::Content) -> Option<burn::module::Content> {
        Some(content)
    }
}

use burn::nn::{HardSigmoidConfig, LeakyReluConfig, PReluConfig, SwiGluConfig};

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "snake_case")]
pub enum ActivationConfig {
    Gelu,
    PRelu(PReluConfig),
    Relu,
    LeakyRelu(LeakyReluConfig),
    SwiGlu(SwiGluConfig),
    Sigmoid,
    Tanh,
    HardSigmoid(HardSigmoidConfig),
    #[default]
    None,
}

impl ActivationConfig {
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> Option<burn::nn::activation::Activation<B>> {
        use burn::nn::activation::ActivationConfig as Config;
        let config = match self {
            ActivationConfig::Gelu => Config::Gelu,
            ActivationConfig::PRelu(prelu_config) => Config::PRelu(prelu_config),
            ActivationConfig::Relu => Config::Relu,
            ActivationConfig::LeakyRelu(leaky_relu_config) => Config::LeakyRelu(leaky_relu_config),
            ActivationConfig::SwiGlu(swi_glu_config) => Config::SwiGlu(swi_glu_config),
            ActivationConfig::Sigmoid => Config::Sigmoid,
            ActivationConfig::Tanh => Config::Tanh,
            ActivationConfig::HardSigmoid(hard_sigmoid_config) => {
                Config::HardSigmoid(hard_sigmoid_config)
            }
            ActivationConfig::None => return None,
        };
        Some(config.init(device))
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

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(untagged)]
pub enum Either<A, B, C> {
    One(A),
    Two(A, B),
    Three(A, B, C),
}

impl<A, B, C> Either<A, B, C> {
    pub fn into_tuple(self) -> (A, Option<B>, Option<C>) {
        match self {
            Either::One(a) => (a, None, None),
            Either::Two(a, b) => (a, Some(b), None),
            Either::Three(a, b, c) => (a, Some(b), Some(c)),
        }
    }
}

default_f!(default_epsilon, f64, 1e-5);
default_f!(default_momentum, f64, 0.1);
default_f!(default_affine, bool, true);
