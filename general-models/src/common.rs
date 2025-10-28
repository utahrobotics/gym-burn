use std::{fmt::Debug, marker::PhantomData};

use burn::{
    Tensor,
    config::Config,
    module::{AutodiffModule, ConstantRecord, Module, ModuleDisplay, ModuleDisplayDefault},
    nn::{
        BatchNorm, BatchNormConfig, GroupNorm, GroupNormConfig, Initializer, InstanceNorm,
        InstanceNormConfig, LayerNorm, LayerNormConfig, RmsNorm, RmsNormConfig,
        activation::Activation,
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

#[derive(Debug, Serialize, Deserialize, Clone)]
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
}

impl ActivationConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> burn::nn::activation::Activation<B> {
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
        };
        config.init(device)
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
pub enum Either<A, B, C, D> {
    One(A),
    Two(A, B),
    Three(A, B, C),
    Four(A, B, C, D),
}

impl<A, B: Default, C: Default, D: Default> Either<A, B, C, D> {
    pub fn into_tuple(self) -> (A, B, C, D) {
        match self {
            Either::One(a) => (a, B::default(), C::default(), D::default()),
            Either::Two(a, b) => (a, b, C::default(), D::default()),
            Either::Three(a, b, c) => (a, b, c, D::default()),
            Either::Four(a, b, c, d) => (a, b, c, d),
        }
    }
}

/// A slightly more advanced `Option` that adds a default variant.
/// 
/// The [`Optional::Default`] variant is now the default instead of [`Optional::None`].
/// What this means is omitting this enum, writing `null` in JSON, or writing "default",
/// will result in [`Optional::Default`], which delegates the value to `default_f` in [`Optional::resolve`].
/// 
/// Explicitly writing `"none"` will lead to [`Optional::Default`], which always returns [`Option::None`]
/// in [`Optional::resolve`].
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default)]
#[serde(rename_all = "snake_case")]
pub enum Optional<T> {
    None,
    #[default]
    Default,
    #[serde(untagged)]
    Inner(T),
}

impl<T> Optional<T> {
    pub fn resolve(self, default_f: impl FnOnce() -> Option<T>) -> Option<T> {
        match self {
            Optional::Inner(inner) => Some(inner),
            Optional::None => None,
            Optional::Default => default_f(),
        }
    }
}

pub(crate) fn handle_norm_activation<B: Backend>(
    norm: Optional<NormConfig>,
    activation: Optional<ActivationConfig>,
    default_norm: &Option<NormConfig>,
    default_activation: &Option<ActivationConfig>,
    default_weights_gain: Option<f64>,
    input_size: usize,
    device: &B::Device,
) -> (Option<Norm<B>>, Option<Activation<B>>, Initializer) {
    let norm = norm
        .resolve(|| default_norm.clone())
        .map(|norm| norm.init(device, input_size))
        .flatten();
    let activation = activation
        .resolve(|| default_activation.clone())
        .map(|x| x.init(device));

    let init = match &activation {
        Some(Activation::Gelu(_) | Activation::Relu(_) | Activation::PRelu(_)) => {
            Initializer::KaimingNormal {
                gain: default_weights_gain.unwrap_or(2.0f64.sqrt()),
                fan_out_only: false,
            }
        }
        Some(Activation::LeakyRelu(relu)) => Initializer::KaimingNormal {
            gain: default_weights_gain
                .unwrap_or_else(|| (2.0 / (1.0 + relu.negative_slope.powi(2))).sqrt()),
            fan_out_only: false,
        },
        Some(
            Activation::HardSigmoid(_)
            | Activation::Sigmoid(_)
            | Activation::Tanh(_)
            | Activation::SwiGlu(_),
        )
        | None => Initializer::XavierNormal {
            gain: default_weights_gain.unwrap_or(1.0),
        },
        // the compiler can't tell that all variants are handled
        Some(_) => unreachable!(),
    };

    (norm, activation, init)
}

default_f!(default_epsilon, f64, 1e-5);
default_f!(default_momentum, f64, 0.1);
default_f!(default_affine, bool, true);
