use burn::{
    lr_scheduler::{
        constant::ConstantLr,
        exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig},
        linear::{LinearLrScheduler, LinearLrSchedulerConfig},
        step::{StepLrScheduler, StepLrSchedulerConfig},
    },
    prelude::Backend,
    record::Record,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub enum LrScheduler {
    Constant(ConstantLr),
    Step(StepLrScheduler),
    Linear(LinearLrScheduler),
    Exponential(ExponentialLrScheduler),
}

impl burn::lr_scheduler::LrScheduler for LrScheduler {
    type Record<B: Backend> = LrSchedulerRecord;

    fn step(&mut self) -> burn::optim::LearningRate {
        match self {
            LrScheduler::Constant(x) => x.step(),
            LrScheduler::Step(x) => x.step(),
            LrScheduler::Linear(x) => x.step(),
            LrScheduler::Exponential(x) => x.step(),
        }
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        match self {
            LrScheduler::Constant(_) => LrSchedulerRecord::Constant,
            LrScheduler::Step(x) => LrSchedulerRecord::Step(x.to_record::<B>()),
            LrScheduler::Linear(x) => LrSchedulerRecord::Linear(x.to_record::<B>()),
            LrScheduler::Exponential(x) => LrSchedulerRecord::Exponential(x.to_record::<B>()),
        }
    }

    fn load_record<B: Backend>(self, record: Self::Record<B>) -> Self {
        macro_rules! unwrap {
            ($ident: ident) => {{
                let LrSchedulerRecord::$ident(x) = record else {
                    panic!("Unexpected record for LrScheduler");
                };
                x
            }};
        }
        match self {
            LrScheduler::Constant(_) => {
                assert_eq!(
                    record,
                    LrSchedulerRecord::Constant,
                    "Unexpected record for LrScheduler"
                );
                self
            }
            LrScheduler::Step(x) => Self::Step(x.load_record::<B>(unwrap!(Step))),
            LrScheduler::Linear(x) => Self::Linear(x.load_record::<B>(unwrap!(Linear))),
            LrScheduler::Exponential(x) => {
                Self::Exponential(x.load_record::<B>(unwrap!(Exponential)))
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LrSchedulerConfig {
    Constant(f64),
    Step {
        initial_lr: f64,
        step_size: usize,
    },
    Linear {
        initial_lr: f64,
        final_lr: f64,
        num_iters: usize,
    },
    Exponential {
        initial_lr: f64,
        gamma: f64,
    },
}

impl LrSchedulerConfig {
    pub fn init(self) -> LrScheduler {
        match self {
            LrSchedulerConfig::Constant(lr) => LrScheduler::Constant(ConstantLr::new(lr)),
            LrSchedulerConfig::Step {
                initial_lr,
                step_size,
            } => LrScheduler::Step(
                StepLrSchedulerConfig::new(initial_lr, step_size)
                    .init()
                    .expect("Invalid Step LR Scheduler"),
            ),
            LrSchedulerConfig::Linear {
                initial_lr,
                final_lr,
                num_iters,
            } => LrScheduler::Linear(
                LinearLrSchedulerConfig::new(initial_lr, final_lr, num_iters)
                    .init()
                    .expect("Invalid Linear LR Scheduler"),
            ),
            LrSchedulerConfig::Exponential { initial_lr, gamma } => LrScheduler::Exponential(
                ExponentialLrSchedulerConfig::new(initial_lr, gamma)
                    .init()
                    .expect("Invalid Exponential LR Scheduler"),
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LrSchedulerRecord {
    Constant,
    Step(i32),
    Linear(usize),
    Exponential(f64),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LrSchedulerRecordItem {
    Constant,
    Step(i32),
    Linear(usize),
    Exponential(f64),
}

impl<B: Backend> Record<B> for LrSchedulerRecord {
    type Item<S: burn::record::PrecisionSettings> = LrSchedulerRecordItem;

    fn from_item<S: burn::record::PrecisionSettings>(
        item: Self::Item<S>,
        _: &<B as Backend>::Device,
    ) -> Self {
        match item {
            LrSchedulerRecordItem::Constant => Self::Constant,
            LrSchedulerRecordItem::Step(x) => Self::Step(x),
            LrSchedulerRecordItem::Linear(x) => Self::Linear(x),
            LrSchedulerRecordItem::Exponential(x) => Self::Exponential(x),
        }
    }

    fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
        match self {
            LrSchedulerRecord::Constant => LrSchedulerRecordItem::Constant,
            LrSchedulerRecord::Step(x) => LrSchedulerRecordItem::Step(x),
            LrSchedulerRecord::Linear(x) => LrSchedulerRecordItem::Linear(x),
            LrSchedulerRecord::Exponential(x) => LrSchedulerRecordItem::Exponential(x),
        }
    }
}
