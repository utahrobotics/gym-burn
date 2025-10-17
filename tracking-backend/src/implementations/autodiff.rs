use burn::tensor::{
    backend::AutodiffBackend,
    ops::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor},
};

use crate::{InnerAutodiffBackend, InnerBackend, TrackingBackend};

impl AutodiffBackend for TrackingBackend {
    type InnerBackend = InnerBackend;

    type Gradients = <InnerAutodiffBackend as AutodiffBackend>::Gradients;

    fn backward(tensor: FloatTensor<Self>) -> Self::Gradients {
        InnerAutodiffBackend::backward(tensor)
    }

    fn grad(
        tensor: &FloatTensor<Self>,
        grads: &Self::Gradients,
    ) -> Option<FloatTensor<Self::InnerBackend>> {
        InnerAutodiffBackend::grad(tensor, grads)
    }

    fn grad_remove(
        tensor: &FloatTensor<Self>,
        grads: &mut Self::Gradients,
    ) -> Option<FloatTensor<Self::InnerBackend>> {
        InnerAutodiffBackend::grad_remove(tensor, grads)
    }

    fn grad_replace(
        tensor: &FloatTensor<Self>,
        grads: &mut Self::Gradients,
        grad: FloatTensor<Self::InnerBackend>,
    ) {
        InnerAutodiffBackend::grad_replace(tensor, grads, grad)
    }

    fn inner(tensor: FloatTensor<Self>) -> FloatTensor<Self::InnerBackend> {
        InnerAutodiffBackend::inner(tensor)
    }

    fn int_inner(tensor: IntTensor<Self>) -> IntTensor<Self::InnerBackend> {
        InnerAutodiffBackend::int_inner(tensor)
    }

    fn bool_inner(tensor: BoolTensor<Self>) -> BoolTensor<Self::InnerBackend> {
        InnerAutodiffBackend::bool_inner(tensor)
    }

    fn q_inner(tensor: QuantizedTensor<Self>) -> QuantizedTensor<Self::InnerBackend> {
        InnerAutodiffBackend::q_inner(tensor)
    }

    fn from_inner(tensor: FloatTensor<Self::InnerBackend>) -> FloatTensor<Self> {
        InnerAutodiffBackend::from_inner(tensor)
    }

    fn int_from_inner(tensor: IntTensor<Self::InnerBackend>) -> IntTensor<Self> {
        InnerAutodiffBackend::int_from_inner(tensor)
    }

    fn bool_from_inner(tensor: BoolTensor<Self::InnerBackend>) -> BoolTensor<Self> {
        InnerAutodiffBackend::bool_from_inner(tensor)
    }

    fn q_from_inner(tensor: QuantizedTensor<Self::InnerBackend>) -> QuantizedTensor<Self> {
        InnerAutodiffBackend::q_from_inner(tensor)
    }
}
