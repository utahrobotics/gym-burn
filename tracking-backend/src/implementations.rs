use std::mem::transmute;

use burn::tensor::Device;
use burn::tensor::FloatDType;
use burn::tensor::IntDType;
use burn::tensor::Slice;
use burn::tensor::ops::ConvOptions;
use burn::tensor::ops::ConvTransposeOptions;
use burn::tensor::ops::DeformConvOptions;
use burn::tensor::ops::FloatElem;
use burn::tensor::ops::IntElem;
use burn::tensor::ops::InterpolateOptions;
use burn::tensor::ops::MaxPool2dBackward;
use burn::tensor::ops::MaxPool2dWithIndices;
use burn::tensor::ops::ModuleOps;
use burn::tensor::ops::TransactionOps;
use burn::tensor::ops::TransactionPrimitiveResult;

use burn::tensor::ops::TransactionPrimitive;

use burn::tensor::quantization::QuantizationParametersPrimitive;

use burn::tensor::quantization::QuantScheme;

use burn::tensor::ops::QuantizedTensor;

use burn::tensor::ops::QTensorOps;

use burn::tensor::ops::FloatTensorOps;

use burn::tensor::ops::DeformConv2dBackward;

use burn::tensor::ops::ActivationOps;

use burn::tensor::Distribution;

use burn::tensor::ops::IntTensorOps;

use burn::tensor::ops::FloatTensor;

use burn::tensor::ops::IntTensor;

use burn::prelude::TensorData;

use burn::tensor::ops::BoolTensor;

use burn::prelude::Shape;

use burn::tensor::ops::BoolTensorOps;
use serde_json::Value;
use serde_json::json;

use crate::tracking::finish_iter;
use crate::tracking::hash_tensor;
use crate::tracking::start_tracking_tensor;

use super::InnerAutodiffBackend;

use super::TrackingBackend;

use burn::prelude::Backend;

mod autodiff;

impl Backend for TrackingBackend {
    type Device = <InnerAutodiffBackend as Backend>::Device;

    type FloatTensorPrimitive = <InnerAutodiffBackend as Backend>::FloatTensorPrimitive;

    type FloatElem = <InnerAutodiffBackend as Backend>::FloatElem;

    type IntTensorPrimitive = <InnerAutodiffBackend as Backend>::IntTensorPrimitive;

    type IntElem = <InnerAutodiffBackend as Backend>::IntElem;

    type BoolTensorPrimitive = <InnerAutodiffBackend as Backend>::BoolTensorPrimitive;

    type BoolElem = <InnerAutodiffBackend as Backend>::BoolElem;

    type QuantizedTensorPrimitive = <InnerAutodiffBackend as Backend>::QuantizedTensorPrimitive;

    fn name(device: &Self::Device) -> String {
        InnerAutodiffBackend::name(device)
    }

    fn seed(device: &Self::Device, seed: u64) {
        InnerAutodiffBackend::seed(device, seed)
    }

    fn ad_enabled() -> bool {
        InnerAutodiffBackend::ad_enabled()
    }

    fn memory_persistent_allocations<Output, Input, Func: Fn(Input) -> Output>(
        device: &Self::Device,
        input: Input,
        func: Func,
    ) -> Output {
        InnerAutodiffBackend::memory_persistent_allocations(device, input, func)
    }

    fn memory_cleanup(device: &Self::Device) {
        InnerAutodiffBackend::memory_cleanup(device);
    }

    fn sync(device: &Self::Device) {
        InnerAutodiffBackend::sync(device);
    }
}

impl BoolTensorOps<TrackingBackend> for TrackingBackend {
    fn bool_empty(shape: Shape, device: &Device<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_empty(shape, device)
    }

    fn bool_zeros(shape: Shape, device: &Device<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_zeros(shape, device)
    }

    fn bool_ones(shape: Shape, device: &Device<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_ones(shape, device)
    }

    fn bool_into_data(
        tensor: BoolTensor<TrackingBackend>,
    ) -> impl Future<Output = TensorData> + Send {
        InnerAutodiffBackend::bool_into_data(tensor)
    }

    fn bool_from_data(
        data: TensorData,
        device: &Device<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_from_data(data, device)
    }

    fn bool_into_int(tensor: BoolTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_into_int(tensor)
    }

    fn bool_into_float(tensor: BoolTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_into_float(tensor)
    }

    fn bool_device(tensor: &BoolTensor<TrackingBackend>) -> Device<TrackingBackend> {
        InnerAutodiffBackend::bool_device(tensor)
    }

    fn bool_to_device(
        tensor: BoolTensor<TrackingBackend>,
        device: &Device<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_to_device(tensor, device)
    }

    fn bool_reshape(
        tensor: BoolTensor<TrackingBackend>,
        shape: Shape,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_reshape(tensor, shape)
    }

    fn bool_slice(
        tensor: BoolTensor<TrackingBackend>,
        slices: &[Slice],
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_slice(tensor, slices)
    }

    fn bool_slice_assign(
        tensor: BoolTensor<TrackingBackend>,
        slices: &[Slice],
        value: BoolTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_slice_assign(tensor, slices, value)
    }

    fn bool_equal(
        lhs: BoolTensor<TrackingBackend>,
        rhs: BoolTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_equal(lhs, rhs)
    }

    fn bool_not(tensor: BoolTensor<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_not(tensor)
    }

    fn bool_and(
        tensor: BoolTensor<TrackingBackend>,
        rhs: BoolTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_and(tensor, rhs)
    }

    fn bool_or(
        tensor: BoolTensor<TrackingBackend>,
        rhs: BoolTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_or(tensor, rhs)
    }

    fn bool_swap_dims(
        tensor: BoolTensor<TrackingBackend>,
        dim1: usize,
        dim2: usize,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_swap_dims(tensor, dim1, dim2)
    }

    fn bool_permute(
        tensor: BoolTensor<TrackingBackend>,
        axes: &[usize],
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_permute(tensor, axes)
    }

    fn bool_flip(
        tensor: BoolTensor<TrackingBackend>,
        axes: &[usize],
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_flip(tensor, axes)
    }

    fn bool_expand(
        tensor: BoolTensor<TrackingBackend>,
        shape: Shape,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_expand(tensor, shape)
    }

    fn bool_unfold(
        tensor: BoolTensor<TrackingBackend>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_unfold(tensor, dim, size, step)
    }

    fn bool_select(
        tensor: BoolTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_select(tensor, dim, indices)
    }

    fn bool_select_assign(
        tensor: BoolTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
        value: BoolTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_select_assign(tensor, dim, indices, value)
    }

    fn bool_repeat_dim(
        tensor: BoolTensor<TrackingBackend>,
        dim: usize,
        times: usize,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_repeat_dim(tensor, dim, times)
    }

    fn bool_cat(
        tensors: Vec<BoolTensor<TrackingBackend>>,
        dim: usize,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_cat(tensors, dim)
    }

    fn bool_not_equal(
        lhs: BoolTensor<TrackingBackend>,
        rhs: BoolTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_not_equal(lhs, rhs)
    }

    fn bool_xor(
        lhs: BoolTensor<TrackingBackend>,
        rhs: BoolTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_xor(lhs, rhs)
    }

    fn bool_transpose(tensor: BoolTensor<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_transpose(tensor)
    }

    fn bool_any(tensor: BoolTensor<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_any(tensor)
    }

    fn bool_any_dim(
        tensor: BoolTensor<TrackingBackend>,
        dim: usize,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_any_dim(tensor, dim)
    }

    fn bool_all(tensor: BoolTensor<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_all(tensor)
    }

    fn bool_all_dim(
        tensor: BoolTensor<TrackingBackend>,
        dim: usize,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::bool_all_dim(tensor, dim)
    }

    fn bool_argwhere(
        tensor: BoolTensor<TrackingBackend>,
    ) -> impl Future<Output = IntTensor<TrackingBackend>> + 'static + Send {
        InnerAutodiffBackend::bool_argwhere(tensor)
    }
}

impl IntTensorOps<TrackingBackend> for TrackingBackend {
    fn int_empty(
        shape: Shape,
        device: &Device<TrackingBackend>,
        dtype: IntDType,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_empty(shape, device, dtype)
    }

    fn int_into_data(
        tensor: IntTensor<TrackingBackend>,
    ) -> impl Future<Output = TensorData> + Send {
        InnerAutodiffBackend::int_into_data(tensor)
    }

    fn int_from_data(
        data: TensorData,
        device: &Device<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_from_data(data, device)
    }

    fn int_device(tensor: &IntTensor<TrackingBackend>) -> Device<TrackingBackend> {
        InnerAutodiffBackend::int_device(tensor)
    }

    fn int_to_device(
        tensor: IntTensor<TrackingBackend>,
        device: &Device<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_to_device(tensor, device)
    }

    fn int_reshape(tensor: IntTensor<TrackingBackend>, shape: Shape) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_reshape(tensor, shape)
    }

    fn int_slice(
        tensor: IntTensor<TrackingBackend>,
        slices: &[Slice],
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_slice(tensor, slices)
    }

    fn int_slice_assign(
        tensor: IntTensor<TrackingBackend>,
        slices: &[Slice],
        value: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_slice_assign(tensor, slices, value)
    }

    fn int_into_float(tensor: IntTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::int_into_float(tensor)
    }

    fn int_mask_where(
        tensor: IntTensor<TrackingBackend>,
        mask: BoolTensor<TrackingBackend>,
        source: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_mask_where(tensor, mask, source)
    }

    fn int_mask_fill(
        tensor: IntTensor<TrackingBackend>,
        mask: BoolTensor<TrackingBackend>,
        value: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_mask_fill(tensor, mask, value)
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_gather(dim, tensor, indices)
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
        value: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_scatter(dim, tensor, indices, value)
    }

    fn int_select(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_select(tensor, dim, indices)
    }

    fn int_select_assign(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
        value: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_select_assign(tensor, dim, indices, value)
    }

    fn int_equal(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_equal(lhs, rhs)
    }

    fn int_equal_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_equal_elem(lhs, rhs)
    }

    fn int_greater(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_greater(lhs, rhs)
    }

    fn int_greater_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_greater_elem(lhs, rhs)
    }

    fn int_greater_equal(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_greater_equal(lhs, rhs)
    }

    fn int_greater_equal_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_greater_equal_elem(lhs, rhs)
    }

    fn int_lower(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_lower(lhs, rhs)
    }

    fn int_lower_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_lower_elem(lhs, rhs)
    }

    fn int_lower_equal(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_lower_equal(lhs, rhs)
    }

    fn int_lower_equal_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_lower_equal_elem(lhs, rhs)
    }

    fn int_add(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_add(lhs, rhs)
    }

    fn int_add_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_add_scalar(lhs, rhs)
    }

    fn int_sub(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_sub(lhs, rhs)
    }

    fn int_sub_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_sub_scalar(lhs, rhs)
    }

    fn int_mul(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_mul(lhs, rhs)
    }

    fn int_mul_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_mul_scalar(lhs, rhs)
    }

    fn int_div(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_div(lhs, rhs)
    }

    fn int_div_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_div_scalar(lhs, rhs)
    }

    fn int_remainder(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_remainder(lhs, rhs)
    }

    fn int_remainder_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_remainder_scalar(lhs, rhs)
    }

    fn int_matmul(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_matmul(lhs, rhs)
    }

    fn int_sum(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_sum(tensor)
    }

    fn int_sum_dim(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_sum_dim(tensor, dim)
    }

    fn int_prod(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_prod(tensor)
    }

    fn int_prod_dim(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_prod_dim(tensor, dim)
    }

    fn int_mean_dim(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_mean_dim(tensor, dim)
    }

    fn int_cumsum(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_cumsum(tensor, dim)
    }

    fn int_argmax(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_argmax(tensor, dim)
    }

    fn int_argmin(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_argmin(tensor, dim)
    }

    fn int_abs(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_abs(tensor)
    }

    fn int_swap_dims(
        tensor: IntTensor<TrackingBackend>,
        dim1: usize,
        dim2: usize,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_swap_dims(tensor, dim1, dim2)
    }

    fn int_permute(
        tensor: IntTensor<TrackingBackend>,
        axes: &[usize],
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_permute(tensor, axes)
    }

    fn int_flip(tensor: IntTensor<TrackingBackend>, axes: &[usize]) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_flip(tensor, axes)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_random(shape, distribution, device)
    }

    fn int_expand(tensor: IntTensor<TrackingBackend>, shape: Shape) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_expand(tensor, shape)
    }

    fn bitwise_and(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::bitwise_and(lhs, rhs)
    }

    fn bitwise_and_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::bitwise_and_scalar(lhs, rhs)
    }

    fn bitwise_or(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::bitwise_or(lhs, rhs)
    }

    fn bitwise_or_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::bitwise_or_scalar(lhs, rhs)
    }

    fn bitwise_xor(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::bitwise_xor(lhs, rhs)
    }

    fn bitwise_xor_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::bitwise_xor_scalar(lhs, rhs)
    }

    fn bitwise_not(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::bitwise_not(tensor)
    }

    fn bitwise_left_shift(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::bitwise_left_shift(lhs, rhs)
    }

    fn bitwise_left_shift_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::bitwise_left_shift_scalar(lhs, rhs)
    }

    fn bitwise_right_shift(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::bitwise_right_shift(lhs, rhs)
    }

    fn bitwise_right_shift_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::bitwise_right_shift_scalar(lhs, rhs)
    }

    fn int_cast(tensor: IntTensor<TrackingBackend>, dtype: IntDType) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_cast(tensor, dtype)
    }

    fn int_unfold(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_unfold(tensor, dim, size, step)
    }

    fn int_repeat_dim(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
        times: usize,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_repeat_dim(tensor, dim, times)
    }

    fn int_cat(tensors: Vec<IntTensor<TrackingBackend>>, dim: usize) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_cat(tensors, dim)
    }

    fn int_not_equal(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_not_equal(lhs, rhs)
    }

    fn int_not_equal_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_not_equal_elem(lhs, rhs)
    }

    fn int_powi(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_powi(lhs, rhs)
    }

    fn int_powf(
        lhs: IntTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_powf(lhs, rhs)
    }

    fn int_powi_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_powi_scalar(lhs, rhs)
    }

    fn int_powf_scalar(lhs: IntTensor<TrackingBackend>, rhs: f32) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_powf_scalar(lhs, rhs)
    }

    fn int_clamp_min(
        tensor: IntTensor<TrackingBackend>,
        min: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_clamp_min(tensor, min)
    }

    fn int_clamp_max(
        tensor: IntTensor<TrackingBackend>,
        max: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_clamp_max(tensor, max)
    }

    fn int_clamp(
        tensor: IntTensor<TrackingBackend>,
        min: IntElem<TrackingBackend>,
        max: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_clamp(tensor, min, max)
    }

    fn int_neg(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_neg(tensor)
    }

    fn int_zeros(
        shape: Shape,
        device: &Device<TrackingBackend>,
        dtype: IntDType,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_zeros(shape, device, dtype)
    }

    fn int_ones(
        shape: Shape,
        device: &Device<TrackingBackend>,
        dtype: IntDType,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_ones(shape, device, dtype)
    }

    fn int_full(
        shape: Shape,
        fill_value: IntElem<TrackingBackend>,
        device: &Device<TrackingBackend>,
        dtype: IntDType,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_full(shape, fill_value, device, dtype)
    }

    fn int_mean(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_mean(tensor)
    }

    fn int_max(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_max(tensor)
    }

    fn int_max_dim(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_max_dim(tensor, dim)
    }

    fn int_max_dim_with_indices(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
    ) -> (IntTensor<TrackingBackend>, IntTensor<TrackingBackend>) {
        InnerAutodiffBackend::int_max_dim_with_indices(tensor, dim)
    }

    fn int_max_abs(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_max_abs(tensor)
    }

    fn int_max_abs_dim(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_max_abs_dim(tensor, dim)
    }

    fn int_min(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_min(tensor)
    }

    fn int_min_dim(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_min_dim(tensor, dim)
    }

    fn int_min_dim_with_indices(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
    ) -> (IntTensor<TrackingBackend>, IntTensor<TrackingBackend>) {
        InnerAutodiffBackend::int_min_dim_with_indices(tensor, dim)
    }

    fn int_transpose(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_transpose(tensor)
    }

    fn int_arange_step(
        range: std::ops::Range<i64>,
        step: usize,
        device: &Device<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_arange_step(range, step, device)
    }

    fn int_arange(
        range: std::ops::Range<i64>,
        device: &Device<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_arange(range, device)
    }

    fn int_any(tensor: IntTensor<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_any(tensor)
    }

    fn int_any_dim(tensor: IntTensor<TrackingBackend>, dim: usize) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_any_dim(tensor, dim)
    }

    fn int_all(tensor: IntTensor<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_all(tensor)
    }

    fn int_all_dim(tensor: IntTensor<TrackingBackend>, dim: usize) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::int_all_dim(tensor, dim)
    }

    fn int_sign(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_sign(tensor)
    }

    fn int_sort(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
        descending: bool,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_sort(tensor, dim, descending)
    }

    fn int_sort_with_indices(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
        descending: bool,
    ) -> (IntTensor<TrackingBackend>, IntTensor<TrackingBackend>) {
        InnerAutodiffBackend::int_sort_with_indices(tensor, dim, descending)
    }

    fn int_argsort(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
        descending: bool,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::int_argsort(tensor, dim, descending)
    }
}

impl ActivationOps<TrackingBackend> for TrackingBackend {
    fn leaky_relu(
        tensor: FloatTensor<TrackingBackend>,
        negative_slope: burn::tensor::ops::FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::leaky_relu(tensor, negative_slope)
    }

    fn relu(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::relu(tensor)
    }

    fn relu_backward(
        output: FloatTensor<TrackingBackend>,
        grad: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::relu_backward(output, grad)
    }

    fn gelu(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        let builder = start_tracking_tensor(&tensor, "gelu", Value::Null);
        builder.finish(InnerAutodiffBackend::gelu(tensor))
    }

    fn prelu(
        tensor: FloatTensor<TrackingBackend>,
        alpha: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::prelu(tensor, alpha)
    }

    fn gelu_backward(
        x: FloatTensor<TrackingBackend>,
        grad: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::gelu_backward(x, grad)
    }

    fn sigmoid(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::sigmoid(tensor)
    }

    fn sigmoid_backward(
        output: FloatTensor<TrackingBackend>,
        grad: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::sigmoid_backward(output, grad)
    }

    fn hard_sigmoid(
        tensor: FloatTensor<TrackingBackend>,
        alpha: burn::tensor::ops::FloatElem<TrackingBackend>,
        beta: burn::tensor::ops::FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::hard_sigmoid(tensor, alpha, beta)
    }

    fn log_sigmoid(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::log_sigmoid(tensor)
    }

    fn log_sigmoid_backward(
        x: FloatTensor<TrackingBackend>,
        grad: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::log_sigmoid_backward(x, grad)
    }
}

impl ModuleOps<TrackingBackend> for TrackingBackend {
    fn conv2d(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<TrackingBackend> {
        let builder = start_tracking_tensor(&x, "conv2d", json!({
            "weight_shape": weight.primitive.shape,
            "stride": options.stride,
            "padding": options.padding,
            "dilation": options.dilation,
            "groups": options.groups
        }));
        builder.finish(InnerAutodiffBackend::conv2d(x, weight, bias, options))
    }

    fn deform_conv2d(
        x: FloatTensor<TrackingBackend>,
        offset: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        mask: Option<FloatTensor<TrackingBackend>>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::deform_conv2d(x, offset, weight, mask, bias, options)
    }

    fn deform_conv2d_backward(
        x: FloatTensor<TrackingBackend>,
        offset: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        mask: Option<FloatTensor<TrackingBackend>>,
        bias: Option<FloatTensor<TrackingBackend>>,
        output_grad: FloatTensor<TrackingBackend>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<TrackingBackend> {
        let result = InnerAutodiffBackend::deform_conv2d_backward(
            x,
            offset,
            weight,
            mask,
            bias,
            output_grad,
            options,
        );
        DeformConv2dBackward {
            x_grad: result.x_grad,
            offset_grad: result.offset_grad,
            weight_grad: result.weight_grad,
            mask_grad: result.mask_grad,
            bias_grad: result.bias_grad,
        }
    }

    fn conv3d(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv3d(x, weight, bias, options)
    }

    fn conv_transpose2d(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<TrackingBackend> {
        let builder = start_tracking_tensor(&x, "conv_transpose2d", json!({
            "weight_shape": weight.primitive.shape,
            "stride": options.stride,
            "padding": options.padding,
            "padding_out": options.padding_out,
            "dilation": options.dilation,
            "groups": options.groups
        }));
        builder.finish(InnerAutodiffBackend::conv_transpose2d(x, weight, bias, options))
    }

    fn conv_transpose3d(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv_transpose3d(x, weight, bias, options)
    }

    fn avg_pool2d(
        x: FloatTensor<TrackingBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::avg_pool2d(x, kernel_size, stride, padding, count_include_pad)
    }

    fn avg_pool2d_backward(
        x: FloatTensor<TrackingBackend>,
        grad: FloatTensor<TrackingBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::avg_pool2d_backward(x, grad, kernel_size, stride, padding, count_include_pad)
    }

    fn adaptive_avg_pool2d(
        x: FloatTensor<TrackingBackend>,
        output_size: [usize; 2],
    ) -> FloatTensor<TrackingBackend> {
        let builder = start_tracking_tensor(&x, "adaptive_avg_pool2d", json!({
            "output_size": output_size,
        }));
        builder.finish(InnerAutodiffBackend::adaptive_avg_pool2d(x, output_size))
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<TrackingBackend>,
        grad: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::adaptive_avg_pool2d_backward(x, grad)
    }

    fn max_pool2d(
        x: FloatTensor<TrackingBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::max_pool2d(x, kernel_size, stride, padding, dilation)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<TrackingBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<TrackingBackend> {
        let result =
            InnerAutodiffBackend::max_pool2d_with_indices(x, kernel_size, stride, padding, dilation);
        MaxPool2dWithIndices {
            output: result.output,
            indices: result.indices,
        }
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<TrackingBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: FloatTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
    ) -> MaxPool2dBackward<TrackingBackend> {
        let result = InnerAutodiffBackend::max_pool2d_with_indices_backward(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            output_grad,
            indices,
        );
        MaxPool2dBackward {
            x_grad: result.x_grad,
        }
    }

    fn interpolate(
        x: FloatTensor<TrackingBackend>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::interpolate(x, output_size, options)
    }

    fn interpolate_backward(
        x: FloatTensor<TrackingBackend>,
        grad: FloatTensor<TrackingBackend>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::interpolate_backward(x, grad, output_size, options)
    }

    fn embedding(
        weights: FloatTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::embedding(weights, indices)
    }

    fn embedding_backward(
        weights: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::embedding_backward(weights, output_grad, indices)
    }

    fn conv1d(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv1d(x, weight, bias, options)
    }

    fn conv1d_x_backward(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
        options: ConvOptions<1>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv1d_x_backward(x, weight, output_grad, options)
    }

    fn conv1d_weight_backward(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
        options: ConvOptions<1>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv1d_weight_backward(x, weight, output_grad, options)
    }

    fn conv1d_bias_backward(
        x: FloatTensor<TrackingBackend>,
        bias: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv1d_bias_backward(x, bias, output_grad)
    }

    fn conv2d_x_backward(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
        options: ConvOptions<2>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv2d_x_backward(x, weight, output_grad, options)
    }

    fn conv2d_weight_backward(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
        options: ConvOptions<2>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv2d_weight_backward(x, weight, output_grad, options)
    }

    fn conv2d_bias_backward(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv2d_bias_backward(x, weight, bias, output_grad)
    }

    fn conv3d_x_backward(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
        options: ConvOptions<3>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv3d_x_backward(x, weight, output_grad, options)
    }

    fn conv3d_weight_backward(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
        options: ConvOptions<3>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv3d_weight_backward(x, weight, output_grad, options)
    }

    fn conv3d_bias_backward(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv3d_bias_backward(x, weight, bias, output_grad)
    }

    fn conv_transpose1d(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv_transpose1d(x, weight, bias, options)
    }

    fn conv_transpose1d_x_backward(
        weight: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv_transpose1d_x_backward(weight, output_grad, options)
    }

    fn conv_transpose1d_weight_backward(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv_transpose1d_weight_backward(x, weight, output_grad, options)
    }

    fn conv_transpose1d_bias_backward(
        x: FloatTensor<TrackingBackend>,
        bias: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv_transpose1d_bias_backward(x, bias, output_grad)
    }

    fn conv_transpose2d_x_backward(
        weight: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv_transpose2d_x_backward(weight, output_grad, options)
    }

    fn conv_transpose2d_weight_backward(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv_transpose2d_weight_backward(x, weight, output_grad, options)
    }

    fn conv_transpose2d_bias_backward(
        x: FloatTensor<TrackingBackend>,
        bias: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv_transpose2d_bias_backward(x, bias, output_grad)
    }

    fn conv_transpose3d_x_backward(
        weight: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv_transpose3d_x_backward(weight, output_grad, options)
    }

    fn conv_transpose3d_weight_backward(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv_transpose3d_weight_backward(x, weight, output_grad, options)
    }

    fn conv_transpose3d_bias_backward(
        x: FloatTensor<TrackingBackend>,
        bias: FloatTensor<TrackingBackend>,
        output_grad: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::conv_transpose3d_bias_backward(x, bias, output_grad)
    }

    fn unfold4d(
        x: FloatTensor<TrackingBackend>,
        kernel_size: [usize; 2],
        options: burn::tensor::ops::UnfoldOptions,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::unfold4d(x, kernel_size, options)
    }

    fn avg_pool1d(
        x: FloatTensor<TrackingBackend>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::avg_pool1d(x, kernel_size, stride, padding, count_include_pad)
    }

    fn avg_pool1d_backward(
        x: FloatTensor<TrackingBackend>,
        grad: FloatTensor<TrackingBackend>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::avg_pool1d_backward(x, grad, kernel_size, stride, padding, count_include_pad)
    }

    fn adaptive_avg_pool1d(
        x: FloatTensor<TrackingBackend>,
        output_size: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::adaptive_avg_pool1d(x, output_size)
    }

    fn adaptive_avg_pool1d_backward(
        x: FloatTensor<TrackingBackend>,
        grad: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::adaptive_avg_pool1d_backward(x, grad)
    }

    fn max_pool1d(
        x: FloatTensor<TrackingBackend>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::max_pool1d(x, kernel_size, stride, padding, dilation)
    }

    fn max_pool1d_with_indices(
        x: FloatTensor<TrackingBackend>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> burn::tensor::ops::MaxPool1dWithIndices<TrackingBackend> {
        let result =
            InnerAutodiffBackend::max_pool1d_with_indices(x, kernel_size, stride, padding, dilation);
        burn::tensor::ops::MaxPool1dWithIndices {
            output: result.output,
            indices: result.indices,
        }
    }

    fn max_pool1d_with_indices_backward(
        x: FloatTensor<TrackingBackend>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output_grad: FloatTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
    ) -> burn::tensor::ops::MaxPool1dBackward<TrackingBackend> {
        let result = InnerAutodiffBackend::max_pool1d_with_indices_backward(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            output_grad,
            indices,
        );
        burn::tensor::ops::MaxPool1dBackward {
            x_grad: result.x_grad,
        }
    }

    fn linear(
        input: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: Option<FloatTensor<TrackingBackend>>,
    ) -> FloatTensor<TrackingBackend> {
        let builder = start_tracking_tensor(&input, "linear", json!({
            "weight_shape": weight.primitive.shape.as_slice(),
        }));
        println!("A");
        let tmp = builder.finish(InnerAutodiffBackend::linear(input, weight, bias));
        println!("B");
        tmp
    }
}

impl FloatTensorOps<TrackingBackend> for TrackingBackend {
    fn float_from_data(
        data: TensorData,
        device: &Device<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_from_data(data, device)
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_random(shape, distribution, device)
    }

    fn float_into_data(
        tensor: FloatTensor<TrackingBackend>,
    ) -> impl Future<Output = TensorData> + Send {
        InnerAutodiffBackend::float_into_data(tensor)
    }

    fn float_device(tensor: &FloatTensor<TrackingBackend>) -> Device<TrackingBackend> {
        InnerAutodiffBackend::float_device(tensor)
    }

    fn float_to_device(
        tensor: FloatTensor<TrackingBackend>,
        device: &Device<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_to_device(tensor, device)
    }

    fn float_into_int(tensor: FloatTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::float_into_int(tensor)
    }

    fn float_empty(
        shape: Shape,
        device: &Device<TrackingBackend>,
        dtype: FloatDType,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_empty(shape, device, dtype)
    }

    fn float_add(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_add(lhs, rhs)
    }

    fn float_add_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_add_scalar(lhs, rhs)
    }

    fn float_sub(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_sub(lhs, rhs)
    }

    fn float_sub_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_sub_scalar(lhs, rhs)
    }

    fn float_mul(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_mul(lhs, rhs)
    }

    fn float_mul_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_mul_scalar(lhs, rhs)
    }

    fn float_div(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_div(lhs, rhs)
    }

    fn float_div_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_div_scalar(lhs, rhs)
    }

    fn float_remainder(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_remainder(lhs, rhs)
    }

    fn float_remainder_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_remainder_scalar(lhs, rhs)
    }

    fn float_matmul(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_matmul(lhs, rhs)
    }

    fn float_cross(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_cross(lhs, rhs, dim)
    }

    fn float_recip(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_recip(tensor)
    }

    fn float_swap_dims(
        tensor: FloatTensor<TrackingBackend>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_swap_dims(tensor, dim1, dim2)
    }

    fn float_permute(
        tensor: FloatTensor<TrackingBackend>,
        axes: &[usize],
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_permute(tensor, axes)
    }

    fn float_flip(
        tensor: FloatTensor<TrackingBackend>,
        axes: &[usize],
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_flip(tensor, axes)
    }

    fn float_reshape(
        tensor: FloatTensor<TrackingBackend>,
        shape: Shape,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_reshape(tensor, shape)
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_gather(dim, tensor, indices)
    }

    fn float_scatter(
        dim: usize,
        tensor: FloatTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
        value: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_scatter(dim, tensor, indices, value)
    }

    fn float_select(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_select(tensor, dim, indices)
    }

    fn float_select_assign(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
        value: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_select_assign(tensor, dim, indices, value)
    }

    fn float_slice(
        tensor: FloatTensor<TrackingBackend>,
        slices: &[Slice],
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_slice(tensor, slices)
    }

    fn float_slice_assign(
        tensor: FloatTensor<TrackingBackend>,
        slices: &[Slice],
        value: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_slice_assign(tensor, slices, value)
    }

    fn float_mask_where(
        tensor: FloatTensor<TrackingBackend>,
        mask: BoolTensor<TrackingBackend>,
        value: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_mask_where(tensor, mask, value)
    }

    fn float_mask_fill(
        tensor: FloatTensor<TrackingBackend>,
        mask: BoolTensor<TrackingBackend>,
        value: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_mask_fill(tensor, mask, value)
    }

    fn float_equal(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_equal(lhs, rhs)
    }

    fn float_equal_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_equal_elem(lhs, rhs)
    }

    fn float_greater(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_greater(lhs, rhs)
    }

    fn float_greater_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_greater_elem(lhs, rhs)
    }

    fn float_greater_equal(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_greater_equal(lhs, rhs)
    }

    fn float_greater_equal_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_greater_equal_elem(lhs, rhs)
    }

    fn float_lower(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_lower(lhs, rhs)
    }

    fn float_lower_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_lower_elem(lhs, rhs)
    }

    fn float_lower_equal(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_lower_equal(lhs, rhs)
    }

    fn float_lower_equal_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_lower_equal_elem(lhs, rhs)
    }

    fn float_sum(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_sum(tensor)
    }

    fn float_sum_dim(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_sum_dim(tensor, dim)
    }

    fn float_mean_dim(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_mean_dim(tensor, dim)
    }

    fn float_cumsum(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_cumsum(tensor, dim)
    }

    fn float_cast(
        tensor: FloatTensor<TrackingBackend>,
        dtype: FloatDType,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_cast(tensor, dtype)
    }

    fn float_exp(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_exp(tensor)
    }

    fn float_log(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_log(tensor)
    }

    fn float_log1p(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_log1p(tensor)
    }

    fn float_powf(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_powf(lhs, rhs)
    }

    fn float_powf_scalar(
        tensor: FloatTensor<TrackingBackend>,
        value: f32,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_powf_scalar(tensor, value)
    }

    fn float_sqrt(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_sqrt(tensor)
    }

    fn float_abs(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_abs(tensor)
    }

    fn float_cos(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_cos(tensor)
    }

    fn float_sin(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_sin(tensor)
    }

    fn float_round(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_round(tensor)
    }

    fn float_floor(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_floor(tensor)
    }

    fn float_ceil(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_ceil(tensor)
    }

    fn float_erf(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_erf(tensor)
    }

    fn float_argmax(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::float_argmax(tensor, dim)
    }

    fn float_argmin(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::float_argmin(tensor, dim)
    }

    fn float_expand(
        tensor: FloatTensor<TrackingBackend>,
        shape: Shape,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_expand(tensor, shape)
    }

    fn float_unfold(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_unfold(tensor, dim, size, step)
    }

    fn float_zeros(
        shape: Shape,
        device: &Device<TrackingBackend>,
        dtype: FloatDType,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_zeros(shape, device, dtype)
    }

    fn float_ones(
        shape: Shape,
        device: &Device<TrackingBackend>,
        dtype: FloatDType,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_ones(shape, device, dtype)
    }

    fn float_full(
        shape: Shape,
        fill_value: FloatElem<TrackingBackend>,
        device: &Device<TrackingBackend>,
        dtype: FloatDType,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_full(shape, fill_value, device, dtype)
    }

    fn float_repeat_dim(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
        times: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_repeat_dim(tensor, dim, times)
    }

    fn float_clamp_min(
        tensor: FloatTensor<TrackingBackend>,
        min: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_clamp_min(tensor, min)
    }

    fn float_clamp_max(
        tensor: FloatTensor<TrackingBackend>,
        max: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_clamp_max(tensor, max)
    }

    fn float_clamp(
        tensor: FloatTensor<TrackingBackend>,
        min: FloatElem<TrackingBackend>,
        max: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_clamp(tensor, min, max)
    }

    fn float_neg(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_neg(tensor)
    }

    fn float_transpose(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_transpose(tensor)
    }

    fn float_not_equal(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_not_equal(lhs, rhs)
    }

    fn float_not_equal_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_not_equal_elem(lhs, rhs)
    }

    fn float_detach(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        // Should only be overridden by autodiff backends.
        tensor
    }

    fn float_set_require_grad(
        tensor: FloatTensor<TrackingBackend>,
        _require_grad: bool,
    ) -> FloatTensor<TrackingBackend> {
        // Should only be overridden by autodiff backends.
        tensor
    }

    fn float_is_require_grad(_tensor: &FloatTensor<TrackingBackend>) -> bool {
        // Should only be overridden by autodiff backends.
        false
    }

    fn float_prod(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_prod(tensor)
    }

    fn float_prod_dim(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_prod_dim(tensor, dim)
    }

    fn float_mean(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_mean(tensor)
    }

    fn float_powi(
        lhs: FloatTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_powi(lhs, rhs)
    }

    fn float_powi_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_powi_scalar(lhs, rhs)
    }

    fn float_tan(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_tan(tensor)
    }

    fn float_cosh(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_cosh(tensor)
    }

    fn float_sinh(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_sinh(tensor)
    }

    fn float_tanh(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_tanh(tensor)
    }

    fn float_cat(
        tensors: Vec<FloatTensor<TrackingBackend>>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        let builders: Vec<_> = tensors.iter().map(|tensor| start_tracking_tensor(tensor, "float_cat", Value::Null)).collect();
        let out = InnerAutodiffBackend::float_cat(tensors, dim);
        let to_hash = hash_tensor(&out);
        finish_iter(builders, Some(to_hash));
        out
    }

    fn float_max(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_max(tensor)
    }

    fn float_max_dim(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_max_dim(tensor, dim)
    }

    fn float_max_dim_with_indices(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> (FloatTensor<TrackingBackend>, IntTensor<TrackingBackend>) {
        InnerAutodiffBackend::float_max_dim_with_indices(tensor, dim)
    }

    fn float_min(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_min(tensor)
    }

    fn float_min_dim(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_min_dim(tensor, dim)
    }

    fn float_min_dim_with_indices(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> (FloatTensor<TrackingBackend>, IntTensor<TrackingBackend>) {
        InnerAutodiffBackend::float_min_dim_with_indices(tensor, dim)
    }

    fn float_max_abs(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_max_abs(tensor)
    }

    fn float_max_abs_dim(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_max_abs_dim(tensor, dim)
    }

    fn float_any(tensor: FloatTensor<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_any(tensor)
    }

    fn float_any_dim(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_any_dim(tensor, dim)
    }

    fn float_all(tensor: FloatTensor<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_all(tensor)
    }

    fn float_all_dim(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_all_dim(tensor, dim)
    }

    fn float_sign(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_sign(tensor)
    }

    fn float_sort(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
        descending: bool,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_sort(tensor, dim, descending)
    }

    fn float_sort_with_indices(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
        descending: bool,
    ) -> (FloatTensor<TrackingBackend>, IntTensor<TrackingBackend>) {
        InnerAutodiffBackend::float_sort_with_indices(tensor, dim, descending)
    }

    fn float_argsort(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
        descending: bool,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::float_argsort(tensor, dim, descending)
    }

    fn float_grid_sample_2d(
        tensor: FloatTensor<TrackingBackend>,
        grid: FloatTensor<TrackingBackend>,
        method: burn::tensor::ops::InterpolateMode,
    ) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::float_grid_sample_2d(tensor, grid, method)
    }

    fn float_is_nan(tensor: FloatTensor<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_is_nan(tensor)
    }

    fn float_is_inf(tensor: FloatTensor<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::float_is_inf(tensor)
    }
}

impl QTensorOps<TrackingBackend> for TrackingBackend {
    fn q_from_data(
        data: TensorData,
        device: &Device<TrackingBackend>,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_from_data(data, device)
    }

    fn quantize(
        tensor: FloatTensor<TrackingBackend>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<TrackingBackend>,
    ) -> QuantizedTensor<TrackingBackend> {
        // Safe transmute for qparams since TrackingBackend is repr(transparent) over InnerBackend
        InnerAutodiffBackend::quantize(tensor, scheme, unsafe { transmute(qparams) })
    }

    fn dequantize(tensor: QuantizedTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerAutodiffBackend::dequantize(tensor)
    }

    fn q_device(tensor: &QuantizedTensor<TrackingBackend>) -> Device<TrackingBackend> {
        InnerAutodiffBackend::q_device(tensor)
    }

    fn q_to_device(
        tensor: QuantizedTensor<TrackingBackend>,
        device: &Device<TrackingBackend>,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_to_device(tensor, device)
    }

    fn q_reshape(
        tensor: QuantizedTensor<TrackingBackend>,
        shape: Shape,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_reshape(tensor, shape)
    }

    fn q_into_data(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> impl Future<Output = TensorData> + Send {
        InnerAutodiffBackend::q_into_data(tensor)
    }

    fn q_expand(
        tensor: QuantizedTensor<TrackingBackend>,
        shape: Shape,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_expand(tensor, shape)
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<TrackingBackend>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(
        tensor: QuantizedTensor<TrackingBackend>,
        axes: &[usize],
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_permute(tensor, axes)
    }

    fn q_flip(
        tensor: QuantizedTensor<TrackingBackend>,
        axes: &[usize],
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_flip(tensor, axes)
    }

    fn q_select(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_select(tensor, dim, indices)
    }

    fn q_slice(
        tensor: QuantizedTensor<TrackingBackend>,
        slices: &[Slice],
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_slice(tensor, slices)
    }

    fn quantize_dynamic(
        tensor: FloatTensor<TrackingBackend>,
        scheme: &QuantScheme,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::quantize_dynamic(tensor, scheme)
    }

    fn q_detach(tensor: QuantizedTensor<TrackingBackend>) -> QuantizedTensor<TrackingBackend> {
        // Should only be overridden by autodiff backends.
        tensor
    }

    fn q_set_require_grad(
        tensor: QuantizedTensor<TrackingBackend>,
        _require_grad: bool,
    ) -> QuantizedTensor<TrackingBackend> {
        // Should only be overridden by autodiff backends.
        tensor
    }

    fn q_is_require_grad(_tensor: &QuantizedTensor<TrackingBackend>) -> bool {
        // Should only be overridden by autodiff backends.
        false
    }

    fn q_transpose(tensor: QuantizedTensor<TrackingBackend>) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_transpose(tensor)
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_gather(dim, tensor, indices)
    }

    fn q_repeat_dim(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
        times: usize,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_repeat_dim(tensor, dim, times)
    }

    fn q_add(
        lhs: QuantizedTensor<TrackingBackend>,
        rhs: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_add(lhs, rhs)) }
    }

    fn q_add_scalar(
        lhs: QuantizedTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_add_scalar(lhs, rhs)) }
    }

    fn q_clamp_min(
        tensor: QuantizedTensor<TrackingBackend>,
        min: FloatElem<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_clamp_min(tensor, min)) }
    }

    fn q_clamp_max(
        tensor: QuantizedTensor<TrackingBackend>,
        max: FloatElem<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_clamp_max(tensor, max)) }
    }

    fn q_clamp(
        tensor: QuantizedTensor<TrackingBackend>,
        min: FloatElem<TrackingBackend>,
        max: FloatElem<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_clamp(tensor, min, max)) }
    }

    fn q_sub(
        lhs: QuantizedTensor<TrackingBackend>,
        rhs: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_sub(lhs, rhs)) }
    }

    fn q_sub_scalar(
        lhs: QuantizedTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_sub_scalar(lhs, rhs)) }
    }

    fn q_mul(
        lhs: QuantizedTensor<TrackingBackend>,
        rhs: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_mul(lhs, rhs)) }
    }

    fn q_mul_scalar(
        lhs: QuantizedTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_mul_scalar(lhs, rhs)) }
    }

    fn q_div(
        lhs: QuantizedTensor<TrackingBackend>,
        rhs: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_div(lhs, rhs)) }
    }

    fn q_div_scalar(
        lhs: QuantizedTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_div_scalar(lhs, rhs)) }
    }

    fn q_matmul(
        lhs: QuantizedTensor<TrackingBackend>,
        rhs: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_matmul(lhs, rhs)) }
    }

    fn q_neg(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_neg(tensor)) }
    }

    fn q_recip(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_recip(tensor)) }
    }

    fn q_sum(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_sum(tensor)) }
    }

    fn q_sum_dim(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_sum_dim(tensor, dim)) }
    }

    fn q_prod(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_prod(tensor)) }
    }

    fn q_prod_dim(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_prod_dim(tensor, dim)) }
    }

    fn q_mean(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_mean(tensor)) }
    }

    fn q_mean_dim(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_mean_dim(tensor, dim)) }
    }

    fn q_cumsum(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_cumsum(tensor, dim)) }
    }

    fn q_exp(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_exp(tensor)) }
    }

    fn q_log(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_log(tensor)) }
    }

    fn q_log1p(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_log1p(tensor)) }
    }

    fn q_powf(
        lhs: QuantizedTensor<TrackingBackend>,
        rhs: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_powf(lhs, rhs)) }
    }

    fn q_powi(
        lhs: QuantizedTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_powi(lhs, rhs)) }
    }

    fn q_powi_scalar(
        lhs: QuantizedTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_powi_scalar(lhs, rhs)) }
    }

    fn q_powf_scalar(
        tensor: QuantizedTensor<TrackingBackend>,
        value: f32,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_powf_scalar(tensor, value)) }
    }

    fn q_sqrt(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_sqrt(tensor)) }
    }

    fn q_abs(tensor: QuantizedTensor<TrackingBackend>) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_abs(tensor)
    }

    fn q_cos(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_cos(tensor)) }
    }

    fn q_sin(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_sin(tensor)) }
    }

    fn q_tan(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_tan(tensor)) }
    }

    fn q_cosh(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_cosh(tensor)) }
    }

    fn q_sinh(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_sinh(tensor)) }
    }

    fn q_tanh(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_tanh(tensor)) }
    }

    fn q_erf(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> burn::tensor::TensorPrimitive<TrackingBackend> {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        unsafe { transmute(InnerAutodiffBackend::q_erf(tensor)) }
    }

    fn q_cat(
        tensors: Vec<QuantizedTensor<TrackingBackend>>,
        dim: usize,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_cat(tensors, dim)
    }

    fn q_argmax(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::q_argmax(tensor, dim)
    }

    fn q_argmin(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::q_argmin(tensor, dim)
    }

    fn q_max(tensor: QuantizedTensor<TrackingBackend>) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_max(tensor)
    }

    fn q_max_dim(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_max_dim(tensor, dim)
    }

    fn q_max_dim_with_indices(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
    ) -> (QuantizedTensor<TrackingBackend>, IntTensor<TrackingBackend>) {
        InnerAutodiffBackend::q_max_dim_with_indices(tensor, dim)
    }

    fn q_min(tensor: QuantizedTensor<TrackingBackend>) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_min(tensor)
    }

    fn q_min_dim(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_min_dim(tensor, dim)
    }

    fn q_min_dim_with_indices(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
    ) -> (QuantizedTensor<TrackingBackend>, IntTensor<TrackingBackend>) {
        InnerAutodiffBackend::q_min_dim_with_indices(tensor, dim)
    }

    fn q_max_abs(tensor: QuantizedTensor<TrackingBackend>) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_max_abs(tensor)
    }

    fn q_max_abs_dim(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_max_abs_dim(tensor, dim)
    }

    fn q_any(tensor: QuantizedTensor<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::q_any(tensor)
    }

    fn q_any_dim(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::q_any_dim(tensor, dim)
    }

    fn q_all(tensor: QuantizedTensor<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::q_all(tensor)
    }

    fn q_all_dim(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
    ) -> BoolTensor<TrackingBackend> {
        InnerAutodiffBackend::q_all_dim(tensor, dim)
    }

    fn q_sort(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
        descending: bool,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerAutodiffBackend::q_sort(tensor, dim, descending)
    }

    fn q_sort_with_indices(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
        descending: bool,
    ) -> (QuantizedTensor<TrackingBackend>, IntTensor<TrackingBackend>) {
        InnerAutodiffBackend::q_sort_with_indices(tensor, dim, descending)
    }

    fn q_argsort(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
        descending: bool,
    ) -> IntTensor<TrackingBackend> {
        InnerAutodiffBackend::q_argsort(tensor, dim, descending)
    }
}

impl TransactionOps<TrackingBackend> for TrackingBackend {
    fn tr_execute(
        transaction: TransactionPrimitive<TrackingBackend>,
    ) -> impl Future<Output = TransactionPrimitiveResult> + Send {
        // Safe transmute since TrackingBackend is repr(transparent) over InnerBackend
        InnerAutodiffBackend::tr_execute(unsafe { transmute(transaction) })
    }
}
