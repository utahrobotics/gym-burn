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

use super::InnerBackend;

use super::TrackingBackend;

use burn::prelude::Backend;

impl Backend for TrackingBackend {
    type Device = <InnerBackend as Backend>::Device;

    type FloatTensorPrimitive = <InnerBackend as Backend>::FloatTensorPrimitive;

    type FloatElem = <InnerBackend as Backend>::FloatElem;

    type IntTensorPrimitive = <InnerBackend as Backend>::IntTensorPrimitive;

    type IntElem = <InnerBackend as Backend>::IntElem;

    type BoolTensorPrimitive = <InnerBackend as Backend>::BoolTensorPrimitive;

    type BoolElem = <InnerBackend as Backend>::BoolElem;

    type QuantizedTensorPrimitive = <InnerBackend as Backend>::QuantizedTensorPrimitive;

    fn name(device: &Self::Device) -> String {
        InnerBackend::name(device)
    }

    fn seed(device: &Self::Device, seed: u64) {
        InnerBackend::seed(device, seed)
    }

    fn ad_enabled() -> bool {
        InnerBackend::ad_enabled()
    }

    fn memory_persistent_allocations<Output, Input, Func: Fn(Input) -> Output>(
        device: &Self::Device,
        input: Input,
        func: Func,
    ) -> Output {
        InnerBackend::memory_persistent_allocations(device, input, func)
    }

    fn memory_cleanup(device: &Self::Device) {
        InnerBackend::memory_cleanup(device);
    }

    fn sync(device: &Self::Device) {
        InnerBackend::sync(device);
    }
}

impl BoolTensorOps<TrackingBackend> for TrackingBackend {
    fn bool_empty(shape: Shape, device: &Device<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_empty(shape, device)
    }

    fn bool_zeros(shape: Shape, device: &Device<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_zeros(shape, device)
    }

    fn bool_ones(shape: Shape, device: &Device<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_ones(shape, device)
    }

    fn bool_into_data(
        tensor: BoolTensor<TrackingBackend>,
    ) -> impl Future<Output = TensorData> + Send {
        InnerBackend::bool_into_data(tensor)
    }

    fn bool_from_data(
        data: TensorData,
        device: &Device<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_from_data(data, device)
    }

    fn bool_into_int(tensor: BoolTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerBackend::bool_into_int(tensor)
    }

    fn bool_into_float(tensor: BoolTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::bool_into_float(tensor)
    }

    fn bool_device(tensor: &BoolTensor<TrackingBackend>) -> Device<TrackingBackend> {
        InnerBackend::bool_device(tensor)
    }

    fn bool_to_device(
        tensor: BoolTensor<TrackingBackend>,
        device: &Device<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_to_device(tensor, device)
    }

    fn bool_reshape(
        tensor: BoolTensor<TrackingBackend>,
        shape: Shape,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_reshape(tensor, shape)
    }

    fn bool_slice(
        tensor: BoolTensor<TrackingBackend>,
        slices: &[Slice],
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_slice(tensor, slices)
    }

    fn bool_slice_assign(
        tensor: BoolTensor<TrackingBackend>,
        slices: &[Slice],
        value: BoolTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_slice_assign(tensor, slices, value)
    }

    fn bool_equal(
        lhs: BoolTensor<TrackingBackend>,
        rhs: BoolTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_equal(lhs, rhs)
    }

    fn bool_not(tensor: BoolTensor<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_not(tensor)
    }

    fn bool_and(
        tensor: BoolTensor<TrackingBackend>,
        rhs: BoolTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_and(tensor, rhs)
    }

    fn bool_or(
        tensor: BoolTensor<TrackingBackend>,
        rhs: BoolTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_or(tensor, rhs)
    }

    fn bool_swap_dims(
        tensor: BoolTensor<TrackingBackend>,
        dim1: usize,
        dim2: usize,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_swap_dims(tensor, dim1, dim2)
    }

    fn bool_permute(
        tensor: BoolTensor<TrackingBackend>,
        axes: &[usize],
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_permute(tensor, axes)
    }

    fn bool_flip(
        tensor: BoolTensor<TrackingBackend>,
        axes: &[usize],
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_flip(tensor, axes)
    }

    fn bool_expand(
        tensor: BoolTensor<TrackingBackend>,
        shape: Shape,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_expand(tensor, shape)
    }

    fn bool_unfold(
        tensor: BoolTensor<TrackingBackend>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::bool_unfold(tensor, dim, size, step)
    }
}

impl IntTensorOps<TrackingBackend> for TrackingBackend {
    fn int_empty(
        shape: Shape,
        device: &Device<TrackingBackend>,
        dtype: IntDType,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_empty(shape, device, dtype)
    }

    fn int_into_data(
        tensor: IntTensor<TrackingBackend>,
    ) -> impl Future<Output = TensorData> + Send {
        InnerBackend::int_into_data(tensor)
    }

    fn int_from_data(
        data: TensorData,
        device: &Device<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_from_data(data, device)
    }

    fn int_device(tensor: &IntTensor<TrackingBackend>) -> Device<TrackingBackend> {
        InnerBackend::int_device(tensor)
    }

    fn int_to_device(
        tensor: IntTensor<TrackingBackend>,
        device: &Device<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_to_device(tensor, device)
    }

    fn int_reshape(tensor: IntTensor<TrackingBackend>, shape: Shape) -> IntTensor<TrackingBackend> {
        InnerBackend::int_reshape(tensor, shape)
    }

    fn int_slice(
        tensor: IntTensor<TrackingBackend>,
        slices: &[Slice],
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_slice(tensor, slices)
    }

    fn int_slice_assign(
        tensor: IntTensor<TrackingBackend>,
        slices: &[Slice],
        value: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_slice_assign(tensor, slices, value)
    }

    fn int_into_float(tensor: IntTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::int_into_float(tensor)
    }

    fn int_mask_where(
        tensor: IntTensor<TrackingBackend>,
        mask: BoolTensor<TrackingBackend>,
        source: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_mask_where(tensor, mask, source)
    }

    fn int_mask_fill(
        tensor: IntTensor<TrackingBackend>,
        mask: BoolTensor<TrackingBackend>,
        value: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_mask_fill(tensor, mask, value)
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_gather(dim, tensor, indices)
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
        value: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_scatter(dim, tensor, indices, value)
    }

    fn int_select(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_select(tensor, dim, indices)
    }

    fn int_select_assign(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
        value: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_select_assign(tensor, dim, indices, value)
    }

    fn int_equal(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::int_equal(lhs, rhs)
    }

    fn int_equal_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::int_equal_elem(lhs, rhs)
    }

    fn int_greater(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::int_greater(lhs, rhs)
    }

    fn int_greater_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::int_greater_elem(lhs, rhs)
    }

    fn int_greater_equal(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::int_greater_equal(lhs, rhs)
    }

    fn int_greater_equal_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::int_greater_equal_elem(lhs, rhs)
    }

    fn int_lower(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::int_lower(lhs, rhs)
    }

    fn int_lower_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::int_lower_elem(lhs, rhs)
    }

    fn int_lower_equal(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::int_lower_equal(lhs, rhs)
    }

    fn int_lower_equal_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::int_lower_equal_elem(lhs, rhs)
    }

    fn int_add(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_add(lhs, rhs)
    }

    fn int_add_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_add_scalar(lhs, rhs)
    }

    fn int_sub(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_sub(lhs, rhs)
    }

    fn int_sub_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_sub_scalar(lhs, rhs)
    }

    fn int_mul(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_mul(lhs, rhs)
    }

    fn int_mul_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_mul_scalar(lhs, rhs)
    }

    fn int_div(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_div(lhs, rhs)
    }

    fn int_div_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_div_scalar(lhs, rhs)
    }

    fn int_remainder(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_remainder(lhs, rhs)
    }

    fn int_remainder_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_remainder_scalar(lhs, rhs)
    }

    fn int_matmul(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_matmul(lhs, rhs)
    }

    fn int_sum(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerBackend::int_sum(tensor)
    }

    fn int_sum_dim(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        InnerBackend::int_sum_dim(tensor, dim)
    }

    fn int_prod(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerBackend::int_prod(tensor)
    }

    fn int_prod_dim(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        InnerBackend::int_prod_dim(tensor, dim)
    }

    fn int_mean_dim(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        InnerBackend::int_mean_dim(tensor, dim)
    }

    fn int_cumsum(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        InnerBackend::int_cumsum(tensor, dim)
    }

    fn int_argmax(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        InnerBackend::int_argmax(tensor, dim)
    }

    fn int_argmin(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        InnerBackend::int_argmin(tensor, dim)
    }

    fn int_abs(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerBackend::int_abs(tensor)
    }

    fn int_swap_dims(
        tensor: IntTensor<TrackingBackend>,
        dim1: usize,
        dim2: usize,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_swap_dims(tensor, dim1, dim2)
    }

    fn int_permute(
        tensor: IntTensor<TrackingBackend>,
        axes: &[usize],
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_permute(tensor, axes)
    }

    fn int_flip(tensor: IntTensor<TrackingBackend>, axes: &[usize]) -> IntTensor<TrackingBackend> {
        InnerBackend::int_flip(tensor, axes)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_random(shape, distribution, device)
    }

    fn int_expand(tensor: IntTensor<TrackingBackend>, shape: Shape) -> IntTensor<TrackingBackend> {
        InnerBackend::int_expand(tensor, shape)
    }

    fn bitwise_and(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::bitwise_and(lhs, rhs)
    }

    fn bitwise_and_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::bitwise_and_scalar(lhs, rhs)
    }

    fn bitwise_or(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::bitwise_or(lhs, rhs)
    }

    fn bitwise_or_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::bitwise_or_scalar(lhs, rhs)
    }

    fn bitwise_xor(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::bitwise_xor(lhs, rhs)
    }

    fn bitwise_xor_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::bitwise_xor_scalar(lhs, rhs)
    }

    fn bitwise_not(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerBackend::bitwise_not(tensor)
    }

    fn bitwise_left_shift(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::bitwise_left_shift(lhs, rhs)
    }

    fn bitwise_left_shift_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::bitwise_left_shift_scalar(lhs, rhs)
    }

    fn bitwise_right_shift(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::bitwise_right_shift(lhs, rhs)
    }

    fn bitwise_right_shift_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::bitwise_right_shift_scalar(lhs, rhs)
    }

    fn int_cast(tensor: IntTensor<TrackingBackend>, dtype: IntDType) -> IntTensor<TrackingBackend> {
        InnerBackend::int_cast(tensor, dtype)
    }

    fn int_unfold(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::int_unfold(tensor, dim, size, step)
    }
}

impl ActivationOps<TrackingBackend> for TrackingBackend {}

impl ModuleOps<TrackingBackend> for TrackingBackend {
    fn conv2d(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::conv2d(x, weight, bias, options)
    }

    fn deform_conv2d(
        x: FloatTensor<TrackingBackend>,
        offset: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        mask: Option<FloatTensor<TrackingBackend>>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::deform_conv2d(x, offset, weight, mask, bias, options)
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
        unsafe { transmute(InnerBackend::deform_conv2d_backward(x, offset, weight, mask, bias, output_grad, options)) }
    }

    fn conv3d(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::conv3d(x, weight, bias, options)
    }

    fn conv_transpose2d(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::conv_transpose2d(x, weight, bias, options)
    }

    fn conv_transpose3d(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::conv_transpose3d(x, weight, bias, options)
    }

    fn avg_pool2d(
        x: FloatTensor<TrackingBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::avg_pool2d(x, kernel_size, stride, padding, count_include_pad)
    }

    fn avg_pool2d_backward(
        x: FloatTensor<TrackingBackend>,
        grad: FloatTensor<TrackingBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::avg_pool2d_backward(x, grad, kernel_size, stride, padding, count_include_pad)
    }

    fn adaptive_avg_pool2d(
        x: FloatTensor<TrackingBackend>,
        output_size: [usize; 2],
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::adaptive_avg_pool2d(x, output_size)
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<TrackingBackend>,
        grad: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::adaptive_avg_pool2d_backward(x, grad)
    }

    fn max_pool2d(
        x: FloatTensor<TrackingBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::max_pool2d(x, kernel_size, stride, padding, dilation)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<TrackingBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<TrackingBackend> {
        unsafe { transmute(InnerBackend::max_pool2d_with_indices(x, kernel_size, stride, padding, dilation)) }
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
        unsafe { transmute(InnerBackend::max_pool2d_with_indices_backward(x, kernel_size, stride, padding, dilation, output_grad, indices)) }
    }

    fn interpolate(
        x: FloatTensor<TrackingBackend>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::interpolate(x, output_size, options)
    }

    fn interpolate_backward(
        x: FloatTensor<TrackingBackend>,
        grad: FloatTensor<TrackingBackend>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::interpolate_backward(x, grad, output_size, options)
    }
}

impl FloatTensorOps<TrackingBackend> for TrackingBackend {
    fn float_from_data(
        data: TensorData,
        device: &Device<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_from_data(data, device)
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_random(shape, distribution, device)
    }

    fn float_into_data(
        tensor: FloatTensor<TrackingBackend>,
    ) -> impl Future<Output = TensorData> + Send {
        InnerBackend::float_into_data(tensor)
    }

    fn float_device(tensor: &FloatTensor<TrackingBackend>) -> Device<TrackingBackend> {
        InnerBackend::float_device(tensor)
    }

    fn float_to_device(
        tensor: FloatTensor<TrackingBackend>,
        device: &Device<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_to_device(tensor, device)
    }

    fn float_into_int(tensor: FloatTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        InnerBackend::float_into_int(tensor)
    }

    fn float_empty(
        shape: Shape,
        device: &Device<TrackingBackend>,
        dtype: FloatDType,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_empty(shape, device, dtype)
    }

    fn float_add(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_add(lhs, rhs)
    }

    fn float_add_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_add_scalar(lhs, rhs)
    }

    fn float_sub(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_sub(lhs, rhs)
    }

    fn float_sub_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_sub_scalar(lhs, rhs)
    }

    fn float_mul(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_mul(lhs, rhs)
    }

    fn float_mul_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_mul_scalar(lhs, rhs)
    }

    fn float_div(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_div(lhs, rhs)
    }

    fn float_div_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_div_scalar(lhs, rhs)
    }

    fn float_remainder(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_remainder(lhs, rhs)
    }

    fn float_remainder_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_remainder_scalar(lhs, rhs)
    }

    fn float_matmul(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_matmul(lhs, rhs)
    }

    fn float_cross(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_cross(lhs, rhs, dim)
    }

    fn float_recip(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_recip(tensor)
    }

    fn float_swap_dims(
        tensor: FloatTensor<TrackingBackend>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_swap_dims(tensor, dim1, dim2)
    }

    fn float_permute(
        tensor: FloatTensor<TrackingBackend>,
        axes: &[usize],
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_permute(tensor, axes)
    }

    fn float_flip(
        tensor: FloatTensor<TrackingBackend>,
        axes: &[usize],
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_flip(tensor, axes)
    }

    fn float_reshape(
        tensor: FloatTensor<TrackingBackend>,
        shape: Shape,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_reshape(tensor, shape)
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_gather(dim, tensor, indices)
    }

    fn float_scatter(
        dim: usize,
        tensor: FloatTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
        value: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_scatter(dim, tensor, indices, value)
    }

    fn float_select(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_select(tensor, dim, indices)
    }

    fn float_select_assign(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
        value: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_select_assign(tensor, dim, indices, value)
    }

    fn float_slice(
        tensor: FloatTensor<TrackingBackend>,
        slices: &[Slice],
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_slice(tensor, slices)
    }

    fn float_slice_assign(
        tensor: FloatTensor<TrackingBackend>,
        slices: &[Slice],
        value: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_slice_assign(tensor, slices, value)
    }

    fn float_mask_where(
        tensor: FloatTensor<TrackingBackend>,
        mask: BoolTensor<TrackingBackend>,
        value: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_mask_where(tensor, mask, value)
    }

    fn float_mask_fill(
        tensor: FloatTensor<TrackingBackend>,
        mask: BoolTensor<TrackingBackend>,
        value: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_mask_fill(tensor, mask, value)
    }

    fn float_equal(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::float_equal(lhs, rhs)
    }

    fn float_equal_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::float_equal_elem(lhs, rhs)
    }

    fn float_greater(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::float_greater(lhs, rhs)
    }

    fn float_greater_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::float_greater_elem(lhs, rhs)
    }

    fn float_greater_equal(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::float_greater_equal(lhs, rhs)
    }

    fn float_greater_equal_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::float_greater_equal_elem(lhs, rhs)
    }

    fn float_lower(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::float_lower(lhs, rhs)
    }

    fn float_lower_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::float_lower_elem(lhs, rhs)
    }

    fn float_lower_equal(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::float_lower_equal(lhs, rhs)
    }

    fn float_lower_equal_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        InnerBackend::float_lower_equal_elem(lhs, rhs)
    }

    fn float_sum(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_sum(tensor)
    }

    fn float_sum_dim(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_sum_dim(tensor, dim)
    }

    fn float_mean_dim(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_mean_dim(tensor, dim)
    }

    fn float_cumsum(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_cumsum(tensor, dim)
    }

    fn float_cast(
        tensor: FloatTensor<TrackingBackend>,
        dtype: FloatDType,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_cast(tensor, dtype)
    }

    fn float_exp(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_exp(tensor)
    }

    fn float_log(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_log(tensor)
    }

    fn float_log1p(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_log1p(tensor)
    }

    fn float_powf(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_powf(lhs, rhs)
    }

    fn float_powf_scalar(
        tensor: FloatTensor<TrackingBackend>,
        value: f32,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_powf_scalar(tensor, value)
    }

    fn float_sqrt(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_sqrt(tensor)
    }

    fn float_abs(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_abs(tensor)
    }

    fn float_cos(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_cos(tensor)
    }

    fn float_sin(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_sin(tensor)
    }

    fn float_round(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_round(tensor)
    }

    fn float_floor(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_floor(tensor)
    }

    fn float_ceil(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_ceil(tensor)
    }

    fn float_erf(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_erf(tensor)
    }

    fn float_argmax(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::float_argmax(tensor, dim)
    }

    fn float_argmin(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> IntTensor<TrackingBackend> {
        InnerBackend::float_argmin(tensor, dim)
    }

    fn float_expand(
        tensor: FloatTensor<TrackingBackend>,
        shape: Shape,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_expand(tensor, shape)
    }

    fn float_unfold(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<TrackingBackend> {
        InnerBackend::float_unfold(tensor, dim, size, step)
    }
}

impl QTensorOps<TrackingBackend> for TrackingBackend {
    fn q_from_data(
        data: TensorData,
        device: &Device<TrackingBackend>,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerBackend::q_from_data(data, device)
    }

    fn quantize(
        tensor: FloatTensor<TrackingBackend>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<TrackingBackend>,
    ) -> QuantizedTensor<TrackingBackend> {
        unsafe { transmute(InnerBackend::quantize(tensor, scheme, transmute(qparams))) }
    }

    fn dequantize(tensor: QuantizedTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        InnerBackend::dequantize(tensor)
    }

    fn q_device(tensor: &QuantizedTensor<TrackingBackend>) -> Device<TrackingBackend> {
        InnerBackend::q_device(tensor)
    }

    fn q_to_device(
        tensor: QuantizedTensor<TrackingBackend>,
        device: &Device<TrackingBackend>,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerBackend::q_to_device(tensor, device)
    }

    fn q_reshape(
        tensor: QuantizedTensor<TrackingBackend>,
        shape: Shape,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerBackend::q_reshape(tensor, shape)
    }

    fn q_into_data(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> impl Future<Output = TensorData> + Send {
        InnerBackend::q_into_data(tensor)
    }

    fn q_expand(
        tensor: QuantizedTensor<TrackingBackend>,
        shape: Shape,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerBackend::q_expand(tensor, shape)
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<TrackingBackend>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerBackend::q_swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(
        tensor: QuantizedTensor<TrackingBackend>,
        axes: &[usize],
    ) -> QuantizedTensor<TrackingBackend> {
        InnerBackend::q_permute(tensor, axes)
    }

    fn q_flip(
        tensor: QuantizedTensor<TrackingBackend>,
        axes: &[usize],
    ) -> QuantizedTensor<TrackingBackend> {
        InnerBackend::q_flip(tensor, axes)
    }

    fn q_select(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
    ) -> QuantizedTensor<TrackingBackend> {
        InnerBackend::q_select(tensor, dim, indices)
    }

    fn q_slice(
        tensor: QuantizedTensor<TrackingBackend>,
        slices: &[Slice],
    ) -> QuantizedTensor<TrackingBackend> {
        InnerBackend::q_slice(tensor, slices)
    }
}

impl TransactionOps<TrackingBackend> for TrackingBackend {
    fn tr_execute(
        transaction: TransactionPrimitive<TrackingBackend>,
    ) -> impl Future<Output = TransactionPrimitiveResult> + Send {
        InnerBackend::tr_execute(unsafe { transmute(transaction) })
    }
}
