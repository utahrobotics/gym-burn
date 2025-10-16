use std::{marker::PhantomData, mem::transmute};

use burn::{
    backend::{Autodiff, Wgpu},
    prelude::{Backend, Device, Shape, TensorData},
    tensor::{
        Distribution, FloatDType, IntDType, Slice,
        ops::{
            ActivationOps, BoolTensor, BoolTensorOps, ConvOptions, ConvTransposeOptions,
            DeformConv2dBackward, DeformConvOptions, FloatElem, FloatTensor, FloatTensorOps,
            IntElem, IntTensor, IntTensorOps, InterpolateOptions, MaxPool2dBackward,
            MaxPool2dWithIndices, ModuleOps, QTensorOps, QuantizedTensor, TransactionOps,
            TransactionPrimitive, TransactionPrimitiveResult,
        },
        quantization::{QuantScheme, QuantizationParametersPrimitive},
    },
};

pub type InnerBackend = Autodiff<Wgpu<f32, i32, u32>>;

#[derive(Default, Debug, Clone)]
#[repr(transparent)]
pub struct TrackingBackend(InnerBackend);

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
        todo!()
    }

    fn bool_zeros(shape: Shape, device: &Device<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn bool_ones(shape: Shape, device: &Device<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn bool_into_data(
        tensor: BoolTensor<TrackingBackend>,
    ) -> impl Future<Output = TensorData> + Send {
        todo!()
    }

    fn bool_from_data(
        data: TensorData,
        device: &Device<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn bool_into_int(tensor: BoolTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn bool_into_float(tensor: BoolTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn bool_device(tensor: &BoolTensor<TrackingBackend>) -> Device<TrackingBackend> {
        todo!()
    }

    fn bool_to_device(
        tensor: BoolTensor<TrackingBackend>,
        device: &Device<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn bool_reshape(
        tensor: BoolTensor<TrackingBackend>,
        shape: Shape,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn bool_slice(
        tensor: BoolTensor<TrackingBackend>,
        slices: &[Slice],
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn bool_slice_assign(
        tensor: BoolTensor<TrackingBackend>,
        slices: &[Slice],
        value: BoolTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn bool_equal(
        lhs: BoolTensor<TrackingBackend>,
        rhs: BoolTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn bool_not(tensor: BoolTensor<TrackingBackend>) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn bool_and(
        tensor: BoolTensor<TrackingBackend>,
        rhs: BoolTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn bool_or(
        tensor: BoolTensor<TrackingBackend>,
        rhs: BoolTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn bool_swap_dims(
        tensor: BoolTensor<TrackingBackend>,
        dim1: usize,
        dim2: usize,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn bool_permute(
        tensor: BoolTensor<TrackingBackend>,
        axes: &[usize],
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn bool_flip(
        tensor: BoolTensor<TrackingBackend>,
        axes: &[usize],
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn bool_expand(
        tensor: BoolTensor<TrackingBackend>,
        shape: Shape,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn bool_unfold(
        tensor: BoolTensor<TrackingBackend>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }
}

impl IntTensorOps<TrackingBackend> for TrackingBackend {
    fn int_empty(
        shape: Shape,
        device: &Device<TrackingBackend>,
        dtype: IntDType,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_into_data(
        tensor: IntTensor<TrackingBackend>,
    ) -> impl Future<Output = TensorData> + Send {
        todo!()
    }

    fn int_from_data(
        data: TensorData,
        device: &Device<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_device(tensor: &IntTensor<TrackingBackend>) -> Device<TrackingBackend> {
        todo!()
    }

    fn int_to_device(
        tensor: IntTensor<TrackingBackend>,
        device: &Device<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_reshape(tensor: IntTensor<TrackingBackend>, shape: Shape) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_slice(
        tensor: IntTensor<TrackingBackend>,
        slices: &[Slice],
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_slice_assign(
        tensor: IntTensor<TrackingBackend>,
        slices: &[Slice],
        value: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_into_float(tensor: IntTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn int_mask_where(
        tensor: IntTensor<TrackingBackend>,
        mask: BoolTensor<TrackingBackend>,
        source: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_mask_fill(
        tensor: IntTensor<TrackingBackend>,
        mask: BoolTensor<TrackingBackend>,
        value: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
        value: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_select(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_select_assign(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
        value: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_equal(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn int_equal_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn int_greater(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn int_greater_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn int_greater_equal(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn int_greater_equal_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn int_lower(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn int_lower_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn int_lower_equal(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn int_lower_equal_elem(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn int_add(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_add_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_sub(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_sub_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_mul(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_mul_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_div(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_div_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_remainder(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_remainder_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_matmul(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_sum(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_sum_dim(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_prod(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_prod_dim(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_mean_dim(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_cumsum(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_argmax(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_argmin(tensor: IntTensor<TrackingBackend>, dim: usize) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_abs(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_swap_dims(
        tensor: IntTensor<TrackingBackend>,
        dim1: usize,
        dim2: usize,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_permute(
        tensor: IntTensor<TrackingBackend>,
        axes: &[usize],
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_flip(tensor: IntTensor<TrackingBackend>, axes: &[usize]) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_expand(tensor: IntTensor<TrackingBackend>, shape: Shape) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn bitwise_and(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn bitwise_and_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn bitwise_or(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn bitwise_or_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn bitwise_xor(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn bitwise_xor_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn bitwise_not(tensor: IntTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn bitwise_left_shift(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn bitwise_left_shift_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn bitwise_right_shift(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntTensor<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn bitwise_right_shift_scalar(
        lhs: IntTensor<TrackingBackend>,
        rhs: IntElem<TrackingBackend>,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_cast(tensor: IntTensor<TrackingBackend>, dtype: IntDType) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn int_unfold(
        tensor: IntTensor<TrackingBackend>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<TrackingBackend> {
        todo!()
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
        todo!()
    }

    fn deform_conv2d(
        x: FloatTensor<TrackingBackend>,
        offset: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        mask: Option<FloatTensor<TrackingBackend>>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
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
        todo!()
    }

    fn conv3d(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn conv_transpose2d(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn conv_transpose3d(
        x: FloatTensor<TrackingBackend>,
        weight: FloatTensor<TrackingBackend>,
        bias: Option<FloatTensor<TrackingBackend>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn avg_pool2d(
        x: FloatTensor<TrackingBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn avg_pool2d_backward(
        x: FloatTensor<TrackingBackend>,
        grad: FloatTensor<TrackingBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn adaptive_avg_pool2d(
        x: FloatTensor<TrackingBackend>,
        output_size: [usize; 2],
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<TrackingBackend>,
        grad: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn max_pool2d(
        x: FloatTensor<TrackingBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<TrackingBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<TrackingBackend> {
        todo!()
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
        todo!()
    }

    fn interpolate(
        x: FloatTensor<TrackingBackend>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn interpolate_backward(
        x: FloatTensor<TrackingBackend>,
        grad: FloatTensor<TrackingBackend>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }
}

impl FloatTensorOps<TrackingBackend> for TrackingBackend {
    fn float_from_data(
        data: TensorData,
        device: &Device<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_into_data(
        tensor: FloatTensor<TrackingBackend>,
    ) -> impl Future<Output = TensorData> + Send {
        todo!()
    }

    fn float_device(tensor: &FloatTensor<TrackingBackend>) -> Device<TrackingBackend> {
        todo!()
    }

    fn float_to_device(
        tensor: FloatTensor<TrackingBackend>,
        device: &Device<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_into_int(tensor: FloatTensor<TrackingBackend>) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn float_empty(
        shape: Shape,
        device: &Device<TrackingBackend>,
        dtype: FloatDType,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_add(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_add_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_sub(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_sub_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_mul(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_mul_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_div(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_div_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_remainder(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_remainder_scalar(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_matmul(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_cross(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_recip(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_swap_dims(
        tensor: FloatTensor<TrackingBackend>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_permute(
        tensor: FloatTensor<TrackingBackend>,
        axes: &[usize],
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_flip(
        tensor: FloatTensor<TrackingBackend>,
        axes: &[usize],
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_reshape(
        tensor: FloatTensor<TrackingBackend>,
        shape: Shape,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_scatter(
        dim: usize,
        tensor: FloatTensor<TrackingBackend>,
        indices: IntTensor<TrackingBackend>,
        value: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_select(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_select_assign(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
        value: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_slice(
        tensor: FloatTensor<TrackingBackend>,
        slices: &[Slice],
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_slice_assign(
        tensor: FloatTensor<TrackingBackend>,
        slices: &[Slice],
        value: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_mask_where(
        tensor: FloatTensor<TrackingBackend>,
        mask: BoolTensor<TrackingBackend>,
        value: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_mask_fill(
        tensor: FloatTensor<TrackingBackend>,
        mask: BoolTensor<TrackingBackend>,
        value: FloatElem<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_equal(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn float_equal_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn float_greater(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn float_greater_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn float_greater_equal(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn float_greater_equal_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn float_lower(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn float_lower_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn float_lower_equal(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn float_lower_equal_elem(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatElem<TrackingBackend>,
    ) -> BoolTensor<TrackingBackend> {
        todo!()
    }

    fn float_sum(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_sum_dim(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_mean_dim(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_cumsum(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_cast(
        tensor: FloatTensor<TrackingBackend>,
        dtype: FloatDType,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_exp(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_log(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_log1p(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_powf(
        lhs: FloatTensor<TrackingBackend>,
        rhs: FloatTensor<TrackingBackend>,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_powf_scalar(
        tensor: FloatTensor<TrackingBackend>,
        value: f32,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_sqrt(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_abs(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_cos(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_sin(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_round(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_floor(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_ceil(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_erf(tensor: FloatTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_argmax(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn float_argmin(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
    ) -> IntTensor<TrackingBackend> {
        todo!()
    }

    fn float_expand(
        tensor: FloatTensor<TrackingBackend>,
        shape: Shape,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn float_unfold(
        tensor: FloatTensor<TrackingBackend>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<TrackingBackend> {
        todo!()
    }
}

impl QTensorOps<TrackingBackend> for TrackingBackend {
    fn q_from_data(
        data: TensorData,
        device: &Device<TrackingBackend>,
    ) -> QuantizedTensor<TrackingBackend> {
        todo!()
    }

    fn quantize(
        tensor: FloatTensor<TrackingBackend>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<TrackingBackend>,
    ) -> QuantizedTensor<TrackingBackend> {
        todo!()
    }

    fn dequantize(tensor: QuantizedTensor<TrackingBackend>) -> FloatTensor<TrackingBackend> {
        todo!()
    }

    fn q_device(tensor: &QuantizedTensor<TrackingBackend>) -> Device<TrackingBackend> {
        todo!()
    }

    fn q_to_device(
        tensor: QuantizedTensor<TrackingBackend>,
        device: &Device<TrackingBackend>,
    ) -> QuantizedTensor<TrackingBackend> {
        todo!()
    }

    fn q_reshape(
        tensor: QuantizedTensor<TrackingBackend>,
        shape: Shape,
    ) -> QuantizedTensor<TrackingBackend> {
        todo!()
    }

    fn q_into_data(
        tensor: QuantizedTensor<TrackingBackend>,
    ) -> impl Future<Output = TensorData> + Send {
        todo!()
    }

    fn q_expand(
        tensor: QuantizedTensor<TrackingBackend>,
        shape: Shape,
    ) -> QuantizedTensor<TrackingBackend> {
        todo!()
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<TrackingBackend>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<TrackingBackend> {
        todo!()
    }

    fn q_permute(
        tensor: QuantizedTensor<TrackingBackend>,
        axes: &[usize],
    ) -> QuantizedTensor<TrackingBackend> {
        todo!()
    }

    fn q_flip(
        tensor: QuantizedTensor<TrackingBackend>,
        axes: &[usize],
    ) -> QuantizedTensor<TrackingBackend> {
        todo!()
    }

    fn q_select(
        tensor: QuantizedTensor<TrackingBackend>,
        dim: usize,
        indices: IntTensor<TrackingBackend>,
    ) -> QuantizedTensor<TrackingBackend> {
        todo!()
    }

    fn q_slice(
        tensor: QuantizedTensor<TrackingBackend>,
        slices: &[Slice],
    ) -> QuantizedTensor<TrackingBackend> {
        todo!()
    }
}

impl TransactionOps<TrackingBackend> for TrackingBackend {
    fn tr_execute(
        transaction: TransactionPrimitive<TrackingBackend>,
    ) -> impl Future<Output = TransactionPrimitiveResult> + Send {
        InnerBackend::tr_execute(unsafe { transmute(transaction) })
    }
}
