/// Convolution module
pub mod conv {
    mod conv1d {
        use alloc::format;
        use burn_core as burn;
        use crate::{PaddingConfig1d, conv::checks};
        use burn::tensor::{Tensor, backend::Backend, module::conv1d, ops::ConvOptions};
        use burn::{
            config::Config,
            module::{
                Content, DisplaySettings, Ignored, Initializer, Module, ModuleDisplay,
                Param,
            },
        };
        /// Configuration to create a [1D convolution](Conv1d) layer using the [init function](Conv1dConfig::init).
        pub struct Conv1dConfig {
            /// The number of input channels.
            pub channels_in: usize,
            /// The number of output channels.
            pub channels_out: usize,
            /// The size of the kernel.
            pub kernel_size: usize,
            /// The stride of the convolution.
            #[config(default = "1")]
            pub stride: usize,
            /// Spacing between kernel elements.
            #[config(default = "1")]
            pub dilation: usize,
            /// Controls the connections between input and output channels.
            #[config(default = "1")]
            pub groups: usize,
            /// The padding configuration.
            ///
            /// ### Warning
            /// Only symmetric padding is currently supported. As such, using `Same` padding with an even kernel
            /// size is not supported as it will not produce the same output size.
            #[config(default = "PaddingConfig1d::Valid")]
            pub padding: PaddingConfig1d,
            /// If bias should be added to the output.
            #[config(default = true)]
            pub bias: bool,
            /// The type of function used to initialize neural network parameters
            #[config(
                default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
            )]
            pub initializer: Initializer,
        }
        impl burn::config::Config for Conv1dConfig {}
        impl Conv1dConfig {
            /// Create a new instance of the config.
            #[allow(clippy::too_many_arguments)]
            pub fn new(
                channels_in: usize,
                channels_out: usize,
                kernel_size: usize,
            ) -> Self {
                Self {
                    channels_in: channels_in,
                    channels_out: channels_out,
                    kernel_size: kernel_size,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    padding: PaddingConfig1d::Valid,
                    bias: true,
                    initializer: Initializer::KaimingUniform {
                        gain: 1.0 / num_traits::Float::sqrt(3.0),
                        fan_out_only: false,
                    },
                }
            }
        }
        impl Conv1dConfig {
            /// The stride of the convolution.
            pub fn with_stride(mut self, stride: usize) -> Self {
                self.stride = stride;
                self
            }
            /// Spacing between kernel elements.
            pub fn with_dilation(mut self, dilation: usize) -> Self {
                self.dilation = dilation;
                self
            }
            /// Controls the connections between input and output channels.
            pub fn with_groups(mut self, groups: usize) -> Self {
                self.groups = groups;
                self
            }
            /// The padding configuration.
            pub fn with_padding(mut self, padding: PaddingConfig1d) -> Self {
                self.padding = padding;
                self
            }
            /// If bias should be added to the output.
            pub fn with_bias(mut self, bias: bool) -> Self {
                self.bias = bias;
                self
            }
            /// The type of function used to initialize neural network parameters
            pub fn with_initializer(mut self, initializer: Initializer) -> Self {
                self.initializer = initializer;
                self
            }
        }
        impl burn::serde::Serialize for Conv1dConfig {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: burn::serde::Serializer,
            {
                #[serde(crate = "burn::serde")]
                struct Conv1dConfigSerde {
                    channels_in: usize,
                    channels_out: usize,
                    kernel_size: usize,
                    stride: usize,
                    dilation: usize,
                    groups: usize,
                    padding: PaddingConfig1d,
                    bias: bool,
                    initializer: Initializer,
                }
                #[doc(hidden)]
                #[allow(
                    non_upper_case_globals,
                    unused_attributes,
                    unused_qualifications,
                    clippy::absolute_paths,
                )]
                const _: () = {
                    use burn::serde as _serde;
                    #[automatically_derived]
                    impl _serde::Serialize for Conv1dConfigSerde {
                        fn serialize<__S>(
                            &self,
                            __serializer: __S,
                        ) -> _serde::__private226::Result<__S::Ok, __S::Error>
                        where
                            __S: _serde::Serializer,
                        {
                            let mut __serde_state = _serde::Serializer::serialize_struct(
                                __serializer,
                                "Conv1dConfigSerde",
                                false as usize + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "channels_in",
                                &self.channels_in,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "channels_out",
                                &self.channels_out,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "kernel_size",
                                &self.kernel_size,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "stride",
                                &self.stride,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "dilation",
                                &self.dilation,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "groups",
                                &self.groups,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "padding",
                                &self.padding,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "bias",
                                &self.bias,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "initializer",
                                &self.initializer,
                            )?;
                            _serde::ser::SerializeStruct::end(__serde_state)
                        }
                    }
                };
                let serde_state = Conv1dConfigSerde {
                    channels_in: self.channels_in.clone(),
                    channels_out: self.channels_out.clone(),
                    kernel_size: self.kernel_size.clone(),
                    stride: self.stride.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                    bias: self.bias.clone(),
                    initializer: self.initializer.clone(),
                };
                serde_state.serialize(serializer)
            }
        }
        impl<'de> burn::serde::Deserialize<'de> for Conv1dConfig {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: burn::serde::Deserializer<'de>,
            {
                #[serde(crate = "burn::serde")]
                struct Conv1dConfigSerde {
                    channels_in: usize,
                    channels_out: usize,
                    kernel_size: usize,
                    stride: usize,
                    dilation: usize,
                    groups: usize,
                    padding: PaddingConfig1d,
                    bias: bool,
                    initializer: Initializer,
                }
                #[doc(hidden)]
                #[allow(
                    non_upper_case_globals,
                    unused_attributes,
                    unused_qualifications,
                    clippy::absolute_paths,
                )]
                const _: () = {
                    use burn::serde as _serde;
                    #[automatically_derived]
                    impl<'de> _serde::Deserialize<'de> for Conv1dConfigSerde {
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private226::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            #[allow(non_camel_case_types)]
                            #[doc(hidden)]
                            enum __Field {
                                __field0,
                                __field1,
                                __field2,
                                __field3,
                                __field4,
                                __field5,
                                __field6,
                                __field7,
                                __field8,
                                __ignore,
                            }
                            #[doc(hidden)]
                            struct __FieldVisitor;
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                type Value = __Field;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private226::Formatter,
                                ) -> _serde::__private226::fmt::Result {
                                    _serde::__private226::Formatter::write_str(
                                        __formatter,
                                        "field identifier",
                                    )
                                }
                                fn visit_u64<__E>(
                                    self,
                                    __value: u64,
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        0u64 => _serde::__private226::Ok(__Field::__field0),
                                        1u64 => _serde::__private226::Ok(__Field::__field1),
                                        2u64 => _serde::__private226::Ok(__Field::__field2),
                                        3u64 => _serde::__private226::Ok(__Field::__field3),
                                        4u64 => _serde::__private226::Ok(__Field::__field4),
                                        5u64 => _serde::__private226::Ok(__Field::__field5),
                                        6u64 => _serde::__private226::Ok(__Field::__field6),
                                        7u64 => _serde::__private226::Ok(__Field::__field7),
                                        8u64 => _serde::__private226::Ok(__Field::__field8),
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_str<__E>(
                                    self,
                                    __value: &str,
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        "channels_in" => _serde::__private226::Ok(__Field::__field0),
                                        "channels_out" => {
                                            _serde::__private226::Ok(__Field::__field1)
                                        }
                                        "kernel_size" => _serde::__private226::Ok(__Field::__field2),
                                        "stride" => _serde::__private226::Ok(__Field::__field3),
                                        "dilation" => _serde::__private226::Ok(__Field::__field4),
                                        "groups" => _serde::__private226::Ok(__Field::__field5),
                                        "padding" => _serde::__private226::Ok(__Field::__field6),
                                        "bias" => _serde::__private226::Ok(__Field::__field7),
                                        "initializer" => _serde::__private226::Ok(__Field::__field8),
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_bytes<__E>(
                                    self,
                                    __value: &[u8],
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        b"channels_in" => {
                                            _serde::__private226::Ok(__Field::__field0)
                                        }
                                        b"channels_out" => {
                                            _serde::__private226::Ok(__Field::__field1)
                                        }
                                        b"kernel_size" => {
                                            _serde::__private226::Ok(__Field::__field2)
                                        }
                                        b"stride" => _serde::__private226::Ok(__Field::__field3),
                                        b"dilation" => _serde::__private226::Ok(__Field::__field4),
                                        b"groups" => _serde::__private226::Ok(__Field::__field5),
                                        b"padding" => _serde::__private226::Ok(__Field::__field6),
                                        b"bias" => _serde::__private226::Ok(__Field::__field7),
                                        b"initializer" => {
                                            _serde::__private226::Ok(__Field::__field8)
                                        }
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                            }
                            #[automatically_derived]
                            impl<'de> _serde::Deserialize<'de> for __Field {
                                #[inline]
                                fn deserialize<__D>(
                                    __deserializer: __D,
                                ) -> _serde::__private226::Result<Self, __D::Error>
                                where
                                    __D: _serde::Deserializer<'de>,
                                {
                                    _serde::Deserializer::deserialize_identifier(
                                        __deserializer,
                                        __FieldVisitor,
                                    )
                                }
                            }
                            #[doc(hidden)]
                            struct __Visitor<'de> {
                                marker: _serde::__private226::PhantomData<
                                    Conv1dConfigSerde,
                                >,
                                lifetime: _serde::__private226::PhantomData<&'de ()>,
                            }
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                type Value = Conv1dConfigSerde;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private226::Formatter,
                                ) -> _serde::__private226::fmt::Result {
                                    _serde::__private226::Formatter::write_str(
                                        __formatter,
                                        "struct Conv1dConfigSerde",
                                    )
                                }
                                #[inline]
                                fn visit_seq<__A>(
                                    self,
                                    mut __seq: __A,
                                ) -> _serde::__private226::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::SeqAccess<'de>,
                                {
                                    let __field0 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    0usize,
                                                    &"struct Conv1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field1 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    1usize,
                                                    &"struct Conv1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field2 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    2usize,
                                                    &"struct Conv1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field3 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    3usize,
                                                    &"struct Conv1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field4 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    4usize,
                                                    &"struct Conv1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field5 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    5usize,
                                                    &"struct Conv1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field6 = match _serde::de::SeqAccess::next_element::<
                                        PaddingConfig1d,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    6usize,
                                                    &"struct Conv1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field7 = match _serde::de::SeqAccess::next_element::<
                                        bool,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    7usize,
                                                    &"struct Conv1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field8 = match _serde::de::SeqAccess::next_element::<
                                        Initializer,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    8usize,
                                                    &"struct Conv1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    _serde::__private226::Ok(Conv1dConfigSerde {
                                        channels_in: __field0,
                                        channels_out: __field1,
                                        kernel_size: __field2,
                                        stride: __field3,
                                        dilation: __field4,
                                        groups: __field5,
                                        padding: __field6,
                                        bias: __field7,
                                        initializer: __field8,
                                    })
                                }
                                #[inline]
                                fn visit_map<__A>(
                                    self,
                                    mut __map: __A,
                                ) -> _serde::__private226::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::MapAccess<'de>,
                                {
                                    let mut __field0: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field1: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field2: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field3: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field4: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field5: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field6: _serde::__private226::Option<
                                        PaddingConfig1d,
                                    > = _serde::__private226::None;
                                    let mut __field7: _serde::__private226::Option<bool> = _serde::__private226::None;
                                    let mut __field8: _serde::__private226::Option<
                                        Initializer,
                                    > = _serde::__private226::None;
                                    while let _serde::__private226::Some(__key) = _serde::de::MapAccess::next_key::<
                                        __Field,
                                    >(&mut __map)? {
                                        match __key {
                                            __Field::__field0 => {
                                                if _serde::__private226::Option::is_some(&__field0) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "channels_in",
                                                        ),
                                                    );
                                                }
                                                __field0 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field1 => {
                                                if _serde::__private226::Option::is_some(&__field1) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "channels_out",
                                                        ),
                                                    );
                                                }
                                                __field1 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field2 => {
                                                if _serde::__private226::Option::is_some(&__field2) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "kernel_size",
                                                        ),
                                                    );
                                                }
                                                __field2 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field3 => {
                                                if _serde::__private226::Option::is_some(&__field3) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                    );
                                                }
                                                __field3 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field4 => {
                                                if _serde::__private226::Option::is_some(&__field4) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "dilation",
                                                        ),
                                                    );
                                                }
                                                __field4 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field5 => {
                                                if _serde::__private226::Option::is_some(&__field5) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("groups"),
                                                    );
                                                }
                                                __field5 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field6 => {
                                                if _serde::__private226::Option::is_some(&__field6) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "padding",
                                                        ),
                                                    );
                                                }
                                                __field6 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<
                                                        PaddingConfig1d,
                                                    >(&mut __map)?,
                                                );
                                            }
                                            __Field::__field7 => {
                                                if _serde::__private226::Option::is_some(&__field7) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("bias"),
                                                    );
                                                }
                                                __field7 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<bool>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field8 => {
                                                if _serde::__private226::Option::is_some(&__field8) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "initializer",
                                                        ),
                                                    );
                                                }
                                                __field8 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<
                                                        Initializer,
                                                    >(&mut __map)?,
                                                );
                                            }
                                            _ => {
                                                let _ = _serde::de::MapAccess::next_value::<
                                                    _serde::de::IgnoredAny,
                                                >(&mut __map)?;
                                            }
                                        }
                                    }
                                    let __field0 = match __field0 {
                                        _serde::__private226::Some(__field0) => __field0,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("channels_in")?
                                        }
                                    };
                                    let __field1 = match __field1 {
                                        _serde::__private226::Some(__field1) => __field1,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("channels_out")?
                                        }
                                    };
                                    let __field2 = match __field2 {
                                        _serde::__private226::Some(__field2) => __field2,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("kernel_size")?
                                        }
                                    };
                                    let __field3 = match __field3 {
                                        _serde::__private226::Some(__field3) => __field3,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("stride")?
                                        }
                                    };
                                    let __field4 = match __field4 {
                                        _serde::__private226::Some(__field4) => __field4,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("dilation")?
                                        }
                                    };
                                    let __field5 = match __field5 {
                                        _serde::__private226::Some(__field5) => __field5,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("groups")?
                                        }
                                    };
                                    let __field6 = match __field6 {
                                        _serde::__private226::Some(__field6) => __field6,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("padding")?
                                        }
                                    };
                                    let __field7 = match __field7 {
                                        _serde::__private226::Some(__field7) => __field7,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("bias")?
                                        }
                                    };
                                    let __field8 = match __field8 {
                                        _serde::__private226::Some(__field8) => __field8,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("initializer")?
                                        }
                                    };
                                    _serde::__private226::Ok(Conv1dConfigSerde {
                                        channels_in: __field0,
                                        channels_out: __field1,
                                        kernel_size: __field2,
                                        stride: __field3,
                                        dilation: __field4,
                                        groups: __field5,
                                        padding: __field6,
                                        bias: __field7,
                                        initializer: __field8,
                                    })
                                }
                            }
                            #[doc(hidden)]
                            const FIELDS: &'static [&'static str] = &[
                                "channels_in",
                                "channels_out",
                                "kernel_size",
                                "stride",
                                "dilation",
                                "groups",
                                "padding",
                                "bias",
                                "initializer",
                            ];
                            _serde::Deserializer::deserialize_struct(
                                __deserializer,
                                "Conv1dConfigSerde",
                                FIELDS,
                                __Visitor {
                                    marker: _serde::__private226::PhantomData::<
                                        Conv1dConfigSerde,
                                    >,
                                    lifetime: _serde::__private226::PhantomData,
                                },
                            )
                        }
                    }
                };
                let serde_state = Conv1dConfigSerde::deserialize(deserializer)?;
                Ok(Conv1dConfig {
                    channels_in: serde_state.channels_in,
                    channels_out: serde_state.channels_out,
                    kernel_size: serde_state.kernel_size,
                    stride: serde_state.stride,
                    dilation: serde_state.dilation,
                    groups: serde_state.groups,
                    padding: serde_state.padding,
                    bias: serde_state.bias,
                    initializer: serde_state.initializer,
                })
            }
        }
        impl Clone for Conv1dConfig {
            fn clone(&self) -> Self {
                Self {
                    channels_in: self.channels_in.clone(),
                    channels_out: self.channels_out.clone(),
                    kernel_size: self.kernel_size.clone(),
                    stride: self.stride.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                    bias: self.bias.clone(),
                    initializer: self.initializer.clone(),
                }
            }
        }
        impl core::fmt::Display for Conv1dConfig {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(&burn::config::config_to_json(self))
            }
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for Conv1dConfig {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                let names: &'static _ = &[
                    "channels_in",
                    "channels_out",
                    "kernel_size",
                    "stride",
                    "dilation",
                    "groups",
                    "padding",
                    "bias",
                    "initializer",
                ];
                let values: &[&dyn ::core::fmt::Debug] = &[
                    &self.channels_in,
                    &self.channels_out,
                    &self.kernel_size,
                    &self.stride,
                    &self.dilation,
                    &self.groups,
                    &self.padding,
                    &self.bias,
                    &&self.initializer,
                ];
                ::core::fmt::Formatter::debug_struct_fields_finish(
                    f,
                    "Conv1dConfig",
                    names,
                    values,
                )
            }
        }
        /// Applies a 1D convolution over input tensors.
        ///
        /// Should be created with [Conv1dConfig].
        #[module(custom_display)]
        pub struct Conv1d<B: Backend> {
            /// Tensor of shape `[channels_out, channels_in / groups, kernel_size]`
            pub weight: Param<Tensor<B, 3>>,
            /// Tensor of shape `[channels_out]`
            pub bias: Option<Param<Tensor<B, 1>>>,
            /// Stride of the convolution.
            pub stride: usize,
            /// Size of the kernel.
            pub kernel_size: usize,
            /// Spacing between kernel elements.
            pub dilation: usize,
            /// Controls the connections between input and output channels.
            pub groups: usize,
            /// Padding configuration.
            pub padding: Ignored<PaddingConfig1d>,
        }
        impl<B: Backend> burn::module::Module<B> for Conv1d<B> {
            type Record = Conv1dRecord<B>;
            fn load_record(self, record: Self::Record) -> Self {
                Self {
                    weight: burn::module::Module::<
                        B,
                    >::load_record(self.weight, record.weight),
                    bias: burn::module::Module::<B>::load_record(self.bias, record.bias),
                    stride: burn::module::Module::<
                        B,
                    >::load_record(self.stride, record.stride),
                    kernel_size: burn::module::Module::<
                        B,
                    >::load_record(self.kernel_size, record.kernel_size),
                    dilation: burn::module::Module::<
                        B,
                    >::load_record(self.dilation, record.dilation),
                    groups: burn::module::Module::<
                        B,
                    >::load_record(self.groups, record.groups),
                    padding: burn::module::Module::<
                        B,
                    >::load_record(self.padding, record.padding),
                }
            }
            fn into_record(self) -> Self::Record {
                Self::Record {
                    weight: burn::module::Module::<B>::into_record(self.weight),
                    bias: burn::module::Module::<B>::into_record(self.bias),
                    stride: burn::module::Module::<B>::into_record(self.stride),
                    kernel_size: burn::module::Module::<
                        B,
                    >::into_record(self.kernel_size),
                    dilation: burn::module::Module::<B>::into_record(self.dilation),
                    groups: burn::module::Module::<B>::into_record(self.groups),
                    padding: burn::module::Module::<B>::into_record(self.padding),
                }
            }
            fn num_params(&self) -> usize {
                let mut num_params = 0;
                num_params += burn::module::Module::<B>::num_params(&self.weight);
                num_params += burn::module::Module::<B>::num_params(&self.bias);
                num_params += burn::module::Module::<B>::num_params(&self.stride);
                num_params += burn::module::Module::<B>::num_params(&self.kernel_size);
                num_params += burn::module::Module::<B>::num_params(&self.dilation);
                num_params += burn::module::Module::<B>::num_params(&self.groups);
                num_params += burn::module::Module::<B>::num_params(&self.padding);
                num_params
            }
            fn visit<Visitor: burn::module::ModuleVisitor<B>>(
                &self,
                visitor: &mut Visitor,
            ) {
                visitor.enter_module("weight", "Conv1d");
                burn::module::Module::visit(&self.weight, visitor);
                visitor.exit_module("weight", "Conv1d");
                visitor.enter_module("bias", "Conv1d");
                burn::module::Module::visit(&self.bias, visitor);
                visitor.exit_module("bias", "Conv1d");
                visitor.enter_module("stride", "Conv1d");
                burn::module::Module::visit(&self.stride, visitor);
                visitor.exit_module("stride", "Conv1d");
                visitor.enter_module("kernel_size", "Conv1d");
                burn::module::Module::visit(&self.kernel_size, visitor);
                visitor.exit_module("kernel_size", "Conv1d");
                visitor.enter_module("dilation", "Conv1d");
                burn::module::Module::visit(&self.dilation, visitor);
                visitor.exit_module("dilation", "Conv1d");
                visitor.enter_module("groups", "Conv1d");
                burn::module::Module::visit(&self.groups, visitor);
                visitor.exit_module("groups", "Conv1d");
                visitor.enter_module("padding", "Conv1d");
                burn::module::Module::visit(&self.padding, visitor);
                visitor.exit_module("padding", "Conv1d");
            }
            fn map<Mapper: burn::module::ModuleMapper<B>>(
                self,
                mapper: &mut Mapper,
            ) -> Self {
                mapper.enter_module("weight", "Conv1d");
                let weight = burn::module::Module::<B>::map(self.weight, mapper);
                mapper.exit_module("weight", "Conv1d");
                mapper.enter_module("bias", "Conv1d");
                let bias = burn::module::Module::<B>::map(self.bias, mapper);
                mapper.exit_module("bias", "Conv1d");
                mapper.enter_module("stride", "Conv1d");
                let stride = burn::module::Module::<B>::map(self.stride, mapper);
                mapper.exit_module("stride", "Conv1d");
                mapper.enter_module("kernel_size", "Conv1d");
                let kernel_size = burn::module::Module::<
                    B,
                >::map(self.kernel_size, mapper);
                mapper.exit_module("kernel_size", "Conv1d");
                mapper.enter_module("dilation", "Conv1d");
                let dilation = burn::module::Module::<B>::map(self.dilation, mapper);
                mapper.exit_module("dilation", "Conv1d");
                mapper.enter_module("groups", "Conv1d");
                let groups = burn::module::Module::<B>::map(self.groups, mapper);
                mapper.exit_module("groups", "Conv1d");
                mapper.enter_module("padding", "Conv1d");
                let padding = burn::module::Module::<B>::map(self.padding, mapper);
                mapper.exit_module("padding", "Conv1d");
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                }
            }
            fn collect_devices(
                &self,
                devices: burn::module::Devices<B>,
            ) -> burn::module::Devices<B> {
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.weight, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.bias, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.stride, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.kernel_size, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.dilation, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.groups, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.padding, devices);
                devices
            }
            fn to_device(self, device: &B::Device) -> Self {
                let weight = burn::module::Module::<B>::to_device(self.weight, device);
                let bias = burn::module::Module::<B>::to_device(self.bias, device);
                let stride = burn::module::Module::<B>::to_device(self.stride, device);
                let kernel_size = burn::module::Module::<
                    B,
                >::to_device(self.kernel_size, device);
                let dilation = burn::module::Module::<
                    B,
                >::to_device(self.dilation, device);
                let groups = burn::module::Module::<B>::to_device(self.groups, device);
                let padding = burn::module::Module::<B>::to_device(self.padding, device);
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                }
            }
            fn fork(self, device: &B::Device) -> Self {
                let weight = burn::module::Module::<B>::fork(self.weight, device);
                let bias = burn::module::Module::<B>::fork(self.bias, device);
                let stride = burn::module::Module::<B>::fork(self.stride, device);
                let kernel_size = burn::module::Module::<
                    B,
                >::fork(self.kernel_size, device);
                let dilation = burn::module::Module::<B>::fork(self.dilation, device);
                let groups = burn::module::Module::<B>::fork(self.groups, device);
                let padding = burn::module::Module::<B>::fork(self.padding, device);
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                }
            }
        }
        impl<B: Backend> burn::module::AutodiffModule<B> for Conv1d<B>
        where
            B: burn::tensor::backend::AutodiffBackend,
            <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
        {
            type InnerModule = Conv1d<B::InnerBackend>;
            fn valid(&self) -> Self::InnerModule {
                let weight = burn::module::AutodiffModule::<B>::valid(&self.weight);
                let bias = burn::module::AutodiffModule::<B>::valid(&self.bias);
                let stride = burn::module::AutodiffModule::<B>::valid(&self.stride);
                let kernel_size = burn::module::AutodiffModule::<
                    B,
                >::valid(&self.kernel_size);
                let dilation = burn::module::AutodiffModule::<B>::valid(&self.dilation);
                let groups = burn::module::AutodiffModule::<B>::valid(&self.groups);
                let padding = burn::module::AutodiffModule::<B>::valid(&self.padding);
                Self::InnerModule {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                }
            }
        }
        impl<B: Backend> core::fmt::Display for Conv1d<B> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let formatted = burn::module::ModuleDisplay::format(
                    self,
                    Default::default(),
                );
                f.write_fmt(format_args!("{0}", formatted))
            }
        }
        impl<B: Backend> burn::module::ModuleDisplayDefault for Conv1d<B> {
            fn content(
                &self,
                mut content: burn::module::Content,
            ) -> Option<burn::module::Content> {
                content
                    .set_top_level_type(&"Conv1d")
                    .add("weight", &self.weight)
                    .add("bias", &self.bias)
                    .add("stride", &self.stride)
                    .add("kernel_size", &self.kernel_size)
                    .add("dilation", &self.dilation)
                    .add("groups", &self.groups)
                    .add("padding", &self.padding)
                    .optional()
            }
            fn num_params(&self) -> usize {
                burn::module::Module::num_params(self)
            }
        }
        impl<B: Backend> Clone for Conv1d<B> {
            fn clone(&self) -> Self {
                let weight = self.weight.clone();
                let bias = self.bias.clone();
                let stride = self.stride.clone();
                let kernel_size = self.kernel_size.clone();
                let dilation = self.dilation.clone();
                let groups = self.groups.clone();
                let padding = self.padding.clone();
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                }
            }
        }
        /// The record type for the module.
        pub struct Conv1dRecord<B: Backend> {
            /// The module record associative type.
            pub weight: <Param<Tensor<B, 3>> as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub bias: <Option<Param<Tensor<B, 1>>> as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub stride: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub kernel_size: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub dilation: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub groups: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub padding: <Ignored<PaddingConfig1d> as burn::module::Module<B>>::Record,
        }
        /// The record item type for the module.
        #[serde(crate = "burn::serde")]
        #[serde(
            bound = "< < Param < Tensor < B, 3 > > as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize +\nburn :: serde :: de :: DeserializeOwned, < < Option < Param < Tensor < B, 1 >\n> > as burn :: module :: Module < B > > :: Record as burn :: record :: Record\n< B >> :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de ::\nDeserializeOwned, < < usize as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize +\nburn :: serde :: de :: DeserializeOwned, < < usize as burn :: module :: Module\n< B > > :: Record as burn :: record :: Record < B >> :: Item < S > : burn ::\nserde :: Serialize + burn :: serde :: de :: DeserializeOwned, < < usize as\nburn :: module :: Module < B > > :: Record as burn :: record :: Record < B >>\n:: Item < S > : burn :: serde :: Serialize + burn :: serde :: de ::\nDeserializeOwned, < < usize as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize +\nburn :: serde :: de :: DeserializeOwned, < < Ignored < PaddingConfig1d > as\nburn :: module :: Module < B > > :: Record as burn :: record :: Record < B >>\n:: Item < S > : burn :: serde :: Serialize + burn :: serde :: de ::\nDeserializeOwned,"
        )]
        pub struct Conv1dRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
            /// Field to be serialized.
            pub weight: <<Param<
                Tensor<B, 3>,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub bias: <<Option<
                Param<Tensor<B, 1>>,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub stride: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub kernel_size: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub dilation: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub groups: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub padding: <<Ignored<
                PaddingConfig1d,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
        }
        #[doc(hidden)]
        #[allow(
            non_upper_case_globals,
            unused_attributes,
            unused_qualifications,
            clippy::absolute_paths,
        )]
        const _: () = {
            use burn::serde as _serde;
            #[automatically_derived]
            impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
            for Conv1dRecordItem<B, S>
            where
                <<Param<
                    Tensor<B, 3>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Option<
                    Param<Tensor<B, 1>>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Ignored<
                    PaddingConfig1d,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            {
                fn serialize<__S>(
                    &self,
                    __serializer: __S,
                ) -> _serde::__private226::Result<__S::Ok, __S::Error>
                where
                    __S: _serde::Serializer,
                {
                    let mut __serde_state = _serde::Serializer::serialize_struct(
                        __serializer,
                        "Conv1dRecordItem",
                        false as usize + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "weight",
                        &self.weight,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "bias",
                        &self.bias,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "stride",
                        &self.stride,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "kernel_size",
                        &self.kernel_size,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "dilation",
                        &self.dilation,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "groups",
                        &self.groups,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "padding",
                        &self.padding,
                    )?;
                    _serde::ser::SerializeStruct::end(__serde_state)
                }
            }
        };
        #[doc(hidden)]
        #[allow(
            non_upper_case_globals,
            unused_attributes,
            unused_qualifications,
            clippy::absolute_paths,
        )]
        const _: () = {
            use burn::serde as _serde;
            #[automatically_derived]
            impl<
                'de,
                B: Backend,
                S: burn::record::PrecisionSettings,
            > _serde::Deserialize<'de> for Conv1dRecordItem<B, S>
            where
                <<Param<
                    Tensor<B, 3>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Option<
                    Param<Tensor<B, 1>>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Ignored<
                    PaddingConfig1d,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            {
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private226::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    #[allow(non_camel_case_types)]
                    #[doc(hidden)]
                    enum __Field {
                        __field0,
                        __field1,
                        __field2,
                        __field3,
                        __field4,
                        __field5,
                        __field6,
                        __ignore,
                    }
                    #[doc(hidden)]
                    struct __FieldVisitor;
                    #[automatically_derived]
                    impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                        type Value = __Field;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private226::Formatter,
                        ) -> _serde::__private226::fmt::Result {
                            _serde::__private226::Formatter::write_str(
                                __formatter,
                                "field identifier",
                            )
                        }
                        fn visit_u64<__E>(
                            self,
                            __value: u64,
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                0u64 => _serde::__private226::Ok(__Field::__field0),
                                1u64 => _serde::__private226::Ok(__Field::__field1),
                                2u64 => _serde::__private226::Ok(__Field::__field2),
                                3u64 => _serde::__private226::Ok(__Field::__field3),
                                4u64 => _serde::__private226::Ok(__Field::__field4),
                                5u64 => _serde::__private226::Ok(__Field::__field5),
                                6u64 => _serde::__private226::Ok(__Field::__field6),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_str<__E>(
                            self,
                            __value: &str,
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                "weight" => _serde::__private226::Ok(__Field::__field0),
                                "bias" => _serde::__private226::Ok(__Field::__field1),
                                "stride" => _serde::__private226::Ok(__Field::__field2),
                                "kernel_size" => _serde::__private226::Ok(__Field::__field3),
                                "dilation" => _serde::__private226::Ok(__Field::__field4),
                                "groups" => _serde::__private226::Ok(__Field::__field5),
                                "padding" => _serde::__private226::Ok(__Field::__field6),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_bytes<__E>(
                            self,
                            __value: &[u8],
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                b"weight" => _serde::__private226::Ok(__Field::__field0),
                                b"bias" => _serde::__private226::Ok(__Field::__field1),
                                b"stride" => _serde::__private226::Ok(__Field::__field2),
                                b"kernel_size" => {
                                    _serde::__private226::Ok(__Field::__field3)
                                }
                                b"dilation" => _serde::__private226::Ok(__Field::__field4),
                                b"groups" => _serde::__private226::Ok(__Field::__field5),
                                b"padding" => _serde::__private226::Ok(__Field::__field6),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                    }
                    #[automatically_derived]
                    impl<'de> _serde::Deserialize<'de> for __Field {
                        #[inline]
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private226::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            _serde::Deserializer::deserialize_identifier(
                                __deserializer,
                                __FieldVisitor,
                            )
                        }
                    }
                    #[doc(hidden)]
                    struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                    where
                        <<Param<
                            Tensor<B, 3>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Option<
                            Param<Tensor<B, 1>>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Ignored<
                            PaddingConfig1d,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    {
                        marker: _serde::__private226::PhantomData<
                            Conv1dRecordItem<B, S>,
                        >,
                        lifetime: _serde::__private226::PhantomData<&'de ()>,
                    }
                    #[automatically_derived]
                    impl<
                        'de,
                        B: Backend,
                        S: burn::record::PrecisionSettings,
                    > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                    where
                        <<Param<
                            Tensor<B, 3>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Option<
                            Param<Tensor<B, 1>>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Ignored<
                            PaddingConfig1d,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    {
                        type Value = Conv1dRecordItem<B, S>;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private226::Formatter,
                        ) -> _serde::__private226::fmt::Result {
                            _serde::__private226::Formatter::write_str(
                                __formatter,
                                "struct Conv1dRecordItem",
                            )
                        }
                        #[inline]
                        fn visit_seq<__A>(
                            self,
                            mut __seq: __A,
                        ) -> _serde::__private226::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::SeqAccess<'de>,
                        {
                            let __field0 = match _serde::de::SeqAccess::next_element::<
                                <<Param<
                                    Tensor<B, 3>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            0usize,
                                            &"struct Conv1dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field1 = match _serde::de::SeqAccess::next_element::<
                                <<Option<
                                    Param<Tensor<B, 1>>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            1usize,
                                            &"struct Conv1dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field2 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            2usize,
                                            &"struct Conv1dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field3 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            3usize,
                                            &"struct Conv1dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field4 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            4usize,
                                            &"struct Conv1dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field5 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            5usize,
                                            &"struct Conv1dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field6 = match _serde::de::SeqAccess::next_element::<
                                <<Ignored<
                                    PaddingConfig1d,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            6usize,
                                            &"struct Conv1dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            _serde::__private226::Ok(Conv1dRecordItem {
                                weight: __field0,
                                bias: __field1,
                                stride: __field2,
                                kernel_size: __field3,
                                dilation: __field4,
                                groups: __field5,
                                padding: __field6,
                            })
                        }
                        #[inline]
                        fn visit_map<__A>(
                            self,
                            mut __map: __A,
                        ) -> _serde::__private226::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::MapAccess<'de>,
                        {
                            let mut __field0: _serde::__private226::Option<
                                <<Param<
                                    Tensor<B, 3>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field1: _serde::__private226::Option<
                                <<Option<
                                    Param<Tensor<B, 1>>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field2: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field3: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field4: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field5: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field6: _serde::__private226::Option<
                                <<Ignored<
                                    PaddingConfig1d,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            while let _serde::__private226::Some(__key) = _serde::de::MapAccess::next_key::<
                                __Field,
                            >(&mut __map)? {
                                match __key {
                                    __Field::__field0 => {
                                        if _serde::__private226::Option::is_some(&__field0) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("weight"),
                                            );
                                        }
                                        __field0 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Param<
                                                    Tensor<B, 3>,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field1 => {
                                        if _serde::__private226::Option::is_some(&__field1) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("bias"),
                                            );
                                        }
                                        __field1 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Option<
                                                    Param<Tensor<B, 1>>,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field2 => {
                                        if _serde::__private226::Option::is_some(&__field2) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                            );
                                        }
                                        __field2 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field3 => {
                                        if _serde::__private226::Option::is_some(&__field3) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "kernel_size",
                                                ),
                                            );
                                        }
                                        __field3 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field4 => {
                                        if _serde::__private226::Option::is_some(&__field4) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "dilation",
                                                ),
                                            );
                                        }
                                        __field4 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field5 => {
                                        if _serde::__private226::Option::is_some(&__field5) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("groups"),
                                            );
                                        }
                                        __field5 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field6 => {
                                        if _serde::__private226::Option::is_some(&__field6) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "padding",
                                                ),
                                            );
                                        }
                                        __field6 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Ignored<
                                                    PaddingConfig1d,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    _ => {
                                        let _ = _serde::de::MapAccess::next_value::<
                                            _serde::de::IgnoredAny,
                                        >(&mut __map)?;
                                    }
                                }
                            }
                            let __field0 = match __field0 {
                                _serde::__private226::Some(__field0) => __field0,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("weight")?
                                }
                            };
                            let __field1 = match __field1 {
                                _serde::__private226::Some(__field1) => __field1,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("bias")?
                                }
                            };
                            let __field2 = match __field2 {
                                _serde::__private226::Some(__field2) => __field2,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("stride")?
                                }
                            };
                            let __field3 = match __field3 {
                                _serde::__private226::Some(__field3) => __field3,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("kernel_size")?
                                }
                            };
                            let __field4 = match __field4 {
                                _serde::__private226::Some(__field4) => __field4,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("dilation")?
                                }
                            };
                            let __field5 = match __field5 {
                                _serde::__private226::Some(__field5) => __field5,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("groups")?
                                }
                            };
                            let __field6 = match __field6 {
                                _serde::__private226::Some(__field6) => __field6,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("padding")?
                                }
                            };
                            _serde::__private226::Ok(Conv1dRecordItem {
                                weight: __field0,
                                bias: __field1,
                                stride: __field2,
                                kernel_size: __field3,
                                dilation: __field4,
                                groups: __field5,
                                padding: __field6,
                            })
                        }
                    }
                    #[doc(hidden)]
                    const FIELDS: &'static [&'static str] = &[
                        "weight",
                        "bias",
                        "stride",
                        "kernel_size",
                        "dilation",
                        "groups",
                        "padding",
                    ];
                    _serde::Deserializer::deserialize_struct(
                        __deserializer,
                        "Conv1dRecordItem",
                        FIELDS,
                        __Visitor {
                            marker: _serde::__private226::PhantomData::<
                                Conv1dRecordItem<B, S>,
                            >,
                            lifetime: _serde::__private226::PhantomData,
                        },
                    )
                }
            }
        };
        impl<B: Backend, S: burn::record::PrecisionSettings> Clone
        for Conv1dRecordItem<B, S> {
            fn clone(&self) -> Self {
                Self {
                    weight: self.weight.clone(),
                    bias: self.bias.clone(),
                    stride: self.stride.clone(),
                    kernel_size: self.kernel_size.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                }
            }
        }
        impl<B: Backend> burn::record::Record<B> for Conv1dRecord<B> {
            type Item<S: burn::record::PrecisionSettings> = Conv1dRecordItem<B, S>;
            fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
                Conv1dRecordItem {
                    weight: burn::record::Record::<B>::into_item::<S>(self.weight),
                    bias: burn::record::Record::<B>::into_item::<S>(self.bias),
                    stride: burn::record::Record::<B>::into_item::<S>(self.stride),
                    kernel_size: burn::record::Record::<
                        B,
                    >::into_item::<S>(self.kernel_size),
                    dilation: burn::record::Record::<B>::into_item::<S>(self.dilation),
                    groups: burn::record::Record::<B>::into_item::<S>(self.groups),
                    padding: burn::record::Record::<B>::into_item::<S>(self.padding),
                }
            }
            fn from_item<S: burn::record::PrecisionSettings>(
                item: Self::Item<S>,
                device: &B::Device,
            ) -> Self {
                Self {
                    weight: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.weight, device),
                    bias: burn::record::Record::<B>::from_item::<S>(item.bias, device),
                    stride: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.stride, device),
                    kernel_size: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.kernel_size, device),
                    dilation: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.dilation, device),
                    groups: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.groups, device),
                    padding: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.padding, device),
                }
            }
        }
        #[automatically_derived]
        impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Conv1d<B> {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                let names: &'static _ = &[
                    "weight",
                    "bias",
                    "stride",
                    "kernel_size",
                    "dilation",
                    "groups",
                    "padding",
                ];
                let values: &[&dyn ::core::fmt::Debug] = &[
                    &self.weight,
                    &self.bias,
                    &self.stride,
                    &self.kernel_size,
                    &self.dilation,
                    &self.groups,
                    &&self.padding,
                ];
                ::core::fmt::Formatter::debug_struct_fields_finish(
                    f,
                    "Conv1d",
                    names,
                    values,
                )
            }
        }
        impl<B: Backend> ModuleDisplay for Conv1d<B> {
            fn custom_settings(&self) -> Option<DisplaySettings> {
                DisplaySettings::new().with_new_line_after_attribute(false).optional()
            }
            fn custom_content(&self, content: Content) -> Option<Content> {
                let padding_formatted = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0}", &self.padding))
                });
                let stride = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", self.stride))
                });
                let kernel_size = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", self.kernel_size))
                });
                let dilation = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", self.dilation))
                });
                let [channels_out, group_channels_in, _] = self.weight.dims();
                let channels_in = group_channels_in * self.groups;
                let ch_out = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", channels_out))
                });
                let ch_in = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", channels_in))
                });
                content
                    .add("ch_in", &ch_in)
                    .add("ch_out", &ch_out)
                    .add("stride", &stride)
                    .add("kernel_size", &kernel_size)
                    .add("dilation", &dilation)
                    .add("groups", &self.groups)
                    .add("padding", &padding_formatted)
                    .optional()
            }
        }
        impl Conv1dConfig {
            /// Initialize a new [conv1d](Conv1d) module.
            pub fn init<B: Backend>(&self, device: &B::Device) -> Conv1d<B> {
                checks::checks_channels_div_groups(
                    self.channels_in,
                    self.channels_out,
                    self.groups,
                );
                if self.padding == PaddingConfig1d::Same {
                    checks::check_same_padding_support(&[self.kernel_size]);
                }
                let shape = [
                    self.channels_out,
                    self.channels_in / self.groups,
                    self.kernel_size,
                ];
                let fan_in: usize = self.channels_in / self.groups * self.kernel_size;
                let weight = self
                    .initializer
                    .init_with(shape, Some(fan_in), None, device);
                let mut bias = None;
                if self.bias {
                    bias = Some(
                        self
                            .initializer
                            .init_with([self.channels_out], Some(fan_in), None, device),
                    );
                }
                Conv1d {
                    weight,
                    bias,
                    stride: self.stride,
                    kernel_size: self.kernel_size,
                    padding: Ignored(self.padding.clone()),
                    dilation: self.dilation,
                    groups: self.groups,
                }
            }
        }
        impl<B: Backend> Conv1d<B> {
            /// Applies the forward pass on the input tensor.
            ///
            /// See [conv1d](burn::tensor::module::conv1d) for more information.
            ///
            /// # Shapes
            ///
            /// - input: `[batch_size, channels_in, length_in]`
            /// - output: `[batch_size, channels_out, length_out]`
            pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
                let length = input.dims()[2];
                let padding = self
                    .padding
                    .calculate_padding_1d(length, self.kernel_size, self.stride);
                conv1d(
                    input,
                    self.weight.val(),
                    self.bias.as_ref().map(|bias| bias.val()),
                    ConvOptions::new(
                        [self.stride],
                        [padding],
                        [self.dilation],
                        self.groups,
                    ),
                )
            }
        }
    }
    mod conv2d {
        use alloc::format;
        use burn_core as burn;
        use crate::PaddingConfig2d;
        use burn::config::Config;
        use burn::module::Initializer;
        use burn::module::{
            Content, DisplaySettings, Ignored, Module, ModuleDisplay, Param,
        };
        use burn::tensor::Tensor;
        use burn::tensor::backend::Backend;
        use burn::tensor::module::conv2d;
        use burn::tensor::ops::ConvOptions;
        use crate::conv::checks;
        /// Configuration to create a [2D convolution](Conv2d) layer, using the [init function](Conv2dConfig::init).
        pub struct Conv2dConfig {
            /// The number of channels.
            pub channels: [usize; 2],
            /// The size of the kernel.
            pub kernel_size: [usize; 2],
            /// The stride of the convolution.
            #[config(default = "[1, 1]")]
            pub stride: [usize; 2],
            /// Spacing between kernel elements.
            #[config(default = "[1, 1]")]
            pub dilation: [usize; 2],
            /// Controls the connections between input and output channels.
            #[config(default = "1")]
            pub groups: usize,
            /// The padding configuration.
            ///
            /// ### Warning
            /// Only symmetric padding is currently supported. As such, using `Same` padding with an even kernel
            /// size is not supported as it will not produce the same output size.
            #[config(default = "PaddingConfig2d::Valid")]
            pub padding: PaddingConfig2d,
            /// If bias should be added to the output.
            #[config(default = true)]
            pub bias: bool,
            /// The type of function used to initialize neural network parameters
            #[config(
                default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
            )]
            pub initializer: Initializer,
        }
        impl burn::config::Config for Conv2dConfig {}
        impl Conv2dConfig {
            /// Create a new instance of the config.
            #[allow(clippy::too_many_arguments)]
            pub fn new(channels: [usize; 2], kernel_size: [usize; 2]) -> Self {
                Self {
                    channels: channels,
                    kernel_size: kernel_size,
                    stride: [1, 1],
                    dilation: [1, 1],
                    groups: 1,
                    padding: PaddingConfig2d::Valid,
                    bias: true,
                    initializer: Initializer::KaimingUniform {
                        gain: 1.0 / num_traits::Float::sqrt(3.0),
                        fan_out_only: false,
                    },
                }
            }
        }
        impl Conv2dConfig {
            /// The stride of the convolution.
            pub fn with_stride(mut self, stride: [usize; 2]) -> Self {
                self.stride = stride;
                self
            }
            /// Spacing between kernel elements.
            pub fn with_dilation(mut self, dilation: [usize; 2]) -> Self {
                self.dilation = dilation;
                self
            }
            /// Controls the connections between input and output channels.
            pub fn with_groups(mut self, groups: usize) -> Self {
                self.groups = groups;
                self
            }
            /// The padding configuration.
            pub fn with_padding(mut self, padding: PaddingConfig2d) -> Self {
                self.padding = padding;
                self
            }
            /// If bias should be added to the output.
            pub fn with_bias(mut self, bias: bool) -> Self {
                self.bias = bias;
                self
            }
            /// The type of function used to initialize neural network parameters
            pub fn with_initializer(mut self, initializer: Initializer) -> Self {
                self.initializer = initializer;
                self
            }
        }
        impl burn::serde::Serialize for Conv2dConfig {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: burn::serde::Serializer,
            {
                #[serde(crate = "burn::serde")]
                struct Conv2dConfigSerde {
                    channels: [usize; 2],
                    kernel_size: [usize; 2],
                    stride: [usize; 2],
                    dilation: [usize; 2],
                    groups: usize,
                    padding: PaddingConfig2d,
                    bias: bool,
                    initializer: Initializer,
                }
                #[doc(hidden)]
                #[allow(
                    non_upper_case_globals,
                    unused_attributes,
                    unused_qualifications,
                    clippy::absolute_paths,
                )]
                const _: () = {
                    use burn::serde as _serde;
                    #[automatically_derived]
                    impl _serde::Serialize for Conv2dConfigSerde {
                        fn serialize<__S>(
                            &self,
                            __serializer: __S,
                        ) -> _serde::__private226::Result<__S::Ok, __S::Error>
                        where
                            __S: _serde::Serializer,
                        {
                            let mut __serde_state = _serde::Serializer::serialize_struct(
                                __serializer,
                                "Conv2dConfigSerde",
                                false as usize + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "channels",
                                &self.channels,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "kernel_size",
                                &self.kernel_size,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "stride",
                                &self.stride,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "dilation",
                                &self.dilation,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "groups",
                                &self.groups,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "padding",
                                &self.padding,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "bias",
                                &self.bias,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "initializer",
                                &self.initializer,
                            )?;
                            _serde::ser::SerializeStruct::end(__serde_state)
                        }
                    }
                };
                let serde_state = Conv2dConfigSerde {
                    channels: self.channels.clone(),
                    kernel_size: self.kernel_size.clone(),
                    stride: self.stride.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                    bias: self.bias.clone(),
                    initializer: self.initializer.clone(),
                };
                serde_state.serialize(serializer)
            }
        }
        impl<'de> burn::serde::Deserialize<'de> for Conv2dConfig {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: burn::serde::Deserializer<'de>,
            {
                #[serde(crate = "burn::serde")]
                struct Conv2dConfigSerde {
                    channels: [usize; 2],
                    kernel_size: [usize; 2],
                    stride: [usize; 2],
                    dilation: [usize; 2],
                    groups: usize,
                    padding: PaddingConfig2d,
                    bias: bool,
                    initializer: Initializer,
                }
                #[doc(hidden)]
                #[allow(
                    non_upper_case_globals,
                    unused_attributes,
                    unused_qualifications,
                    clippy::absolute_paths,
                )]
                const _: () = {
                    use burn::serde as _serde;
                    #[automatically_derived]
                    impl<'de> _serde::Deserialize<'de> for Conv2dConfigSerde {
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private226::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            #[allow(non_camel_case_types)]
                            #[doc(hidden)]
                            enum __Field {
                                __field0,
                                __field1,
                                __field2,
                                __field3,
                                __field4,
                                __field5,
                                __field6,
                                __field7,
                                __ignore,
                            }
                            #[doc(hidden)]
                            struct __FieldVisitor;
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                type Value = __Field;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private226::Formatter,
                                ) -> _serde::__private226::fmt::Result {
                                    _serde::__private226::Formatter::write_str(
                                        __formatter,
                                        "field identifier",
                                    )
                                }
                                fn visit_u64<__E>(
                                    self,
                                    __value: u64,
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        0u64 => _serde::__private226::Ok(__Field::__field0),
                                        1u64 => _serde::__private226::Ok(__Field::__field1),
                                        2u64 => _serde::__private226::Ok(__Field::__field2),
                                        3u64 => _serde::__private226::Ok(__Field::__field3),
                                        4u64 => _serde::__private226::Ok(__Field::__field4),
                                        5u64 => _serde::__private226::Ok(__Field::__field5),
                                        6u64 => _serde::__private226::Ok(__Field::__field6),
                                        7u64 => _serde::__private226::Ok(__Field::__field7),
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_str<__E>(
                                    self,
                                    __value: &str,
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        "channels" => _serde::__private226::Ok(__Field::__field0),
                                        "kernel_size" => _serde::__private226::Ok(__Field::__field1),
                                        "stride" => _serde::__private226::Ok(__Field::__field2),
                                        "dilation" => _serde::__private226::Ok(__Field::__field3),
                                        "groups" => _serde::__private226::Ok(__Field::__field4),
                                        "padding" => _serde::__private226::Ok(__Field::__field5),
                                        "bias" => _serde::__private226::Ok(__Field::__field6),
                                        "initializer" => _serde::__private226::Ok(__Field::__field7),
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_bytes<__E>(
                                    self,
                                    __value: &[u8],
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        b"channels" => _serde::__private226::Ok(__Field::__field0),
                                        b"kernel_size" => {
                                            _serde::__private226::Ok(__Field::__field1)
                                        }
                                        b"stride" => _serde::__private226::Ok(__Field::__field2),
                                        b"dilation" => _serde::__private226::Ok(__Field::__field3),
                                        b"groups" => _serde::__private226::Ok(__Field::__field4),
                                        b"padding" => _serde::__private226::Ok(__Field::__field5),
                                        b"bias" => _serde::__private226::Ok(__Field::__field6),
                                        b"initializer" => {
                                            _serde::__private226::Ok(__Field::__field7)
                                        }
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                            }
                            #[automatically_derived]
                            impl<'de> _serde::Deserialize<'de> for __Field {
                                #[inline]
                                fn deserialize<__D>(
                                    __deserializer: __D,
                                ) -> _serde::__private226::Result<Self, __D::Error>
                                where
                                    __D: _serde::Deserializer<'de>,
                                {
                                    _serde::Deserializer::deserialize_identifier(
                                        __deserializer,
                                        __FieldVisitor,
                                    )
                                }
                            }
                            #[doc(hidden)]
                            struct __Visitor<'de> {
                                marker: _serde::__private226::PhantomData<
                                    Conv2dConfigSerde,
                                >,
                                lifetime: _serde::__private226::PhantomData<&'de ()>,
                            }
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                type Value = Conv2dConfigSerde;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private226::Formatter,
                                ) -> _serde::__private226::fmt::Result {
                                    _serde::__private226::Formatter::write_str(
                                        __formatter,
                                        "struct Conv2dConfigSerde",
                                    )
                                }
                                #[inline]
                                fn visit_seq<__A>(
                                    self,
                                    mut __seq: __A,
                                ) -> _serde::__private226::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::SeqAccess<'de>,
                                {
                                    let __field0 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    0usize,
                                                    &"struct Conv2dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field1 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    1usize,
                                                    &"struct Conv2dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field2 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    2usize,
                                                    &"struct Conv2dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field3 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    3usize,
                                                    &"struct Conv2dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field4 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    4usize,
                                                    &"struct Conv2dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field5 = match _serde::de::SeqAccess::next_element::<
                                        PaddingConfig2d,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    5usize,
                                                    &"struct Conv2dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field6 = match _serde::de::SeqAccess::next_element::<
                                        bool,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    6usize,
                                                    &"struct Conv2dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field7 = match _serde::de::SeqAccess::next_element::<
                                        Initializer,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    7usize,
                                                    &"struct Conv2dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    _serde::__private226::Ok(Conv2dConfigSerde {
                                        channels: __field0,
                                        kernel_size: __field1,
                                        stride: __field2,
                                        dilation: __field3,
                                        groups: __field4,
                                        padding: __field5,
                                        bias: __field6,
                                        initializer: __field7,
                                    })
                                }
                                #[inline]
                                fn visit_map<__A>(
                                    self,
                                    mut __map: __A,
                                ) -> _serde::__private226::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::MapAccess<'de>,
                                {
                                    let mut __field0: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field1: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field2: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field3: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field4: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field5: _serde::__private226::Option<
                                        PaddingConfig2d,
                                    > = _serde::__private226::None;
                                    let mut __field6: _serde::__private226::Option<bool> = _serde::__private226::None;
                                    let mut __field7: _serde::__private226::Option<
                                        Initializer,
                                    > = _serde::__private226::None;
                                    while let _serde::__private226::Some(__key) = _serde::de::MapAccess::next_key::<
                                        __Field,
                                    >(&mut __map)? {
                                        match __key {
                                            __Field::__field0 => {
                                                if _serde::__private226::Option::is_some(&__field0) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "channels",
                                                        ),
                                                    );
                                                }
                                                __field0 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field1 => {
                                                if _serde::__private226::Option::is_some(&__field1) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "kernel_size",
                                                        ),
                                                    );
                                                }
                                                __field1 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field2 => {
                                                if _serde::__private226::Option::is_some(&__field2) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                    );
                                                }
                                                __field2 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field3 => {
                                                if _serde::__private226::Option::is_some(&__field3) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "dilation",
                                                        ),
                                                    );
                                                }
                                                __field3 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field4 => {
                                                if _serde::__private226::Option::is_some(&__field4) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("groups"),
                                                    );
                                                }
                                                __field4 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field5 => {
                                                if _serde::__private226::Option::is_some(&__field5) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "padding",
                                                        ),
                                                    );
                                                }
                                                __field5 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<
                                                        PaddingConfig2d,
                                                    >(&mut __map)?,
                                                );
                                            }
                                            __Field::__field6 => {
                                                if _serde::__private226::Option::is_some(&__field6) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("bias"),
                                                    );
                                                }
                                                __field6 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<bool>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field7 => {
                                                if _serde::__private226::Option::is_some(&__field7) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "initializer",
                                                        ),
                                                    );
                                                }
                                                __field7 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<
                                                        Initializer,
                                                    >(&mut __map)?,
                                                );
                                            }
                                            _ => {
                                                let _ = _serde::de::MapAccess::next_value::<
                                                    _serde::de::IgnoredAny,
                                                >(&mut __map)?;
                                            }
                                        }
                                    }
                                    let __field0 = match __field0 {
                                        _serde::__private226::Some(__field0) => __field0,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("channels")?
                                        }
                                    };
                                    let __field1 = match __field1 {
                                        _serde::__private226::Some(__field1) => __field1,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("kernel_size")?
                                        }
                                    };
                                    let __field2 = match __field2 {
                                        _serde::__private226::Some(__field2) => __field2,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("stride")?
                                        }
                                    };
                                    let __field3 = match __field3 {
                                        _serde::__private226::Some(__field3) => __field3,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("dilation")?
                                        }
                                    };
                                    let __field4 = match __field4 {
                                        _serde::__private226::Some(__field4) => __field4,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("groups")?
                                        }
                                    };
                                    let __field5 = match __field5 {
                                        _serde::__private226::Some(__field5) => __field5,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("padding")?
                                        }
                                    };
                                    let __field6 = match __field6 {
                                        _serde::__private226::Some(__field6) => __field6,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("bias")?
                                        }
                                    };
                                    let __field7 = match __field7 {
                                        _serde::__private226::Some(__field7) => __field7,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("initializer")?
                                        }
                                    };
                                    _serde::__private226::Ok(Conv2dConfigSerde {
                                        channels: __field0,
                                        kernel_size: __field1,
                                        stride: __field2,
                                        dilation: __field3,
                                        groups: __field4,
                                        padding: __field5,
                                        bias: __field6,
                                        initializer: __field7,
                                    })
                                }
                            }
                            #[doc(hidden)]
                            const FIELDS: &'static [&'static str] = &[
                                "channels",
                                "kernel_size",
                                "stride",
                                "dilation",
                                "groups",
                                "padding",
                                "bias",
                                "initializer",
                            ];
                            _serde::Deserializer::deserialize_struct(
                                __deserializer,
                                "Conv2dConfigSerde",
                                FIELDS,
                                __Visitor {
                                    marker: _serde::__private226::PhantomData::<
                                        Conv2dConfigSerde,
                                    >,
                                    lifetime: _serde::__private226::PhantomData,
                                },
                            )
                        }
                    }
                };
                let serde_state = Conv2dConfigSerde::deserialize(deserializer)?;
                Ok(Conv2dConfig {
                    channels: serde_state.channels,
                    kernel_size: serde_state.kernel_size,
                    stride: serde_state.stride,
                    dilation: serde_state.dilation,
                    groups: serde_state.groups,
                    padding: serde_state.padding,
                    bias: serde_state.bias,
                    initializer: serde_state.initializer,
                })
            }
        }
        impl Clone for Conv2dConfig {
            fn clone(&self) -> Self {
                Self {
                    channels: self.channels.clone(),
                    kernel_size: self.kernel_size.clone(),
                    stride: self.stride.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                    bias: self.bias.clone(),
                    initializer: self.initializer.clone(),
                }
            }
        }
        impl core::fmt::Display for Conv2dConfig {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(&burn::config::config_to_json(self))
            }
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for Conv2dConfig {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                let names: &'static _ = &[
                    "channels",
                    "kernel_size",
                    "stride",
                    "dilation",
                    "groups",
                    "padding",
                    "bias",
                    "initializer",
                ];
                let values: &[&dyn ::core::fmt::Debug] = &[
                    &self.channels,
                    &self.kernel_size,
                    &self.stride,
                    &self.dilation,
                    &self.groups,
                    &self.padding,
                    &self.bias,
                    &&self.initializer,
                ];
                ::core::fmt::Formatter::debug_struct_fields_finish(
                    f,
                    "Conv2dConfig",
                    names,
                    values,
                )
            }
        }
        /// Applies a 2D convolution over input tensors.
        ///
        /// Should be created with [Conv2dConfig].
        #[module(custom_display)]
        pub struct Conv2d<B: Backend> {
            /// Tensor of shape `[channels_out, channels_in / groups, kernel_size_1, kernel_size_2]`
            pub weight: Param<Tensor<B, 4>>,
            /// Tensor of shape `[channels_out]`
            pub bias: Option<Param<Tensor<B, 1>>>,
            /// Stride of the convolution.
            pub stride: [usize; 2],
            /// Size of the kernel.
            pub kernel_size: [usize; 2],
            /// Spacing between kernel elements.
            pub dilation: [usize; 2],
            /// Controls the connections between input and output channels.
            pub groups: usize,
            /// The padding configuration.
            pub padding: Ignored<PaddingConfig2d>,
        }
        impl<B: Backend> burn::module::Module<B> for Conv2d<B> {
            type Record = Conv2dRecord<B>;
            fn load_record(self, record: Self::Record) -> Self {
                Self {
                    weight: burn::module::Module::<
                        B,
                    >::load_record(self.weight, record.weight),
                    bias: burn::module::Module::<B>::load_record(self.bias, record.bias),
                    stride: burn::module::Module::<
                        B,
                    >::load_record(self.stride, record.stride),
                    kernel_size: burn::module::Module::<
                        B,
                    >::load_record(self.kernel_size, record.kernel_size),
                    dilation: burn::module::Module::<
                        B,
                    >::load_record(self.dilation, record.dilation),
                    groups: burn::module::Module::<
                        B,
                    >::load_record(self.groups, record.groups),
                    padding: burn::module::Module::<
                        B,
                    >::load_record(self.padding, record.padding),
                }
            }
            fn into_record(self) -> Self::Record {
                Self::Record {
                    weight: burn::module::Module::<B>::into_record(self.weight),
                    bias: burn::module::Module::<B>::into_record(self.bias),
                    stride: burn::module::Module::<B>::into_record(self.stride),
                    kernel_size: burn::module::Module::<
                        B,
                    >::into_record(self.kernel_size),
                    dilation: burn::module::Module::<B>::into_record(self.dilation),
                    groups: burn::module::Module::<B>::into_record(self.groups),
                    padding: burn::module::Module::<B>::into_record(self.padding),
                }
            }
            fn num_params(&self) -> usize {
                let mut num_params = 0;
                num_params += burn::module::Module::<B>::num_params(&self.weight);
                num_params += burn::module::Module::<B>::num_params(&self.bias);
                num_params += burn::module::Module::<B>::num_params(&self.stride);
                num_params += burn::module::Module::<B>::num_params(&self.kernel_size);
                num_params += burn::module::Module::<B>::num_params(&self.dilation);
                num_params += burn::module::Module::<B>::num_params(&self.groups);
                num_params += burn::module::Module::<B>::num_params(&self.padding);
                num_params
            }
            fn visit<Visitor: burn::module::ModuleVisitor<B>>(
                &self,
                visitor: &mut Visitor,
            ) {
                visitor.enter_module("weight", "Conv2d");
                burn::module::Module::visit(&self.weight, visitor);
                visitor.exit_module("weight", "Conv2d");
                visitor.enter_module("bias", "Conv2d");
                burn::module::Module::visit(&self.bias, visitor);
                visitor.exit_module("bias", "Conv2d");
                visitor.enter_module("stride", "Conv2d");
                burn::module::Module::visit(&self.stride, visitor);
                visitor.exit_module("stride", "Conv2d");
                visitor.enter_module("kernel_size", "Conv2d");
                burn::module::Module::visit(&self.kernel_size, visitor);
                visitor.exit_module("kernel_size", "Conv2d");
                visitor.enter_module("dilation", "Conv2d");
                burn::module::Module::visit(&self.dilation, visitor);
                visitor.exit_module("dilation", "Conv2d");
                visitor.enter_module("groups", "Conv2d");
                burn::module::Module::visit(&self.groups, visitor);
                visitor.exit_module("groups", "Conv2d");
                visitor.enter_module("padding", "Conv2d");
                burn::module::Module::visit(&self.padding, visitor);
                visitor.exit_module("padding", "Conv2d");
            }
            fn map<Mapper: burn::module::ModuleMapper<B>>(
                self,
                mapper: &mut Mapper,
            ) -> Self {
                mapper.enter_module("weight", "Conv2d");
                let weight = burn::module::Module::<B>::map(self.weight, mapper);
                mapper.exit_module("weight", "Conv2d");
                mapper.enter_module("bias", "Conv2d");
                let bias = burn::module::Module::<B>::map(self.bias, mapper);
                mapper.exit_module("bias", "Conv2d");
                mapper.enter_module("stride", "Conv2d");
                let stride = burn::module::Module::<B>::map(self.stride, mapper);
                mapper.exit_module("stride", "Conv2d");
                mapper.enter_module("kernel_size", "Conv2d");
                let kernel_size = burn::module::Module::<
                    B,
                >::map(self.kernel_size, mapper);
                mapper.exit_module("kernel_size", "Conv2d");
                mapper.enter_module("dilation", "Conv2d");
                let dilation = burn::module::Module::<B>::map(self.dilation, mapper);
                mapper.exit_module("dilation", "Conv2d");
                mapper.enter_module("groups", "Conv2d");
                let groups = burn::module::Module::<B>::map(self.groups, mapper);
                mapper.exit_module("groups", "Conv2d");
                mapper.enter_module("padding", "Conv2d");
                let padding = burn::module::Module::<B>::map(self.padding, mapper);
                mapper.exit_module("padding", "Conv2d");
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                }
            }
            fn collect_devices(
                &self,
                devices: burn::module::Devices<B>,
            ) -> burn::module::Devices<B> {
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.weight, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.bias, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.stride, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.kernel_size, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.dilation, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.groups, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.padding, devices);
                devices
            }
            fn to_device(self, device: &B::Device) -> Self {
                let weight = burn::module::Module::<B>::to_device(self.weight, device);
                let bias = burn::module::Module::<B>::to_device(self.bias, device);
                let stride = burn::module::Module::<B>::to_device(self.stride, device);
                let kernel_size = burn::module::Module::<
                    B,
                >::to_device(self.kernel_size, device);
                let dilation = burn::module::Module::<
                    B,
                >::to_device(self.dilation, device);
                let groups = burn::module::Module::<B>::to_device(self.groups, device);
                let padding = burn::module::Module::<B>::to_device(self.padding, device);
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                }
            }
            fn fork(self, device: &B::Device) -> Self {
                let weight = burn::module::Module::<B>::fork(self.weight, device);
                let bias = burn::module::Module::<B>::fork(self.bias, device);
                let stride = burn::module::Module::<B>::fork(self.stride, device);
                let kernel_size = burn::module::Module::<
                    B,
                >::fork(self.kernel_size, device);
                let dilation = burn::module::Module::<B>::fork(self.dilation, device);
                let groups = burn::module::Module::<B>::fork(self.groups, device);
                let padding = burn::module::Module::<B>::fork(self.padding, device);
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                }
            }
        }
        impl<B: Backend> burn::module::AutodiffModule<B> for Conv2d<B>
        where
            B: burn::tensor::backend::AutodiffBackend,
            <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
        {
            type InnerModule = Conv2d<B::InnerBackend>;
            fn valid(&self) -> Self::InnerModule {
                let weight = burn::module::AutodiffModule::<B>::valid(&self.weight);
                let bias = burn::module::AutodiffModule::<B>::valid(&self.bias);
                let stride = burn::module::AutodiffModule::<B>::valid(&self.stride);
                let kernel_size = burn::module::AutodiffModule::<
                    B,
                >::valid(&self.kernel_size);
                let dilation = burn::module::AutodiffModule::<B>::valid(&self.dilation);
                let groups = burn::module::AutodiffModule::<B>::valid(&self.groups);
                let padding = burn::module::AutodiffModule::<B>::valid(&self.padding);
                Self::InnerModule {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                }
            }
        }
        impl<B: Backend> core::fmt::Display for Conv2d<B> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let formatted = burn::module::ModuleDisplay::format(
                    self,
                    Default::default(),
                );
                f.write_fmt(format_args!("{0}", formatted))
            }
        }
        impl<B: Backend> burn::module::ModuleDisplayDefault for Conv2d<B> {
            fn content(
                &self,
                mut content: burn::module::Content,
            ) -> Option<burn::module::Content> {
                content
                    .set_top_level_type(&"Conv2d")
                    .add("weight", &self.weight)
                    .add("bias", &self.bias)
                    .add("stride", &self.stride)
                    .add("kernel_size", &self.kernel_size)
                    .add("dilation", &self.dilation)
                    .add("groups", &self.groups)
                    .add("padding", &self.padding)
                    .optional()
            }
            fn num_params(&self) -> usize {
                burn::module::Module::num_params(self)
            }
        }
        impl<B: Backend> Clone for Conv2d<B> {
            fn clone(&self) -> Self {
                let weight = self.weight.clone();
                let bias = self.bias.clone();
                let stride = self.stride.clone();
                let kernel_size = self.kernel_size.clone();
                let dilation = self.dilation.clone();
                let groups = self.groups.clone();
                let padding = self.padding.clone();
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                }
            }
        }
        /// The record type for the module.
        pub struct Conv2dRecord<B: Backend> {
            /// The module record associative type.
            pub weight: <Param<Tensor<B, 4>> as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub bias: <Option<Param<Tensor<B, 1>>> as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub stride: <[usize; 2] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub kernel_size: <[usize; 2] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub dilation: <[usize; 2] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub groups: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub padding: <Ignored<PaddingConfig2d> as burn::module::Module<B>>::Record,
        }
        /// The record item type for the module.
        #[serde(crate = "burn::serde")]
        #[serde(
            bound = "< < Param < Tensor < B, 4 > > as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize +\nburn :: serde :: de :: DeserializeOwned, < < Option < Param < Tensor < B, 1 >\n> > as burn :: module :: Module < B > > :: Record as burn :: record :: Record\n< B >> :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de ::\nDeserializeOwned, < < [usize; 2] as burn :: module :: Module < B > > :: Record\nas burn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize\n+ burn :: serde :: de :: DeserializeOwned, < < [usize; 2] as burn :: module ::\nModule < B > > :: Record as burn :: record :: Record < B >> :: Item < S > :\nburn :: serde :: Serialize + burn :: serde :: de :: DeserializeOwned, < <\n[usize; 2] as burn :: module :: Module < B > > :: Record as burn :: record ::\nRecord < B >> :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de\n:: DeserializeOwned, < < usize as burn :: module :: Module < B > > :: Record\nas burn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize\n+ burn :: serde :: de :: DeserializeOwned, < < Ignored < PaddingConfig2d > as\nburn :: module :: Module < B > > :: Record as burn :: record :: Record < B >>\n:: Item < S > : burn :: serde :: Serialize + burn :: serde :: de ::\nDeserializeOwned,"
        )]
        pub struct Conv2dRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
            /// Field to be serialized.
            pub weight: <<Param<
                Tensor<B, 4>,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub bias: <<Option<
                Param<Tensor<B, 1>>,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub stride: <<[usize; 2] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub kernel_size: <<[usize; 2] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub dilation: <<[usize; 2] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub groups: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub padding: <<Ignored<
                PaddingConfig2d,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
        }
        #[doc(hidden)]
        #[allow(
            non_upper_case_globals,
            unused_attributes,
            unused_qualifications,
            clippy::absolute_paths,
        )]
        const _: () = {
            use burn::serde as _serde;
            #[automatically_derived]
            impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
            for Conv2dRecordItem<B, S>
            where
                <<Param<
                    Tensor<B, 4>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Option<
                    Param<Tensor<B, 1>>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Ignored<
                    PaddingConfig2d,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            {
                fn serialize<__S>(
                    &self,
                    __serializer: __S,
                ) -> _serde::__private226::Result<__S::Ok, __S::Error>
                where
                    __S: _serde::Serializer,
                {
                    let mut __serde_state = _serde::Serializer::serialize_struct(
                        __serializer,
                        "Conv2dRecordItem",
                        false as usize + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "weight",
                        &self.weight,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "bias",
                        &self.bias,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "stride",
                        &self.stride,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "kernel_size",
                        &self.kernel_size,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "dilation",
                        &self.dilation,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "groups",
                        &self.groups,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "padding",
                        &self.padding,
                    )?;
                    _serde::ser::SerializeStruct::end(__serde_state)
                }
            }
        };
        #[doc(hidden)]
        #[allow(
            non_upper_case_globals,
            unused_attributes,
            unused_qualifications,
            clippy::absolute_paths,
        )]
        const _: () = {
            use burn::serde as _serde;
            #[automatically_derived]
            impl<
                'de,
                B: Backend,
                S: burn::record::PrecisionSettings,
            > _serde::Deserialize<'de> for Conv2dRecordItem<B, S>
            where
                <<Param<
                    Tensor<B, 4>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Option<
                    Param<Tensor<B, 1>>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Ignored<
                    PaddingConfig2d,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            {
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private226::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    #[allow(non_camel_case_types)]
                    #[doc(hidden)]
                    enum __Field {
                        __field0,
                        __field1,
                        __field2,
                        __field3,
                        __field4,
                        __field5,
                        __field6,
                        __ignore,
                    }
                    #[doc(hidden)]
                    struct __FieldVisitor;
                    #[automatically_derived]
                    impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                        type Value = __Field;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private226::Formatter,
                        ) -> _serde::__private226::fmt::Result {
                            _serde::__private226::Formatter::write_str(
                                __formatter,
                                "field identifier",
                            )
                        }
                        fn visit_u64<__E>(
                            self,
                            __value: u64,
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                0u64 => _serde::__private226::Ok(__Field::__field0),
                                1u64 => _serde::__private226::Ok(__Field::__field1),
                                2u64 => _serde::__private226::Ok(__Field::__field2),
                                3u64 => _serde::__private226::Ok(__Field::__field3),
                                4u64 => _serde::__private226::Ok(__Field::__field4),
                                5u64 => _serde::__private226::Ok(__Field::__field5),
                                6u64 => _serde::__private226::Ok(__Field::__field6),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_str<__E>(
                            self,
                            __value: &str,
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                "weight" => _serde::__private226::Ok(__Field::__field0),
                                "bias" => _serde::__private226::Ok(__Field::__field1),
                                "stride" => _serde::__private226::Ok(__Field::__field2),
                                "kernel_size" => _serde::__private226::Ok(__Field::__field3),
                                "dilation" => _serde::__private226::Ok(__Field::__field4),
                                "groups" => _serde::__private226::Ok(__Field::__field5),
                                "padding" => _serde::__private226::Ok(__Field::__field6),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_bytes<__E>(
                            self,
                            __value: &[u8],
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                b"weight" => _serde::__private226::Ok(__Field::__field0),
                                b"bias" => _serde::__private226::Ok(__Field::__field1),
                                b"stride" => _serde::__private226::Ok(__Field::__field2),
                                b"kernel_size" => {
                                    _serde::__private226::Ok(__Field::__field3)
                                }
                                b"dilation" => _serde::__private226::Ok(__Field::__field4),
                                b"groups" => _serde::__private226::Ok(__Field::__field5),
                                b"padding" => _serde::__private226::Ok(__Field::__field6),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                    }
                    #[automatically_derived]
                    impl<'de> _serde::Deserialize<'de> for __Field {
                        #[inline]
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private226::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            _serde::Deserializer::deserialize_identifier(
                                __deserializer,
                                __FieldVisitor,
                            )
                        }
                    }
                    #[doc(hidden)]
                    struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                    where
                        <<Param<
                            Tensor<B, 4>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Option<
                            Param<Tensor<B, 1>>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Ignored<
                            PaddingConfig2d,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    {
                        marker: _serde::__private226::PhantomData<
                            Conv2dRecordItem<B, S>,
                        >,
                        lifetime: _serde::__private226::PhantomData<&'de ()>,
                    }
                    #[automatically_derived]
                    impl<
                        'de,
                        B: Backend,
                        S: burn::record::PrecisionSettings,
                    > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                    where
                        <<Param<
                            Tensor<B, 4>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Option<
                            Param<Tensor<B, 1>>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Ignored<
                            PaddingConfig2d,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    {
                        type Value = Conv2dRecordItem<B, S>;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private226::Formatter,
                        ) -> _serde::__private226::fmt::Result {
                            _serde::__private226::Formatter::write_str(
                                __formatter,
                                "struct Conv2dRecordItem",
                            )
                        }
                        #[inline]
                        fn visit_seq<__A>(
                            self,
                            mut __seq: __A,
                        ) -> _serde::__private226::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::SeqAccess<'de>,
                        {
                            let __field0 = match _serde::de::SeqAccess::next_element::<
                                <<Param<
                                    Tensor<B, 4>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            0usize,
                                            &"struct Conv2dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field1 = match _serde::de::SeqAccess::next_element::<
                                <<Option<
                                    Param<Tensor<B, 1>>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            1usize,
                                            &"struct Conv2dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field2 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            2usize,
                                            &"struct Conv2dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field3 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            3usize,
                                            &"struct Conv2dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field4 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            4usize,
                                            &"struct Conv2dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field5 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            5usize,
                                            &"struct Conv2dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field6 = match _serde::de::SeqAccess::next_element::<
                                <<Ignored<
                                    PaddingConfig2d,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            6usize,
                                            &"struct Conv2dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            _serde::__private226::Ok(Conv2dRecordItem {
                                weight: __field0,
                                bias: __field1,
                                stride: __field2,
                                kernel_size: __field3,
                                dilation: __field4,
                                groups: __field5,
                                padding: __field6,
                            })
                        }
                        #[inline]
                        fn visit_map<__A>(
                            self,
                            mut __map: __A,
                        ) -> _serde::__private226::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::MapAccess<'de>,
                        {
                            let mut __field0: _serde::__private226::Option<
                                <<Param<
                                    Tensor<B, 4>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field1: _serde::__private226::Option<
                                <<Option<
                                    Param<Tensor<B, 1>>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field2: _serde::__private226::Option<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field3: _serde::__private226::Option<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field4: _serde::__private226::Option<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field5: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field6: _serde::__private226::Option<
                                <<Ignored<
                                    PaddingConfig2d,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            while let _serde::__private226::Some(__key) = _serde::de::MapAccess::next_key::<
                                __Field,
                            >(&mut __map)? {
                                match __key {
                                    __Field::__field0 => {
                                        if _serde::__private226::Option::is_some(&__field0) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("weight"),
                                            );
                                        }
                                        __field0 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Param<
                                                    Tensor<B, 4>,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field1 => {
                                        if _serde::__private226::Option::is_some(&__field1) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("bias"),
                                            );
                                        }
                                        __field1 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Option<
                                                    Param<Tensor<B, 1>>,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field2 => {
                                        if _serde::__private226::Option::is_some(&__field2) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                            );
                                        }
                                        __field2 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 2] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field3 => {
                                        if _serde::__private226::Option::is_some(&__field3) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "kernel_size",
                                                ),
                                            );
                                        }
                                        __field3 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 2] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field4 => {
                                        if _serde::__private226::Option::is_some(&__field4) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "dilation",
                                                ),
                                            );
                                        }
                                        __field4 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 2] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field5 => {
                                        if _serde::__private226::Option::is_some(&__field5) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("groups"),
                                            );
                                        }
                                        __field5 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field6 => {
                                        if _serde::__private226::Option::is_some(&__field6) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "padding",
                                                ),
                                            );
                                        }
                                        __field6 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Ignored<
                                                    PaddingConfig2d,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    _ => {
                                        let _ = _serde::de::MapAccess::next_value::<
                                            _serde::de::IgnoredAny,
                                        >(&mut __map)?;
                                    }
                                }
                            }
                            let __field0 = match __field0 {
                                _serde::__private226::Some(__field0) => __field0,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("weight")?
                                }
                            };
                            let __field1 = match __field1 {
                                _serde::__private226::Some(__field1) => __field1,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("bias")?
                                }
                            };
                            let __field2 = match __field2 {
                                _serde::__private226::Some(__field2) => __field2,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("stride")?
                                }
                            };
                            let __field3 = match __field3 {
                                _serde::__private226::Some(__field3) => __field3,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("kernel_size")?
                                }
                            };
                            let __field4 = match __field4 {
                                _serde::__private226::Some(__field4) => __field4,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("dilation")?
                                }
                            };
                            let __field5 = match __field5 {
                                _serde::__private226::Some(__field5) => __field5,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("groups")?
                                }
                            };
                            let __field6 = match __field6 {
                                _serde::__private226::Some(__field6) => __field6,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("padding")?
                                }
                            };
                            _serde::__private226::Ok(Conv2dRecordItem {
                                weight: __field0,
                                bias: __field1,
                                stride: __field2,
                                kernel_size: __field3,
                                dilation: __field4,
                                groups: __field5,
                                padding: __field6,
                            })
                        }
                    }
                    #[doc(hidden)]
                    const FIELDS: &'static [&'static str] = &[
                        "weight",
                        "bias",
                        "stride",
                        "kernel_size",
                        "dilation",
                        "groups",
                        "padding",
                    ];
                    _serde::Deserializer::deserialize_struct(
                        __deserializer,
                        "Conv2dRecordItem",
                        FIELDS,
                        __Visitor {
                            marker: _serde::__private226::PhantomData::<
                                Conv2dRecordItem<B, S>,
                            >,
                            lifetime: _serde::__private226::PhantomData,
                        },
                    )
                }
            }
        };
        impl<B: Backend, S: burn::record::PrecisionSettings> Clone
        for Conv2dRecordItem<B, S> {
            fn clone(&self) -> Self {
                Self {
                    weight: self.weight.clone(),
                    bias: self.bias.clone(),
                    stride: self.stride.clone(),
                    kernel_size: self.kernel_size.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                }
            }
        }
        impl<B: Backend> burn::record::Record<B> for Conv2dRecord<B> {
            type Item<S: burn::record::PrecisionSettings> = Conv2dRecordItem<B, S>;
            fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
                Conv2dRecordItem {
                    weight: burn::record::Record::<B>::into_item::<S>(self.weight),
                    bias: burn::record::Record::<B>::into_item::<S>(self.bias),
                    stride: burn::record::Record::<B>::into_item::<S>(self.stride),
                    kernel_size: burn::record::Record::<
                        B,
                    >::into_item::<S>(self.kernel_size),
                    dilation: burn::record::Record::<B>::into_item::<S>(self.dilation),
                    groups: burn::record::Record::<B>::into_item::<S>(self.groups),
                    padding: burn::record::Record::<B>::into_item::<S>(self.padding),
                }
            }
            fn from_item<S: burn::record::PrecisionSettings>(
                item: Self::Item<S>,
                device: &B::Device,
            ) -> Self {
                Self {
                    weight: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.weight, device),
                    bias: burn::record::Record::<B>::from_item::<S>(item.bias, device),
                    stride: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.stride, device),
                    kernel_size: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.kernel_size, device),
                    dilation: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.dilation, device),
                    groups: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.groups, device),
                    padding: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.padding, device),
                }
            }
        }
        #[automatically_derived]
        impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Conv2d<B> {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                let names: &'static _ = &[
                    "weight",
                    "bias",
                    "stride",
                    "kernel_size",
                    "dilation",
                    "groups",
                    "padding",
                ];
                let values: &[&dyn ::core::fmt::Debug] = &[
                    &self.weight,
                    &self.bias,
                    &self.stride,
                    &self.kernel_size,
                    &self.dilation,
                    &self.groups,
                    &&self.padding,
                ];
                ::core::fmt::Formatter::debug_struct_fields_finish(
                    f,
                    "Conv2d",
                    names,
                    values,
                )
            }
        }
        impl Conv2dConfig {
            /// Initialize a new [conv2d](Conv2d) module.
            pub fn init<B: Backend>(&self, device: &B::Device) -> Conv2d<B> {
                checks::checks_channels_div_groups(
                    self.channels[0],
                    self.channels[1],
                    self.groups,
                );
                if self.padding == PaddingConfig2d::Same {
                    checks::check_same_padding_support(&self.kernel_size);
                }
                let shape = [
                    self.channels[1],
                    self.channels[0] / self.groups,
                    self.kernel_size[0],
                    self.kernel_size[1],
                ];
                let k = self.kernel_size.iter().product::<usize>();
                let fan_in = self.channels[0] / self.groups * k;
                let fan_out = self.channels[1] / self.groups * k;
                let weight = self
                    .initializer
                    .init_with(shape, Some(fan_in), Some(fan_out), device);
                let mut bias = None;
                if self.bias {
                    bias = Some(
                        self
                            .initializer
                            .init_with(
                                [self.channels[1]],
                                Some(fan_in),
                                Some(fan_out),
                                device,
                            ),
                    );
                }
                Conv2d {
                    weight,
                    bias,
                    stride: self.stride,
                    kernel_size: self.kernel_size,
                    dilation: self.dilation,
                    padding: Ignored(self.padding.clone()),
                    groups: self.groups,
                }
            }
        }
        impl<B: Backend> ModuleDisplay for Conv2d<B> {
            fn custom_settings(&self) -> Option<DisplaySettings> {
                DisplaySettings::new().with_new_line_after_attribute(false).optional()
            }
            fn custom_content(&self, content: Content) -> Option<Content> {
                let padding_formatted = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0}", &self.padding))
                });
                let stride = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", self.stride))
                });
                let kernel_size = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", self.kernel_size))
                });
                let dilation = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", self.dilation))
                });
                let [channels_out, group_channels_in, _, _] = self.weight.dims();
                let channels_in = group_channels_in * self.groups;
                let ch_out = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", channels_out))
                });
                let ch_in = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", channels_in))
                });
                content
                    .add("ch_in", &ch_in)
                    .add("ch_out", &ch_out)
                    .add("stride", &stride)
                    .add("kernel_size", &kernel_size)
                    .add("dilation", &dilation)
                    .add("groups", &self.groups)
                    .add("padding", &padding_formatted)
                    .optional()
            }
        }
        impl<B: Backend> Conv2d<B> {
            /// Applies the forward pass on the input tensor.
            ///
            /// See [conv2d](burn::tensor::module::conv2d) for more information.
            ///
            /// # Shapes
            /// - `input`: `[batch_size, channels_in, height_in, width_in]`
            /// - `output`: `[batch_size, channels_out, height_out, width_out]`
            ///
            /// # Example
            /// ```rust,ignore
            /// use burn::nn::conv::Conv2dConfig;
            /// use burn::tensor::Tensor;
            ///
            /// // Assuming backend type alias `B`
            /// let device = Default::default();
            /// let conv = Conv2dConfig::new([3, 8], [3, 3]).init::<B>(&device);
            ///
            /// let x = Tensor::<B, 4>::zeros([1, 3, 28, 28], &device);
            /// let y = conv.forward(x);
            ///
            /// println!("{:?}", y.dims()); // [1, 8, 26, 26]
            /// ```
            pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                let [_batch_size, _channels_in, height_in, width_in] = input.dims();
                let padding = self
                    .padding
                    .calculate_padding_2d(
                        height_in,
                        width_in,
                        &self.kernel_size,
                        &self.stride,
                    );
                conv2d(
                    input,
                    self.weight.val(),
                    self.bias.as_ref().map(|bias| bias.val()),
                    ConvOptions::new(self.stride, padding, self.dilation, self.groups),
                )
            }
        }
    }
    mod conv3d {
        use alloc::format;
        use burn_core as burn;
        use crate::PaddingConfig3d;
        use burn::config::Config;
        use burn::module::Initializer;
        use burn::module::{
            Content, DisplaySettings, Ignored, Module, ModuleDisplay, Param,
        };
        use burn::tensor::Tensor;
        use burn::tensor::backend::Backend;
        use burn::tensor::module::conv3d;
        use burn::tensor::ops::ConvOptions;
        use crate::conv::checks;
        /// Configuration to create a [3D convolution](Conv3d) layer, using the [init function](Conv3dConfig::init).
        pub struct Conv3dConfig {
            /// The number of channels.
            pub channels: [usize; 2],
            /// The size of the kernel.
            pub kernel_size: [usize; 3],
            /// The stride of the convolution.
            #[config(default = "[1, 1, 1]")]
            pub stride: [usize; 3],
            /// Spacing between kernel elements.
            #[config(default = "[1, 1, 1]")]
            pub dilation: [usize; 3],
            /// Controls the connections between input and output channels.
            #[config(default = "1")]
            pub groups: usize,
            /// The padding configuration.
            #[config(default = "PaddingConfig3d::Valid")]
            pub padding: PaddingConfig3d,
            /// If bias should be added to the output.
            #[config(default = true)]
            pub bias: bool,
            /// The type of function used to initialize neural network parameters
            #[config(
                default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
            )]
            pub initializer: Initializer,
        }
        impl burn::config::Config for Conv3dConfig {}
        impl Conv3dConfig {
            /// Create a new instance of the config.
            #[allow(clippy::too_many_arguments)]
            pub fn new(channels: [usize; 2], kernel_size: [usize; 3]) -> Self {
                Self {
                    channels: channels,
                    kernel_size: kernel_size,
                    stride: [1, 1, 1],
                    dilation: [1, 1, 1],
                    groups: 1,
                    padding: PaddingConfig3d::Valid,
                    bias: true,
                    initializer: Initializer::KaimingUniform {
                        gain: 1.0 / num_traits::Float::sqrt(3.0),
                        fan_out_only: false,
                    },
                }
            }
        }
        impl Conv3dConfig {
            /// The stride of the convolution.
            pub fn with_stride(mut self, stride: [usize; 3]) -> Self {
                self.stride = stride;
                self
            }
            /// Spacing between kernel elements.
            pub fn with_dilation(mut self, dilation: [usize; 3]) -> Self {
                self.dilation = dilation;
                self
            }
            /// Controls the connections between input and output channels.
            pub fn with_groups(mut self, groups: usize) -> Self {
                self.groups = groups;
                self
            }
            /// The padding configuration.
            pub fn with_padding(mut self, padding: PaddingConfig3d) -> Self {
                self.padding = padding;
                self
            }
            /// If bias should be added to the output.
            pub fn with_bias(mut self, bias: bool) -> Self {
                self.bias = bias;
                self
            }
            /// The type of function used to initialize neural network parameters
            pub fn with_initializer(mut self, initializer: Initializer) -> Self {
                self.initializer = initializer;
                self
            }
        }
        impl burn::serde::Serialize for Conv3dConfig {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: burn::serde::Serializer,
            {
                #[serde(crate = "burn::serde")]
                struct Conv3dConfigSerde {
                    channels: [usize; 2],
                    kernel_size: [usize; 3],
                    stride: [usize; 3],
                    dilation: [usize; 3],
                    groups: usize,
                    padding: PaddingConfig3d,
                    bias: bool,
                    initializer: Initializer,
                }
                #[doc(hidden)]
                #[allow(
                    non_upper_case_globals,
                    unused_attributes,
                    unused_qualifications,
                    clippy::absolute_paths,
                )]
                const _: () = {
                    use burn::serde as _serde;
                    #[automatically_derived]
                    impl _serde::Serialize for Conv3dConfigSerde {
                        fn serialize<__S>(
                            &self,
                            __serializer: __S,
                        ) -> _serde::__private226::Result<__S::Ok, __S::Error>
                        where
                            __S: _serde::Serializer,
                        {
                            let mut __serde_state = _serde::Serializer::serialize_struct(
                                __serializer,
                                "Conv3dConfigSerde",
                                false as usize + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "channels",
                                &self.channels,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "kernel_size",
                                &self.kernel_size,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "stride",
                                &self.stride,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "dilation",
                                &self.dilation,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "groups",
                                &self.groups,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "padding",
                                &self.padding,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "bias",
                                &self.bias,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "initializer",
                                &self.initializer,
                            )?;
                            _serde::ser::SerializeStruct::end(__serde_state)
                        }
                    }
                };
                let serde_state = Conv3dConfigSerde {
                    channels: self.channels.clone(),
                    kernel_size: self.kernel_size.clone(),
                    stride: self.stride.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                    bias: self.bias.clone(),
                    initializer: self.initializer.clone(),
                };
                serde_state.serialize(serializer)
            }
        }
        impl<'de> burn::serde::Deserialize<'de> for Conv3dConfig {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: burn::serde::Deserializer<'de>,
            {
                #[serde(crate = "burn::serde")]
                struct Conv3dConfigSerde {
                    channels: [usize; 2],
                    kernel_size: [usize; 3],
                    stride: [usize; 3],
                    dilation: [usize; 3],
                    groups: usize,
                    padding: PaddingConfig3d,
                    bias: bool,
                    initializer: Initializer,
                }
                #[doc(hidden)]
                #[allow(
                    non_upper_case_globals,
                    unused_attributes,
                    unused_qualifications,
                    clippy::absolute_paths,
                )]
                const _: () = {
                    use burn::serde as _serde;
                    #[automatically_derived]
                    impl<'de> _serde::Deserialize<'de> for Conv3dConfigSerde {
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private226::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            #[allow(non_camel_case_types)]
                            #[doc(hidden)]
                            enum __Field {
                                __field0,
                                __field1,
                                __field2,
                                __field3,
                                __field4,
                                __field5,
                                __field6,
                                __field7,
                                __ignore,
                            }
                            #[doc(hidden)]
                            struct __FieldVisitor;
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                type Value = __Field;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private226::Formatter,
                                ) -> _serde::__private226::fmt::Result {
                                    _serde::__private226::Formatter::write_str(
                                        __formatter,
                                        "field identifier",
                                    )
                                }
                                fn visit_u64<__E>(
                                    self,
                                    __value: u64,
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        0u64 => _serde::__private226::Ok(__Field::__field0),
                                        1u64 => _serde::__private226::Ok(__Field::__field1),
                                        2u64 => _serde::__private226::Ok(__Field::__field2),
                                        3u64 => _serde::__private226::Ok(__Field::__field3),
                                        4u64 => _serde::__private226::Ok(__Field::__field4),
                                        5u64 => _serde::__private226::Ok(__Field::__field5),
                                        6u64 => _serde::__private226::Ok(__Field::__field6),
                                        7u64 => _serde::__private226::Ok(__Field::__field7),
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_str<__E>(
                                    self,
                                    __value: &str,
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        "channels" => _serde::__private226::Ok(__Field::__field0),
                                        "kernel_size" => _serde::__private226::Ok(__Field::__field1),
                                        "stride" => _serde::__private226::Ok(__Field::__field2),
                                        "dilation" => _serde::__private226::Ok(__Field::__field3),
                                        "groups" => _serde::__private226::Ok(__Field::__field4),
                                        "padding" => _serde::__private226::Ok(__Field::__field5),
                                        "bias" => _serde::__private226::Ok(__Field::__field6),
                                        "initializer" => _serde::__private226::Ok(__Field::__field7),
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_bytes<__E>(
                                    self,
                                    __value: &[u8],
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        b"channels" => _serde::__private226::Ok(__Field::__field0),
                                        b"kernel_size" => {
                                            _serde::__private226::Ok(__Field::__field1)
                                        }
                                        b"stride" => _serde::__private226::Ok(__Field::__field2),
                                        b"dilation" => _serde::__private226::Ok(__Field::__field3),
                                        b"groups" => _serde::__private226::Ok(__Field::__field4),
                                        b"padding" => _serde::__private226::Ok(__Field::__field5),
                                        b"bias" => _serde::__private226::Ok(__Field::__field6),
                                        b"initializer" => {
                                            _serde::__private226::Ok(__Field::__field7)
                                        }
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                            }
                            #[automatically_derived]
                            impl<'de> _serde::Deserialize<'de> for __Field {
                                #[inline]
                                fn deserialize<__D>(
                                    __deserializer: __D,
                                ) -> _serde::__private226::Result<Self, __D::Error>
                                where
                                    __D: _serde::Deserializer<'de>,
                                {
                                    _serde::Deserializer::deserialize_identifier(
                                        __deserializer,
                                        __FieldVisitor,
                                    )
                                }
                            }
                            #[doc(hidden)]
                            struct __Visitor<'de> {
                                marker: _serde::__private226::PhantomData<
                                    Conv3dConfigSerde,
                                >,
                                lifetime: _serde::__private226::PhantomData<&'de ()>,
                            }
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                type Value = Conv3dConfigSerde;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private226::Formatter,
                                ) -> _serde::__private226::fmt::Result {
                                    _serde::__private226::Formatter::write_str(
                                        __formatter,
                                        "struct Conv3dConfigSerde",
                                    )
                                }
                                #[inline]
                                fn visit_seq<__A>(
                                    self,
                                    mut __seq: __A,
                                ) -> _serde::__private226::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::SeqAccess<'de>,
                                {
                                    let __field0 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    0usize,
                                                    &"struct Conv3dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field1 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 3],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    1usize,
                                                    &"struct Conv3dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field2 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 3],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    2usize,
                                                    &"struct Conv3dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field3 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 3],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    3usize,
                                                    &"struct Conv3dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field4 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    4usize,
                                                    &"struct Conv3dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field5 = match _serde::de::SeqAccess::next_element::<
                                        PaddingConfig3d,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    5usize,
                                                    &"struct Conv3dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field6 = match _serde::de::SeqAccess::next_element::<
                                        bool,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    6usize,
                                                    &"struct Conv3dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field7 = match _serde::de::SeqAccess::next_element::<
                                        Initializer,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    7usize,
                                                    &"struct Conv3dConfigSerde with 8 elements",
                                                ),
                                            );
                                        }
                                    };
                                    _serde::__private226::Ok(Conv3dConfigSerde {
                                        channels: __field0,
                                        kernel_size: __field1,
                                        stride: __field2,
                                        dilation: __field3,
                                        groups: __field4,
                                        padding: __field5,
                                        bias: __field6,
                                        initializer: __field7,
                                    })
                                }
                                #[inline]
                                fn visit_map<__A>(
                                    self,
                                    mut __map: __A,
                                ) -> _serde::__private226::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::MapAccess<'de>,
                                {
                                    let mut __field0: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field1: _serde::__private226::Option<
                                        [usize; 3],
                                    > = _serde::__private226::None;
                                    let mut __field2: _serde::__private226::Option<
                                        [usize; 3],
                                    > = _serde::__private226::None;
                                    let mut __field3: _serde::__private226::Option<
                                        [usize; 3],
                                    > = _serde::__private226::None;
                                    let mut __field4: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field5: _serde::__private226::Option<
                                        PaddingConfig3d,
                                    > = _serde::__private226::None;
                                    let mut __field6: _serde::__private226::Option<bool> = _serde::__private226::None;
                                    let mut __field7: _serde::__private226::Option<
                                        Initializer,
                                    > = _serde::__private226::None;
                                    while let _serde::__private226::Some(__key) = _serde::de::MapAccess::next_key::<
                                        __Field,
                                    >(&mut __map)? {
                                        match __key {
                                            __Field::__field0 => {
                                                if _serde::__private226::Option::is_some(&__field0) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "channels",
                                                        ),
                                                    );
                                                }
                                                __field0 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field1 => {
                                                if _serde::__private226::Option::is_some(&__field1) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "kernel_size",
                                                        ),
                                                    );
                                                }
                                                __field1 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 3]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field2 => {
                                                if _serde::__private226::Option::is_some(&__field2) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                    );
                                                }
                                                __field2 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 3]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field3 => {
                                                if _serde::__private226::Option::is_some(&__field3) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "dilation",
                                                        ),
                                                    );
                                                }
                                                __field3 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 3]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field4 => {
                                                if _serde::__private226::Option::is_some(&__field4) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("groups"),
                                                    );
                                                }
                                                __field4 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field5 => {
                                                if _serde::__private226::Option::is_some(&__field5) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "padding",
                                                        ),
                                                    );
                                                }
                                                __field5 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<
                                                        PaddingConfig3d,
                                                    >(&mut __map)?,
                                                );
                                            }
                                            __Field::__field6 => {
                                                if _serde::__private226::Option::is_some(&__field6) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("bias"),
                                                    );
                                                }
                                                __field6 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<bool>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field7 => {
                                                if _serde::__private226::Option::is_some(&__field7) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "initializer",
                                                        ),
                                                    );
                                                }
                                                __field7 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<
                                                        Initializer,
                                                    >(&mut __map)?,
                                                );
                                            }
                                            _ => {
                                                let _ = _serde::de::MapAccess::next_value::<
                                                    _serde::de::IgnoredAny,
                                                >(&mut __map)?;
                                            }
                                        }
                                    }
                                    let __field0 = match __field0 {
                                        _serde::__private226::Some(__field0) => __field0,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("channels")?
                                        }
                                    };
                                    let __field1 = match __field1 {
                                        _serde::__private226::Some(__field1) => __field1,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("kernel_size")?
                                        }
                                    };
                                    let __field2 = match __field2 {
                                        _serde::__private226::Some(__field2) => __field2,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("stride")?
                                        }
                                    };
                                    let __field3 = match __field3 {
                                        _serde::__private226::Some(__field3) => __field3,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("dilation")?
                                        }
                                    };
                                    let __field4 = match __field4 {
                                        _serde::__private226::Some(__field4) => __field4,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("groups")?
                                        }
                                    };
                                    let __field5 = match __field5 {
                                        _serde::__private226::Some(__field5) => __field5,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("padding")?
                                        }
                                    };
                                    let __field6 = match __field6 {
                                        _serde::__private226::Some(__field6) => __field6,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("bias")?
                                        }
                                    };
                                    let __field7 = match __field7 {
                                        _serde::__private226::Some(__field7) => __field7,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("initializer")?
                                        }
                                    };
                                    _serde::__private226::Ok(Conv3dConfigSerde {
                                        channels: __field0,
                                        kernel_size: __field1,
                                        stride: __field2,
                                        dilation: __field3,
                                        groups: __field4,
                                        padding: __field5,
                                        bias: __field6,
                                        initializer: __field7,
                                    })
                                }
                            }
                            #[doc(hidden)]
                            const FIELDS: &'static [&'static str] = &[
                                "channels",
                                "kernel_size",
                                "stride",
                                "dilation",
                                "groups",
                                "padding",
                                "bias",
                                "initializer",
                            ];
                            _serde::Deserializer::deserialize_struct(
                                __deserializer,
                                "Conv3dConfigSerde",
                                FIELDS,
                                __Visitor {
                                    marker: _serde::__private226::PhantomData::<
                                        Conv3dConfigSerde,
                                    >,
                                    lifetime: _serde::__private226::PhantomData,
                                },
                            )
                        }
                    }
                };
                let serde_state = Conv3dConfigSerde::deserialize(deserializer)?;
                Ok(Conv3dConfig {
                    channels: serde_state.channels,
                    kernel_size: serde_state.kernel_size,
                    stride: serde_state.stride,
                    dilation: serde_state.dilation,
                    groups: serde_state.groups,
                    padding: serde_state.padding,
                    bias: serde_state.bias,
                    initializer: serde_state.initializer,
                })
            }
        }
        impl Clone for Conv3dConfig {
            fn clone(&self) -> Self {
                Self {
                    channels: self.channels.clone(),
                    kernel_size: self.kernel_size.clone(),
                    stride: self.stride.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                    bias: self.bias.clone(),
                    initializer: self.initializer.clone(),
                }
            }
        }
        impl core::fmt::Display for Conv3dConfig {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(&burn::config::config_to_json(self))
            }
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for Conv3dConfig {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                let names: &'static _ = &[
                    "channels",
                    "kernel_size",
                    "stride",
                    "dilation",
                    "groups",
                    "padding",
                    "bias",
                    "initializer",
                ];
                let values: &[&dyn ::core::fmt::Debug] = &[
                    &self.channels,
                    &self.kernel_size,
                    &self.stride,
                    &self.dilation,
                    &self.groups,
                    &self.padding,
                    &self.bias,
                    &&self.initializer,
                ];
                ::core::fmt::Formatter::debug_struct_fields_finish(
                    f,
                    "Conv3dConfig",
                    names,
                    values,
                )
            }
        }
        /// Applies a 3D convolution over input tensors.
        ///
        /// Should be created with [Conv3dConfig].
        #[module(custom_display)]
        pub struct Conv3d<B: Backend> {
            /// Tensor of shape `[channels_out, channels_in / groups, kernel_size_1, kernel_size_2, kernel_size_3]`
            pub weight: Param<Tensor<B, 5>>,
            /// Tensor of shape `[channels_out]`
            pub bias: Option<Param<Tensor<B, 1>>>,
            /// Stride of the convolution.
            pub stride: [usize; 3],
            /// Size of the kernel.
            pub kernel_size: [usize; 3],
            /// Spacing between kernel elements.
            pub dilation: [usize; 3],
            /// Controls the connections between input and output channels.
            pub groups: usize,
            /// The padding configuration.
            pub padding: Ignored<PaddingConfig3d>,
        }
        impl<B: Backend> burn::module::Module<B> for Conv3d<B> {
            type Record = Conv3dRecord<B>;
            fn load_record(self, record: Self::Record) -> Self {
                Self {
                    weight: burn::module::Module::<
                        B,
                    >::load_record(self.weight, record.weight),
                    bias: burn::module::Module::<B>::load_record(self.bias, record.bias),
                    stride: burn::module::Module::<
                        B,
                    >::load_record(self.stride, record.stride),
                    kernel_size: burn::module::Module::<
                        B,
                    >::load_record(self.kernel_size, record.kernel_size),
                    dilation: burn::module::Module::<
                        B,
                    >::load_record(self.dilation, record.dilation),
                    groups: burn::module::Module::<
                        B,
                    >::load_record(self.groups, record.groups),
                    padding: burn::module::Module::<
                        B,
                    >::load_record(self.padding, record.padding),
                }
            }
            fn into_record(self) -> Self::Record {
                Self::Record {
                    weight: burn::module::Module::<B>::into_record(self.weight),
                    bias: burn::module::Module::<B>::into_record(self.bias),
                    stride: burn::module::Module::<B>::into_record(self.stride),
                    kernel_size: burn::module::Module::<
                        B,
                    >::into_record(self.kernel_size),
                    dilation: burn::module::Module::<B>::into_record(self.dilation),
                    groups: burn::module::Module::<B>::into_record(self.groups),
                    padding: burn::module::Module::<B>::into_record(self.padding),
                }
            }
            fn num_params(&self) -> usize {
                let mut num_params = 0;
                num_params += burn::module::Module::<B>::num_params(&self.weight);
                num_params += burn::module::Module::<B>::num_params(&self.bias);
                num_params += burn::module::Module::<B>::num_params(&self.stride);
                num_params += burn::module::Module::<B>::num_params(&self.kernel_size);
                num_params += burn::module::Module::<B>::num_params(&self.dilation);
                num_params += burn::module::Module::<B>::num_params(&self.groups);
                num_params += burn::module::Module::<B>::num_params(&self.padding);
                num_params
            }
            fn visit<Visitor: burn::module::ModuleVisitor<B>>(
                &self,
                visitor: &mut Visitor,
            ) {
                visitor.enter_module("weight", "Conv3d");
                burn::module::Module::visit(&self.weight, visitor);
                visitor.exit_module("weight", "Conv3d");
                visitor.enter_module("bias", "Conv3d");
                burn::module::Module::visit(&self.bias, visitor);
                visitor.exit_module("bias", "Conv3d");
                visitor.enter_module("stride", "Conv3d");
                burn::module::Module::visit(&self.stride, visitor);
                visitor.exit_module("stride", "Conv3d");
                visitor.enter_module("kernel_size", "Conv3d");
                burn::module::Module::visit(&self.kernel_size, visitor);
                visitor.exit_module("kernel_size", "Conv3d");
                visitor.enter_module("dilation", "Conv3d");
                burn::module::Module::visit(&self.dilation, visitor);
                visitor.exit_module("dilation", "Conv3d");
                visitor.enter_module("groups", "Conv3d");
                burn::module::Module::visit(&self.groups, visitor);
                visitor.exit_module("groups", "Conv3d");
                visitor.enter_module("padding", "Conv3d");
                burn::module::Module::visit(&self.padding, visitor);
                visitor.exit_module("padding", "Conv3d");
            }
            fn map<Mapper: burn::module::ModuleMapper<B>>(
                self,
                mapper: &mut Mapper,
            ) -> Self {
                mapper.enter_module("weight", "Conv3d");
                let weight = burn::module::Module::<B>::map(self.weight, mapper);
                mapper.exit_module("weight", "Conv3d");
                mapper.enter_module("bias", "Conv3d");
                let bias = burn::module::Module::<B>::map(self.bias, mapper);
                mapper.exit_module("bias", "Conv3d");
                mapper.enter_module("stride", "Conv3d");
                let stride = burn::module::Module::<B>::map(self.stride, mapper);
                mapper.exit_module("stride", "Conv3d");
                mapper.enter_module("kernel_size", "Conv3d");
                let kernel_size = burn::module::Module::<
                    B,
                >::map(self.kernel_size, mapper);
                mapper.exit_module("kernel_size", "Conv3d");
                mapper.enter_module("dilation", "Conv3d");
                let dilation = burn::module::Module::<B>::map(self.dilation, mapper);
                mapper.exit_module("dilation", "Conv3d");
                mapper.enter_module("groups", "Conv3d");
                let groups = burn::module::Module::<B>::map(self.groups, mapper);
                mapper.exit_module("groups", "Conv3d");
                mapper.enter_module("padding", "Conv3d");
                let padding = burn::module::Module::<B>::map(self.padding, mapper);
                mapper.exit_module("padding", "Conv3d");
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                }
            }
            fn collect_devices(
                &self,
                devices: burn::module::Devices<B>,
            ) -> burn::module::Devices<B> {
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.weight, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.bias, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.stride, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.kernel_size, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.dilation, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.groups, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.padding, devices);
                devices
            }
            fn to_device(self, device: &B::Device) -> Self {
                let weight = burn::module::Module::<B>::to_device(self.weight, device);
                let bias = burn::module::Module::<B>::to_device(self.bias, device);
                let stride = burn::module::Module::<B>::to_device(self.stride, device);
                let kernel_size = burn::module::Module::<
                    B,
                >::to_device(self.kernel_size, device);
                let dilation = burn::module::Module::<
                    B,
                >::to_device(self.dilation, device);
                let groups = burn::module::Module::<B>::to_device(self.groups, device);
                let padding = burn::module::Module::<B>::to_device(self.padding, device);
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                }
            }
            fn fork(self, device: &B::Device) -> Self {
                let weight = burn::module::Module::<B>::fork(self.weight, device);
                let bias = burn::module::Module::<B>::fork(self.bias, device);
                let stride = burn::module::Module::<B>::fork(self.stride, device);
                let kernel_size = burn::module::Module::<
                    B,
                >::fork(self.kernel_size, device);
                let dilation = burn::module::Module::<B>::fork(self.dilation, device);
                let groups = burn::module::Module::<B>::fork(self.groups, device);
                let padding = burn::module::Module::<B>::fork(self.padding, device);
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                }
            }
        }
        impl<B: Backend> burn::module::AutodiffModule<B> for Conv3d<B>
        where
            B: burn::tensor::backend::AutodiffBackend,
            <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
        {
            type InnerModule = Conv3d<B::InnerBackend>;
            fn valid(&self) -> Self::InnerModule {
                let weight = burn::module::AutodiffModule::<B>::valid(&self.weight);
                let bias = burn::module::AutodiffModule::<B>::valid(&self.bias);
                let stride = burn::module::AutodiffModule::<B>::valid(&self.stride);
                let kernel_size = burn::module::AutodiffModule::<
                    B,
                >::valid(&self.kernel_size);
                let dilation = burn::module::AutodiffModule::<B>::valid(&self.dilation);
                let groups = burn::module::AutodiffModule::<B>::valid(&self.groups);
                let padding = burn::module::AutodiffModule::<B>::valid(&self.padding);
                Self::InnerModule {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                }
            }
        }
        impl<B: Backend> core::fmt::Display for Conv3d<B> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let formatted = burn::module::ModuleDisplay::format(
                    self,
                    Default::default(),
                );
                f.write_fmt(format_args!("{0}", formatted))
            }
        }
        impl<B: Backend> burn::module::ModuleDisplayDefault for Conv3d<B> {
            fn content(
                &self,
                mut content: burn::module::Content,
            ) -> Option<burn::module::Content> {
                content
                    .set_top_level_type(&"Conv3d")
                    .add("weight", &self.weight)
                    .add("bias", &self.bias)
                    .add("stride", &self.stride)
                    .add("kernel_size", &self.kernel_size)
                    .add("dilation", &self.dilation)
                    .add("groups", &self.groups)
                    .add("padding", &self.padding)
                    .optional()
            }
            fn num_params(&self) -> usize {
                burn::module::Module::num_params(self)
            }
        }
        impl<B: Backend> Clone for Conv3d<B> {
            fn clone(&self) -> Self {
                let weight = self.weight.clone();
                let bias = self.bias.clone();
                let stride = self.stride.clone();
                let kernel_size = self.kernel_size.clone();
                let dilation = self.dilation.clone();
                let groups = self.groups.clone();
                let padding = self.padding.clone();
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                }
            }
        }
        /// The record type for the module.
        pub struct Conv3dRecord<B: Backend> {
            /// The module record associative type.
            pub weight: <Param<Tensor<B, 5>> as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub bias: <Option<Param<Tensor<B, 1>>> as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub stride: <[usize; 3] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub kernel_size: <[usize; 3] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub dilation: <[usize; 3] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub groups: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub padding: <Ignored<PaddingConfig3d> as burn::module::Module<B>>::Record,
        }
        /// The record item type for the module.
        #[serde(crate = "burn::serde")]
        #[serde(
            bound = "< < Param < Tensor < B, 5 > > as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize +\nburn :: serde :: de :: DeserializeOwned, < < Option < Param < Tensor < B, 1 >\n> > as burn :: module :: Module < B > > :: Record as burn :: record :: Record\n< B >> :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de ::\nDeserializeOwned, < < [usize; 3] as burn :: module :: Module < B > > :: Record\nas burn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize\n+ burn :: serde :: de :: DeserializeOwned, < < [usize; 3] as burn :: module ::\nModule < B > > :: Record as burn :: record :: Record < B >> :: Item < S > :\nburn :: serde :: Serialize + burn :: serde :: de :: DeserializeOwned, < <\n[usize; 3] as burn :: module :: Module < B > > :: Record as burn :: record ::\nRecord < B >> :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de\n:: DeserializeOwned, < < usize as burn :: module :: Module < B > > :: Record\nas burn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize\n+ burn :: serde :: de :: DeserializeOwned, < < Ignored < PaddingConfig3d > as\nburn :: module :: Module < B > > :: Record as burn :: record :: Record < B >>\n:: Item < S > : burn :: serde :: Serialize + burn :: serde :: de ::\nDeserializeOwned,"
        )]
        pub struct Conv3dRecordItem<B: Backend, S: burn::record::PrecisionSettings> {
            /// Field to be serialized.
            pub weight: <<Param<
                Tensor<B, 5>,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub bias: <<Option<
                Param<Tensor<B, 1>>,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub stride: <<[usize; 3] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub kernel_size: <<[usize; 3] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub dilation: <<[usize; 3] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub groups: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub padding: <<Ignored<
                PaddingConfig3d,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
        }
        #[doc(hidden)]
        #[allow(
            non_upper_case_globals,
            unused_attributes,
            unused_qualifications,
            clippy::absolute_paths,
        )]
        const _: () = {
            use burn::serde as _serde;
            #[automatically_derived]
            impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
            for Conv3dRecordItem<B, S>
            where
                <<Param<
                    Tensor<B, 5>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Option<
                    Param<Tensor<B, 1>>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Ignored<
                    PaddingConfig3d,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            {
                fn serialize<__S>(
                    &self,
                    __serializer: __S,
                ) -> _serde::__private226::Result<__S::Ok, __S::Error>
                where
                    __S: _serde::Serializer,
                {
                    let mut __serde_state = _serde::Serializer::serialize_struct(
                        __serializer,
                        "Conv3dRecordItem",
                        false as usize + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "weight",
                        &self.weight,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "bias",
                        &self.bias,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "stride",
                        &self.stride,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "kernel_size",
                        &self.kernel_size,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "dilation",
                        &self.dilation,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "groups",
                        &self.groups,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "padding",
                        &self.padding,
                    )?;
                    _serde::ser::SerializeStruct::end(__serde_state)
                }
            }
        };
        #[doc(hidden)]
        #[allow(
            non_upper_case_globals,
            unused_attributes,
            unused_qualifications,
            clippy::absolute_paths,
        )]
        const _: () = {
            use burn::serde as _serde;
            #[automatically_derived]
            impl<
                'de,
                B: Backend,
                S: burn::record::PrecisionSettings,
            > _serde::Deserialize<'de> for Conv3dRecordItem<B, S>
            where
                <<Param<
                    Tensor<B, 5>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Option<
                    Param<Tensor<B, 1>>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Ignored<
                    PaddingConfig3d,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            {
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private226::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    #[allow(non_camel_case_types)]
                    #[doc(hidden)]
                    enum __Field {
                        __field0,
                        __field1,
                        __field2,
                        __field3,
                        __field4,
                        __field5,
                        __field6,
                        __ignore,
                    }
                    #[doc(hidden)]
                    struct __FieldVisitor;
                    #[automatically_derived]
                    impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                        type Value = __Field;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private226::Formatter,
                        ) -> _serde::__private226::fmt::Result {
                            _serde::__private226::Formatter::write_str(
                                __formatter,
                                "field identifier",
                            )
                        }
                        fn visit_u64<__E>(
                            self,
                            __value: u64,
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                0u64 => _serde::__private226::Ok(__Field::__field0),
                                1u64 => _serde::__private226::Ok(__Field::__field1),
                                2u64 => _serde::__private226::Ok(__Field::__field2),
                                3u64 => _serde::__private226::Ok(__Field::__field3),
                                4u64 => _serde::__private226::Ok(__Field::__field4),
                                5u64 => _serde::__private226::Ok(__Field::__field5),
                                6u64 => _serde::__private226::Ok(__Field::__field6),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_str<__E>(
                            self,
                            __value: &str,
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                "weight" => _serde::__private226::Ok(__Field::__field0),
                                "bias" => _serde::__private226::Ok(__Field::__field1),
                                "stride" => _serde::__private226::Ok(__Field::__field2),
                                "kernel_size" => _serde::__private226::Ok(__Field::__field3),
                                "dilation" => _serde::__private226::Ok(__Field::__field4),
                                "groups" => _serde::__private226::Ok(__Field::__field5),
                                "padding" => _serde::__private226::Ok(__Field::__field6),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_bytes<__E>(
                            self,
                            __value: &[u8],
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                b"weight" => _serde::__private226::Ok(__Field::__field0),
                                b"bias" => _serde::__private226::Ok(__Field::__field1),
                                b"stride" => _serde::__private226::Ok(__Field::__field2),
                                b"kernel_size" => {
                                    _serde::__private226::Ok(__Field::__field3)
                                }
                                b"dilation" => _serde::__private226::Ok(__Field::__field4),
                                b"groups" => _serde::__private226::Ok(__Field::__field5),
                                b"padding" => _serde::__private226::Ok(__Field::__field6),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                    }
                    #[automatically_derived]
                    impl<'de> _serde::Deserialize<'de> for __Field {
                        #[inline]
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private226::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            _serde::Deserializer::deserialize_identifier(
                                __deserializer,
                                __FieldVisitor,
                            )
                        }
                    }
                    #[doc(hidden)]
                    struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                    where
                        <<Param<
                            Tensor<B, 5>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Option<
                            Param<Tensor<B, 1>>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Ignored<
                            PaddingConfig3d,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    {
                        marker: _serde::__private226::PhantomData<
                            Conv3dRecordItem<B, S>,
                        >,
                        lifetime: _serde::__private226::PhantomData<&'de ()>,
                    }
                    #[automatically_derived]
                    impl<
                        'de,
                        B: Backend,
                        S: burn::record::PrecisionSettings,
                    > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                    where
                        <<Param<
                            Tensor<B, 5>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Option<
                            Param<Tensor<B, 1>>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Ignored<
                            PaddingConfig3d,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    {
                        type Value = Conv3dRecordItem<B, S>;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private226::Formatter,
                        ) -> _serde::__private226::fmt::Result {
                            _serde::__private226::Formatter::write_str(
                                __formatter,
                                "struct Conv3dRecordItem",
                            )
                        }
                        #[inline]
                        fn visit_seq<__A>(
                            self,
                            mut __seq: __A,
                        ) -> _serde::__private226::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::SeqAccess<'de>,
                        {
                            let __field0 = match _serde::de::SeqAccess::next_element::<
                                <<Param<
                                    Tensor<B, 5>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            0usize,
                                            &"struct Conv3dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field1 = match _serde::de::SeqAccess::next_element::<
                                <<Option<
                                    Param<Tensor<B, 1>>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            1usize,
                                            &"struct Conv3dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field2 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            2usize,
                                            &"struct Conv3dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field3 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            3usize,
                                            &"struct Conv3dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field4 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            4usize,
                                            &"struct Conv3dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field5 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            5usize,
                                            &"struct Conv3dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            let __field6 = match _serde::de::SeqAccess::next_element::<
                                <<Ignored<
                                    PaddingConfig3d,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            6usize,
                                            &"struct Conv3dRecordItem with 7 elements",
                                        ),
                                    );
                                }
                            };
                            _serde::__private226::Ok(Conv3dRecordItem {
                                weight: __field0,
                                bias: __field1,
                                stride: __field2,
                                kernel_size: __field3,
                                dilation: __field4,
                                groups: __field5,
                                padding: __field6,
                            })
                        }
                        #[inline]
                        fn visit_map<__A>(
                            self,
                            mut __map: __A,
                        ) -> _serde::__private226::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::MapAccess<'de>,
                        {
                            let mut __field0: _serde::__private226::Option<
                                <<Param<
                                    Tensor<B, 5>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field1: _serde::__private226::Option<
                                <<Option<
                                    Param<Tensor<B, 1>>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field2: _serde::__private226::Option<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field3: _serde::__private226::Option<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field4: _serde::__private226::Option<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field5: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field6: _serde::__private226::Option<
                                <<Ignored<
                                    PaddingConfig3d,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            while let _serde::__private226::Some(__key) = _serde::de::MapAccess::next_key::<
                                __Field,
                            >(&mut __map)? {
                                match __key {
                                    __Field::__field0 => {
                                        if _serde::__private226::Option::is_some(&__field0) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("weight"),
                                            );
                                        }
                                        __field0 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Param<
                                                    Tensor<B, 5>,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field1 => {
                                        if _serde::__private226::Option::is_some(&__field1) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("bias"),
                                            );
                                        }
                                        __field1 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Option<
                                                    Param<Tensor<B, 1>>,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field2 => {
                                        if _serde::__private226::Option::is_some(&__field2) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                            );
                                        }
                                        __field2 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 3] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field3 => {
                                        if _serde::__private226::Option::is_some(&__field3) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "kernel_size",
                                                ),
                                            );
                                        }
                                        __field3 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 3] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field4 => {
                                        if _serde::__private226::Option::is_some(&__field4) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "dilation",
                                                ),
                                            );
                                        }
                                        __field4 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 3] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field5 => {
                                        if _serde::__private226::Option::is_some(&__field5) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("groups"),
                                            );
                                        }
                                        __field5 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field6 => {
                                        if _serde::__private226::Option::is_some(&__field6) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "padding",
                                                ),
                                            );
                                        }
                                        __field6 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Ignored<
                                                    PaddingConfig3d,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    _ => {
                                        let _ = _serde::de::MapAccess::next_value::<
                                            _serde::de::IgnoredAny,
                                        >(&mut __map)?;
                                    }
                                }
                            }
                            let __field0 = match __field0 {
                                _serde::__private226::Some(__field0) => __field0,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("weight")?
                                }
                            };
                            let __field1 = match __field1 {
                                _serde::__private226::Some(__field1) => __field1,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("bias")?
                                }
                            };
                            let __field2 = match __field2 {
                                _serde::__private226::Some(__field2) => __field2,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("stride")?
                                }
                            };
                            let __field3 = match __field3 {
                                _serde::__private226::Some(__field3) => __field3,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("kernel_size")?
                                }
                            };
                            let __field4 = match __field4 {
                                _serde::__private226::Some(__field4) => __field4,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("dilation")?
                                }
                            };
                            let __field5 = match __field5 {
                                _serde::__private226::Some(__field5) => __field5,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("groups")?
                                }
                            };
                            let __field6 = match __field6 {
                                _serde::__private226::Some(__field6) => __field6,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("padding")?
                                }
                            };
                            _serde::__private226::Ok(Conv3dRecordItem {
                                weight: __field0,
                                bias: __field1,
                                stride: __field2,
                                kernel_size: __field3,
                                dilation: __field4,
                                groups: __field5,
                                padding: __field6,
                            })
                        }
                    }
                    #[doc(hidden)]
                    const FIELDS: &'static [&'static str] = &[
                        "weight",
                        "bias",
                        "stride",
                        "kernel_size",
                        "dilation",
                        "groups",
                        "padding",
                    ];
                    _serde::Deserializer::deserialize_struct(
                        __deserializer,
                        "Conv3dRecordItem",
                        FIELDS,
                        __Visitor {
                            marker: _serde::__private226::PhantomData::<
                                Conv3dRecordItem<B, S>,
                            >,
                            lifetime: _serde::__private226::PhantomData,
                        },
                    )
                }
            }
        };
        impl<B: Backend, S: burn::record::PrecisionSettings> Clone
        for Conv3dRecordItem<B, S> {
            fn clone(&self) -> Self {
                Self {
                    weight: self.weight.clone(),
                    bias: self.bias.clone(),
                    stride: self.stride.clone(),
                    kernel_size: self.kernel_size.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                }
            }
        }
        impl<B: Backend> burn::record::Record<B> for Conv3dRecord<B> {
            type Item<S: burn::record::PrecisionSettings> = Conv3dRecordItem<B, S>;
            fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
                Conv3dRecordItem {
                    weight: burn::record::Record::<B>::into_item::<S>(self.weight),
                    bias: burn::record::Record::<B>::into_item::<S>(self.bias),
                    stride: burn::record::Record::<B>::into_item::<S>(self.stride),
                    kernel_size: burn::record::Record::<
                        B,
                    >::into_item::<S>(self.kernel_size),
                    dilation: burn::record::Record::<B>::into_item::<S>(self.dilation),
                    groups: burn::record::Record::<B>::into_item::<S>(self.groups),
                    padding: burn::record::Record::<B>::into_item::<S>(self.padding),
                }
            }
            fn from_item<S: burn::record::PrecisionSettings>(
                item: Self::Item<S>,
                device: &B::Device,
            ) -> Self {
                Self {
                    weight: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.weight, device),
                    bias: burn::record::Record::<B>::from_item::<S>(item.bias, device),
                    stride: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.stride, device),
                    kernel_size: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.kernel_size, device),
                    dilation: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.dilation, device),
                    groups: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.groups, device),
                    padding: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.padding, device),
                }
            }
        }
        #[automatically_derived]
        impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for Conv3d<B> {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                let names: &'static _ = &[
                    "weight",
                    "bias",
                    "stride",
                    "kernel_size",
                    "dilation",
                    "groups",
                    "padding",
                ];
                let values: &[&dyn ::core::fmt::Debug] = &[
                    &self.weight,
                    &self.bias,
                    &self.stride,
                    &self.kernel_size,
                    &self.dilation,
                    &self.groups,
                    &&self.padding,
                ];
                ::core::fmt::Formatter::debug_struct_fields_finish(
                    f,
                    "Conv3d",
                    names,
                    values,
                )
            }
        }
        impl Conv3dConfig {
            /// Initialize a new [conv3d](Conv3d) module.
            pub fn init<B: Backend>(&self, device: &B::Device) -> Conv3d<B> {
                checks::checks_channels_div_groups(
                    self.channels[0],
                    self.channels[1],
                    self.groups,
                );
                if self.padding == PaddingConfig3d::Same {
                    checks::check_same_padding_support(&self.kernel_size);
                }
                let shape = [
                    self.channels[1],
                    self.channels[0] / self.groups,
                    self.kernel_size[0],
                    self.kernel_size[1],
                    self.kernel_size[2],
                ];
                let k = self.kernel_size.iter().product::<usize>();
                let fan_in = self.channels[0] / self.groups * k;
                let fan_out = self.channels[1] / self.groups * k;
                let weight = self
                    .initializer
                    .init_with(shape, Some(fan_in), Some(fan_out), device);
                let mut bias = None;
                if self.bias {
                    bias = Some(
                        self
                            .initializer
                            .init_with(
                                [self.channels[1]],
                                Some(fan_in),
                                Some(fan_out),
                                device,
                            ),
                    );
                }
                Conv3d {
                    weight,
                    bias,
                    stride: self.stride,
                    kernel_size: self.kernel_size,
                    dilation: self.dilation,
                    padding: Ignored(self.padding.clone()),
                    groups: self.groups,
                }
            }
        }
        impl<B: Backend> ModuleDisplay for Conv3d<B> {
            fn custom_settings(&self) -> Option<DisplaySettings> {
                DisplaySettings::new().with_new_line_after_attribute(false).optional()
            }
            fn custom_content(&self, content: Content) -> Option<Content> {
                let padding_formatted = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0}", &self.padding))
                });
                let stride = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", self.stride))
                });
                let kernel_size = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", self.kernel_size))
                });
                let dilation = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", self.dilation))
                });
                let [channels_out, group_channels_in, _, _, _] = self.weight.dims();
                let channels_in = group_channels_in * self.groups;
                let ch_out = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", channels_out))
                });
                let ch_in = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", channels_in))
                });
                content
                    .add("ch_in", &ch_in)
                    .add("ch_out", &ch_out)
                    .add("stride", &stride)
                    .add("kernel_size", &kernel_size)
                    .add("dilation", &dilation)
                    .add("groups", &self.groups)
                    .add("padding", &padding_formatted)
                    .optional()
            }
        }
        impl<B: Backend> Conv3d<B> {
            /// Applies the forward pass on the input tensor.
            ///
            /// See [conv3d](burn::tensor::module::conv3d) for more information.
            ///
            /// # Shapes
            ///
            /// - input: `[batch_size, channels_in, depth_in, height_in, width_in]`
            /// - output: `[batch_size, channels_out, depth_out, height_out, width_out]`
            pub fn forward(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
                let [_batch_size, _channels_in, depth_in, height_in, width_in] = input
                    .dims();
                let padding = self
                    .padding
                    .calculate_padding_3d(
                        depth_in,
                        height_in,
                        width_in,
                        &self.kernel_size,
                        &self.stride,
                    );
                conv3d(
                    input,
                    self.weight.val(),
                    self.bias.as_ref().map(|bias| bias.val()),
                    ConvOptions::new(self.stride, padding, self.dilation, self.groups),
                )
            }
        }
    }
    mod conv_transpose1d {
        use alloc::format;
        use burn_core as burn;
        use crate::conv::checks;
        use burn::config::Config;
        use burn::module::Content;
        use burn::module::DisplaySettings;
        use burn::module::Initializer;
        use burn::module::Module;
        use burn::module::ModuleDisplay;
        use burn::module::Param;
        use burn::tensor::Tensor;
        use burn::tensor::backend::Backend;
        use burn::tensor::module::conv_transpose1d;
        use burn::tensor::ops::ConvTransposeOptions;
        /// Configuration to create an [1D transposed convolution](ConvTranspose1d) layer
        /// using the [init function](ConvTranspose1dConfig::init).
        pub struct ConvTranspose1dConfig {
            /// The number of channels.
            pub channels: [usize; 2],
            /// The size of the kernel.
            pub kernel_size: usize,
            /// The stride of the convolution.
            #[config(default = "1")]
            pub stride: usize,
            /// Spacing between kernel elements.
            #[config(default = "1")]
            pub dilation: usize,
            /// Controls the connections between input and output channels.
            #[config(default = "1")]
            pub groups: usize,
            /// The padding configuration.
            #[config(default = "0")]
            pub padding: usize,
            /// The padding output configuration.
            #[config(default = "0")]
            pub padding_out: usize,
            /// If bias should be added to the output.
            #[config(default = true)]
            pub bias: bool,
            /// The type of function used to initialize neural network parameters
            #[config(
                default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
            )]
            pub initializer: Initializer,
        }
        impl burn::config::Config for ConvTranspose1dConfig {}
        impl ConvTranspose1dConfig {
            /// Create a new instance of the config.
            #[allow(clippy::too_many_arguments)]
            pub fn new(channels: [usize; 2], kernel_size: usize) -> Self {
                Self {
                    channels: channels,
                    kernel_size: kernel_size,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    padding: 0,
                    padding_out: 0,
                    bias: true,
                    initializer: Initializer::KaimingUniform {
                        gain: 1.0 / num_traits::Float::sqrt(3.0),
                        fan_out_only: false,
                    },
                }
            }
        }
        impl ConvTranspose1dConfig {
            /// The stride of the convolution.
            pub fn with_stride(mut self, stride: usize) -> Self {
                self.stride = stride;
                self
            }
            /// Spacing between kernel elements.
            pub fn with_dilation(mut self, dilation: usize) -> Self {
                self.dilation = dilation;
                self
            }
            /// Controls the connections between input and output channels.
            pub fn with_groups(mut self, groups: usize) -> Self {
                self.groups = groups;
                self
            }
            /// The padding configuration.
            pub fn with_padding(mut self, padding: usize) -> Self {
                self.padding = padding;
                self
            }
            /// The padding output configuration.
            pub fn with_padding_out(mut self, padding_out: usize) -> Self {
                self.padding_out = padding_out;
                self
            }
            /// If bias should be added to the output.
            pub fn with_bias(mut self, bias: bool) -> Self {
                self.bias = bias;
                self
            }
            /// The type of function used to initialize neural network parameters
            pub fn with_initializer(mut self, initializer: Initializer) -> Self {
                self.initializer = initializer;
                self
            }
        }
        impl burn::serde::Serialize for ConvTranspose1dConfig {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: burn::serde::Serializer,
            {
                #[serde(crate = "burn::serde")]
                struct ConvTranspose1dConfigSerde {
                    channels: [usize; 2],
                    kernel_size: usize,
                    stride: usize,
                    dilation: usize,
                    groups: usize,
                    padding: usize,
                    padding_out: usize,
                    bias: bool,
                    initializer: Initializer,
                }
                #[doc(hidden)]
                #[allow(
                    non_upper_case_globals,
                    unused_attributes,
                    unused_qualifications,
                    clippy::absolute_paths,
                )]
                const _: () = {
                    use burn::serde as _serde;
                    #[automatically_derived]
                    impl _serde::Serialize for ConvTranspose1dConfigSerde {
                        fn serialize<__S>(
                            &self,
                            __serializer: __S,
                        ) -> _serde::__private226::Result<__S::Ok, __S::Error>
                        where
                            __S: _serde::Serializer,
                        {
                            let mut __serde_state = _serde::Serializer::serialize_struct(
                                __serializer,
                                "ConvTranspose1dConfigSerde",
                                false as usize + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "channels",
                                &self.channels,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "kernel_size",
                                &self.kernel_size,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "stride",
                                &self.stride,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "dilation",
                                &self.dilation,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "groups",
                                &self.groups,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "padding",
                                &self.padding,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "padding_out",
                                &self.padding_out,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "bias",
                                &self.bias,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "initializer",
                                &self.initializer,
                            )?;
                            _serde::ser::SerializeStruct::end(__serde_state)
                        }
                    }
                };
                let serde_state = ConvTranspose1dConfigSerde {
                    channels: self.channels.clone(),
                    kernel_size: self.kernel_size.clone(),
                    stride: self.stride.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                    padding_out: self.padding_out.clone(),
                    bias: self.bias.clone(),
                    initializer: self.initializer.clone(),
                };
                serde_state.serialize(serializer)
            }
        }
        impl<'de> burn::serde::Deserialize<'de> for ConvTranspose1dConfig {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: burn::serde::Deserializer<'de>,
            {
                #[serde(crate = "burn::serde")]
                struct ConvTranspose1dConfigSerde {
                    channels: [usize; 2],
                    kernel_size: usize,
                    stride: usize,
                    dilation: usize,
                    groups: usize,
                    padding: usize,
                    padding_out: usize,
                    bias: bool,
                    initializer: Initializer,
                }
                #[doc(hidden)]
                #[allow(
                    non_upper_case_globals,
                    unused_attributes,
                    unused_qualifications,
                    clippy::absolute_paths,
                )]
                const _: () = {
                    use burn::serde as _serde;
                    #[automatically_derived]
                    impl<'de> _serde::Deserialize<'de> for ConvTranspose1dConfigSerde {
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private226::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            #[allow(non_camel_case_types)]
                            #[doc(hidden)]
                            enum __Field {
                                __field0,
                                __field1,
                                __field2,
                                __field3,
                                __field4,
                                __field5,
                                __field6,
                                __field7,
                                __field8,
                                __ignore,
                            }
                            #[doc(hidden)]
                            struct __FieldVisitor;
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                type Value = __Field;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private226::Formatter,
                                ) -> _serde::__private226::fmt::Result {
                                    _serde::__private226::Formatter::write_str(
                                        __formatter,
                                        "field identifier",
                                    )
                                }
                                fn visit_u64<__E>(
                                    self,
                                    __value: u64,
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        0u64 => _serde::__private226::Ok(__Field::__field0),
                                        1u64 => _serde::__private226::Ok(__Field::__field1),
                                        2u64 => _serde::__private226::Ok(__Field::__field2),
                                        3u64 => _serde::__private226::Ok(__Field::__field3),
                                        4u64 => _serde::__private226::Ok(__Field::__field4),
                                        5u64 => _serde::__private226::Ok(__Field::__field5),
                                        6u64 => _serde::__private226::Ok(__Field::__field6),
                                        7u64 => _serde::__private226::Ok(__Field::__field7),
                                        8u64 => _serde::__private226::Ok(__Field::__field8),
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_str<__E>(
                                    self,
                                    __value: &str,
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        "channels" => _serde::__private226::Ok(__Field::__field0),
                                        "kernel_size" => _serde::__private226::Ok(__Field::__field1),
                                        "stride" => _serde::__private226::Ok(__Field::__field2),
                                        "dilation" => _serde::__private226::Ok(__Field::__field3),
                                        "groups" => _serde::__private226::Ok(__Field::__field4),
                                        "padding" => _serde::__private226::Ok(__Field::__field5),
                                        "padding_out" => _serde::__private226::Ok(__Field::__field6),
                                        "bias" => _serde::__private226::Ok(__Field::__field7),
                                        "initializer" => _serde::__private226::Ok(__Field::__field8),
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_bytes<__E>(
                                    self,
                                    __value: &[u8],
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        b"channels" => _serde::__private226::Ok(__Field::__field0),
                                        b"kernel_size" => {
                                            _serde::__private226::Ok(__Field::__field1)
                                        }
                                        b"stride" => _serde::__private226::Ok(__Field::__field2),
                                        b"dilation" => _serde::__private226::Ok(__Field::__field3),
                                        b"groups" => _serde::__private226::Ok(__Field::__field4),
                                        b"padding" => _serde::__private226::Ok(__Field::__field5),
                                        b"padding_out" => {
                                            _serde::__private226::Ok(__Field::__field6)
                                        }
                                        b"bias" => _serde::__private226::Ok(__Field::__field7),
                                        b"initializer" => {
                                            _serde::__private226::Ok(__Field::__field8)
                                        }
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                            }
                            #[automatically_derived]
                            impl<'de> _serde::Deserialize<'de> for __Field {
                                #[inline]
                                fn deserialize<__D>(
                                    __deserializer: __D,
                                ) -> _serde::__private226::Result<Self, __D::Error>
                                where
                                    __D: _serde::Deserializer<'de>,
                                {
                                    _serde::Deserializer::deserialize_identifier(
                                        __deserializer,
                                        __FieldVisitor,
                                    )
                                }
                            }
                            #[doc(hidden)]
                            struct __Visitor<'de> {
                                marker: _serde::__private226::PhantomData<
                                    ConvTranspose1dConfigSerde,
                                >,
                                lifetime: _serde::__private226::PhantomData<&'de ()>,
                            }
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                type Value = ConvTranspose1dConfigSerde;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private226::Formatter,
                                ) -> _serde::__private226::fmt::Result {
                                    _serde::__private226::Formatter::write_str(
                                        __formatter,
                                        "struct ConvTranspose1dConfigSerde",
                                    )
                                }
                                #[inline]
                                fn visit_seq<__A>(
                                    self,
                                    mut __seq: __A,
                                ) -> _serde::__private226::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::SeqAccess<'de>,
                                {
                                    let __field0 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    0usize,
                                                    &"struct ConvTranspose1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field1 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    1usize,
                                                    &"struct ConvTranspose1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field2 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    2usize,
                                                    &"struct ConvTranspose1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field3 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    3usize,
                                                    &"struct ConvTranspose1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field4 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    4usize,
                                                    &"struct ConvTranspose1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field5 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    5usize,
                                                    &"struct ConvTranspose1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field6 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    6usize,
                                                    &"struct ConvTranspose1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field7 = match _serde::de::SeqAccess::next_element::<
                                        bool,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    7usize,
                                                    &"struct ConvTranspose1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field8 = match _serde::de::SeqAccess::next_element::<
                                        Initializer,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    8usize,
                                                    &"struct ConvTranspose1dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    _serde::__private226::Ok(ConvTranspose1dConfigSerde {
                                        channels: __field0,
                                        kernel_size: __field1,
                                        stride: __field2,
                                        dilation: __field3,
                                        groups: __field4,
                                        padding: __field5,
                                        padding_out: __field6,
                                        bias: __field7,
                                        initializer: __field8,
                                    })
                                }
                                #[inline]
                                fn visit_map<__A>(
                                    self,
                                    mut __map: __A,
                                ) -> _serde::__private226::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::MapAccess<'de>,
                                {
                                    let mut __field0: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field1: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field2: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field3: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field4: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field5: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field6: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field7: _serde::__private226::Option<bool> = _serde::__private226::None;
                                    let mut __field8: _serde::__private226::Option<
                                        Initializer,
                                    > = _serde::__private226::None;
                                    while let _serde::__private226::Some(__key) = _serde::de::MapAccess::next_key::<
                                        __Field,
                                    >(&mut __map)? {
                                        match __key {
                                            __Field::__field0 => {
                                                if _serde::__private226::Option::is_some(&__field0) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "channels",
                                                        ),
                                                    );
                                                }
                                                __field0 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field1 => {
                                                if _serde::__private226::Option::is_some(&__field1) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "kernel_size",
                                                        ),
                                                    );
                                                }
                                                __field1 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field2 => {
                                                if _serde::__private226::Option::is_some(&__field2) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                    );
                                                }
                                                __field2 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field3 => {
                                                if _serde::__private226::Option::is_some(&__field3) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "dilation",
                                                        ),
                                                    );
                                                }
                                                __field3 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field4 => {
                                                if _serde::__private226::Option::is_some(&__field4) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("groups"),
                                                    );
                                                }
                                                __field4 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field5 => {
                                                if _serde::__private226::Option::is_some(&__field5) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "padding",
                                                        ),
                                                    );
                                                }
                                                __field5 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field6 => {
                                                if _serde::__private226::Option::is_some(&__field6) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "padding_out",
                                                        ),
                                                    );
                                                }
                                                __field6 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field7 => {
                                                if _serde::__private226::Option::is_some(&__field7) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("bias"),
                                                    );
                                                }
                                                __field7 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<bool>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field8 => {
                                                if _serde::__private226::Option::is_some(&__field8) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "initializer",
                                                        ),
                                                    );
                                                }
                                                __field8 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<
                                                        Initializer,
                                                    >(&mut __map)?,
                                                );
                                            }
                                            _ => {
                                                let _ = _serde::de::MapAccess::next_value::<
                                                    _serde::de::IgnoredAny,
                                                >(&mut __map)?;
                                            }
                                        }
                                    }
                                    let __field0 = match __field0 {
                                        _serde::__private226::Some(__field0) => __field0,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("channels")?
                                        }
                                    };
                                    let __field1 = match __field1 {
                                        _serde::__private226::Some(__field1) => __field1,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("kernel_size")?
                                        }
                                    };
                                    let __field2 = match __field2 {
                                        _serde::__private226::Some(__field2) => __field2,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("stride")?
                                        }
                                    };
                                    let __field3 = match __field3 {
                                        _serde::__private226::Some(__field3) => __field3,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("dilation")?
                                        }
                                    };
                                    let __field4 = match __field4 {
                                        _serde::__private226::Some(__field4) => __field4,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("groups")?
                                        }
                                    };
                                    let __field5 = match __field5 {
                                        _serde::__private226::Some(__field5) => __field5,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("padding")?
                                        }
                                    };
                                    let __field6 = match __field6 {
                                        _serde::__private226::Some(__field6) => __field6,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("padding_out")?
                                        }
                                    };
                                    let __field7 = match __field7 {
                                        _serde::__private226::Some(__field7) => __field7,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("bias")?
                                        }
                                    };
                                    let __field8 = match __field8 {
                                        _serde::__private226::Some(__field8) => __field8,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("initializer")?
                                        }
                                    };
                                    _serde::__private226::Ok(ConvTranspose1dConfigSerde {
                                        channels: __field0,
                                        kernel_size: __field1,
                                        stride: __field2,
                                        dilation: __field3,
                                        groups: __field4,
                                        padding: __field5,
                                        padding_out: __field6,
                                        bias: __field7,
                                        initializer: __field8,
                                    })
                                }
                            }
                            #[doc(hidden)]
                            const FIELDS: &'static [&'static str] = &[
                                "channels",
                                "kernel_size",
                                "stride",
                                "dilation",
                                "groups",
                                "padding",
                                "padding_out",
                                "bias",
                                "initializer",
                            ];
                            _serde::Deserializer::deserialize_struct(
                                __deserializer,
                                "ConvTranspose1dConfigSerde",
                                FIELDS,
                                __Visitor {
                                    marker: _serde::__private226::PhantomData::<
                                        ConvTranspose1dConfigSerde,
                                    >,
                                    lifetime: _serde::__private226::PhantomData,
                                },
                            )
                        }
                    }
                };
                let serde_state = ConvTranspose1dConfigSerde::deserialize(deserializer)?;
                Ok(ConvTranspose1dConfig {
                    channels: serde_state.channels,
                    kernel_size: serde_state.kernel_size,
                    stride: serde_state.stride,
                    dilation: serde_state.dilation,
                    groups: serde_state.groups,
                    padding: serde_state.padding,
                    padding_out: serde_state.padding_out,
                    bias: serde_state.bias,
                    initializer: serde_state.initializer,
                })
            }
        }
        impl Clone for ConvTranspose1dConfig {
            fn clone(&self) -> Self {
                Self {
                    channels: self.channels.clone(),
                    kernel_size: self.kernel_size.clone(),
                    stride: self.stride.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                    padding_out: self.padding_out.clone(),
                    bias: self.bias.clone(),
                    initializer: self.initializer.clone(),
                }
            }
        }
        impl core::fmt::Display for ConvTranspose1dConfig {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(&burn::config::config_to_json(self))
            }
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for ConvTranspose1dConfig {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                let names: &'static _ = &[
                    "channels",
                    "kernel_size",
                    "stride",
                    "dilation",
                    "groups",
                    "padding",
                    "padding_out",
                    "bias",
                    "initializer",
                ];
                let values: &[&dyn ::core::fmt::Debug] = &[
                    &self.channels,
                    &self.kernel_size,
                    &self.stride,
                    &self.dilation,
                    &self.groups,
                    &self.padding,
                    &self.padding_out,
                    &self.bias,
                    &&self.initializer,
                ];
                ::core::fmt::Formatter::debug_struct_fields_finish(
                    f,
                    "ConvTranspose1dConfig",
                    names,
                    values,
                )
            }
        }
        /// Applies a 1D transposed convolution over input tensors.
        #[module(custom_display)]
        pub struct ConvTranspose1d<B: Backend> {
            /// Tensor of shape `[channels_in, channels_out / groups, kernel_size]`
            pub weight: Param<Tensor<B, 3>>,
            /// Tensor of shape `[channels_out]`
            pub bias: Option<Param<Tensor<B, 1>>>,
            /// Stride of the convolution.
            pub stride: usize,
            /// Size of the kernel.
            pub kernel_size: usize,
            /// Spacing between kernel elements.
            pub dilation: usize,
            /// Controls the connections between input and output channels.
            pub groups: usize,
            /// The padding configuration.
            pub padding: usize,
            /// The padding output configuration.
            pub padding_out: usize,
            /// The number of channels.
            pub channels: [usize; 2],
        }
        impl<B: Backend> burn::module::Module<B> for ConvTranspose1d<B> {
            type Record = ConvTranspose1dRecord<B>;
            fn load_record(self, record: Self::Record) -> Self {
                Self {
                    weight: burn::module::Module::<
                        B,
                    >::load_record(self.weight, record.weight),
                    bias: burn::module::Module::<B>::load_record(self.bias, record.bias),
                    stride: burn::module::Module::<
                        B,
                    >::load_record(self.stride, record.stride),
                    kernel_size: burn::module::Module::<
                        B,
                    >::load_record(self.kernel_size, record.kernel_size),
                    dilation: burn::module::Module::<
                        B,
                    >::load_record(self.dilation, record.dilation),
                    groups: burn::module::Module::<
                        B,
                    >::load_record(self.groups, record.groups),
                    padding: burn::module::Module::<
                        B,
                    >::load_record(self.padding, record.padding),
                    padding_out: burn::module::Module::<
                        B,
                    >::load_record(self.padding_out, record.padding_out),
                    channels: burn::module::Module::<
                        B,
                    >::load_record(self.channels, record.channels),
                }
            }
            fn into_record(self) -> Self::Record {
                Self::Record {
                    weight: burn::module::Module::<B>::into_record(self.weight),
                    bias: burn::module::Module::<B>::into_record(self.bias),
                    stride: burn::module::Module::<B>::into_record(self.stride),
                    kernel_size: burn::module::Module::<
                        B,
                    >::into_record(self.kernel_size),
                    dilation: burn::module::Module::<B>::into_record(self.dilation),
                    groups: burn::module::Module::<B>::into_record(self.groups),
                    padding: burn::module::Module::<B>::into_record(self.padding),
                    padding_out: burn::module::Module::<
                        B,
                    >::into_record(self.padding_out),
                    channels: burn::module::Module::<B>::into_record(self.channels),
                }
            }
            fn num_params(&self) -> usize {
                let mut num_params = 0;
                num_params += burn::module::Module::<B>::num_params(&self.weight);
                num_params += burn::module::Module::<B>::num_params(&self.bias);
                num_params += burn::module::Module::<B>::num_params(&self.stride);
                num_params += burn::module::Module::<B>::num_params(&self.kernel_size);
                num_params += burn::module::Module::<B>::num_params(&self.dilation);
                num_params += burn::module::Module::<B>::num_params(&self.groups);
                num_params += burn::module::Module::<B>::num_params(&self.padding);
                num_params += burn::module::Module::<B>::num_params(&self.padding_out);
                num_params += burn::module::Module::<B>::num_params(&self.channels);
                num_params
            }
            fn visit<Visitor: burn::module::ModuleVisitor<B>>(
                &self,
                visitor: &mut Visitor,
            ) {
                visitor.enter_module("weight", "ConvTranspose1d");
                burn::module::Module::visit(&self.weight, visitor);
                visitor.exit_module("weight", "ConvTranspose1d");
                visitor.enter_module("bias", "ConvTranspose1d");
                burn::module::Module::visit(&self.bias, visitor);
                visitor.exit_module("bias", "ConvTranspose1d");
                visitor.enter_module("stride", "ConvTranspose1d");
                burn::module::Module::visit(&self.stride, visitor);
                visitor.exit_module("stride", "ConvTranspose1d");
                visitor.enter_module("kernel_size", "ConvTranspose1d");
                burn::module::Module::visit(&self.kernel_size, visitor);
                visitor.exit_module("kernel_size", "ConvTranspose1d");
                visitor.enter_module("dilation", "ConvTranspose1d");
                burn::module::Module::visit(&self.dilation, visitor);
                visitor.exit_module("dilation", "ConvTranspose1d");
                visitor.enter_module("groups", "ConvTranspose1d");
                burn::module::Module::visit(&self.groups, visitor);
                visitor.exit_module("groups", "ConvTranspose1d");
                visitor.enter_module("padding", "ConvTranspose1d");
                burn::module::Module::visit(&self.padding, visitor);
                visitor.exit_module("padding", "ConvTranspose1d");
                visitor.enter_module("padding_out", "ConvTranspose1d");
                burn::module::Module::visit(&self.padding_out, visitor);
                visitor.exit_module("padding_out", "ConvTranspose1d");
                visitor.enter_module("channels", "ConvTranspose1d");
                burn::module::Module::visit(&self.channels, visitor);
                visitor.exit_module("channels", "ConvTranspose1d");
            }
            fn map<Mapper: burn::module::ModuleMapper<B>>(
                self,
                mapper: &mut Mapper,
            ) -> Self {
                mapper.enter_module("weight", "ConvTranspose1d");
                let weight = burn::module::Module::<B>::map(self.weight, mapper);
                mapper.exit_module("weight", "ConvTranspose1d");
                mapper.enter_module("bias", "ConvTranspose1d");
                let bias = burn::module::Module::<B>::map(self.bias, mapper);
                mapper.exit_module("bias", "ConvTranspose1d");
                mapper.enter_module("stride", "ConvTranspose1d");
                let stride = burn::module::Module::<B>::map(self.stride, mapper);
                mapper.exit_module("stride", "ConvTranspose1d");
                mapper.enter_module("kernel_size", "ConvTranspose1d");
                let kernel_size = burn::module::Module::<
                    B,
                >::map(self.kernel_size, mapper);
                mapper.exit_module("kernel_size", "ConvTranspose1d");
                mapper.enter_module("dilation", "ConvTranspose1d");
                let dilation = burn::module::Module::<B>::map(self.dilation, mapper);
                mapper.exit_module("dilation", "ConvTranspose1d");
                mapper.enter_module("groups", "ConvTranspose1d");
                let groups = burn::module::Module::<B>::map(self.groups, mapper);
                mapper.exit_module("groups", "ConvTranspose1d");
                mapper.enter_module("padding", "ConvTranspose1d");
                let padding = burn::module::Module::<B>::map(self.padding, mapper);
                mapper.exit_module("padding", "ConvTranspose1d");
                mapper.enter_module("padding_out", "ConvTranspose1d");
                let padding_out = burn::module::Module::<
                    B,
                >::map(self.padding_out, mapper);
                mapper.exit_module("padding_out", "ConvTranspose1d");
                mapper.enter_module("channels", "ConvTranspose1d");
                let channels = burn::module::Module::<B>::map(self.channels, mapper);
                mapper.exit_module("channels", "ConvTranspose1d");
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                    padding_out,
                    channels,
                }
            }
            fn collect_devices(
                &self,
                devices: burn::module::Devices<B>,
            ) -> burn::module::Devices<B> {
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.weight, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.bias, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.stride, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.kernel_size, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.dilation, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.groups, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.padding, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.padding_out, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.channels, devices);
                devices
            }
            fn to_device(self, device: &B::Device) -> Self {
                let weight = burn::module::Module::<B>::to_device(self.weight, device);
                let bias = burn::module::Module::<B>::to_device(self.bias, device);
                let stride = burn::module::Module::<B>::to_device(self.stride, device);
                let kernel_size = burn::module::Module::<
                    B,
                >::to_device(self.kernel_size, device);
                let dilation = burn::module::Module::<
                    B,
                >::to_device(self.dilation, device);
                let groups = burn::module::Module::<B>::to_device(self.groups, device);
                let padding = burn::module::Module::<B>::to_device(self.padding, device);
                let padding_out = burn::module::Module::<
                    B,
                >::to_device(self.padding_out, device);
                let channels = burn::module::Module::<
                    B,
                >::to_device(self.channels, device);
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                    padding_out,
                    channels,
                }
            }
            fn fork(self, device: &B::Device) -> Self {
                let weight = burn::module::Module::<B>::fork(self.weight, device);
                let bias = burn::module::Module::<B>::fork(self.bias, device);
                let stride = burn::module::Module::<B>::fork(self.stride, device);
                let kernel_size = burn::module::Module::<
                    B,
                >::fork(self.kernel_size, device);
                let dilation = burn::module::Module::<B>::fork(self.dilation, device);
                let groups = burn::module::Module::<B>::fork(self.groups, device);
                let padding = burn::module::Module::<B>::fork(self.padding, device);
                let padding_out = burn::module::Module::<
                    B,
                >::fork(self.padding_out, device);
                let channels = burn::module::Module::<B>::fork(self.channels, device);
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                    padding_out,
                    channels,
                }
            }
        }
        impl<B: Backend> burn::module::AutodiffModule<B> for ConvTranspose1d<B>
        where
            B: burn::tensor::backend::AutodiffBackend,
            <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
        {
            type InnerModule = ConvTranspose1d<B::InnerBackend>;
            fn valid(&self) -> Self::InnerModule {
                let weight = burn::module::AutodiffModule::<B>::valid(&self.weight);
                let bias = burn::module::AutodiffModule::<B>::valid(&self.bias);
                let stride = burn::module::AutodiffModule::<B>::valid(&self.stride);
                let kernel_size = burn::module::AutodiffModule::<
                    B,
                >::valid(&self.kernel_size);
                let dilation = burn::module::AutodiffModule::<B>::valid(&self.dilation);
                let groups = burn::module::AutodiffModule::<B>::valid(&self.groups);
                let padding = burn::module::AutodiffModule::<B>::valid(&self.padding);
                let padding_out = burn::module::AutodiffModule::<
                    B,
                >::valid(&self.padding_out);
                let channels = burn::module::AutodiffModule::<B>::valid(&self.channels);
                Self::InnerModule {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                    padding_out,
                    channels,
                }
            }
        }
        impl<B: Backend> core::fmt::Display for ConvTranspose1d<B> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let formatted = burn::module::ModuleDisplay::format(
                    self,
                    Default::default(),
                );
                f.write_fmt(format_args!("{0}", formatted))
            }
        }
        impl<B: Backend> burn::module::ModuleDisplayDefault for ConvTranspose1d<B> {
            fn content(
                &self,
                mut content: burn::module::Content,
            ) -> Option<burn::module::Content> {
                content
                    .set_top_level_type(&"ConvTranspose1d")
                    .add("weight", &self.weight)
                    .add("bias", &self.bias)
                    .add("stride", &self.stride)
                    .add("kernel_size", &self.kernel_size)
                    .add("dilation", &self.dilation)
                    .add("groups", &self.groups)
                    .add("padding", &self.padding)
                    .add("padding_out", &self.padding_out)
                    .add("channels", &self.channels)
                    .optional()
            }
            fn num_params(&self) -> usize {
                burn::module::Module::num_params(self)
            }
        }
        impl<B: Backend> Clone for ConvTranspose1d<B> {
            fn clone(&self) -> Self {
                let weight = self.weight.clone();
                let bias = self.bias.clone();
                let stride = self.stride.clone();
                let kernel_size = self.kernel_size.clone();
                let dilation = self.dilation.clone();
                let groups = self.groups.clone();
                let padding = self.padding.clone();
                let padding_out = self.padding_out.clone();
                let channels = self.channels.clone();
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                    padding_out,
                    channels,
                }
            }
        }
        /// The record type for the module.
        pub struct ConvTranspose1dRecord<B: Backend> {
            /// The module record associative type.
            pub weight: <Param<Tensor<B, 3>> as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub bias: <Option<Param<Tensor<B, 1>>> as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub stride: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub kernel_size: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub dilation: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub groups: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub padding: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub padding_out: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub channels: <[usize; 2] as burn::module::Module<B>>::Record,
        }
        /// The record item type for the module.
        #[serde(crate = "burn::serde")]
        #[serde(
            bound = "< < Param < Tensor < B, 3 > > as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize +\nburn :: serde :: de :: DeserializeOwned, < < Option < Param < Tensor < B, 1 >\n> > as burn :: module :: Module < B > > :: Record as burn :: record :: Record\n< B >> :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de ::\nDeserializeOwned, < < usize as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize +\nburn :: serde :: de :: DeserializeOwned, < < usize as burn :: module :: Module\n< B > > :: Record as burn :: record :: Record < B >> :: Item < S > : burn ::\nserde :: Serialize + burn :: serde :: de :: DeserializeOwned, < < usize as\nburn :: module :: Module < B > > :: Record as burn :: record :: Record < B >>\n:: Item < S > : burn :: serde :: Serialize + burn :: serde :: de ::\nDeserializeOwned, < < usize as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize +\nburn :: serde :: de :: DeserializeOwned, < < usize as burn :: module :: Module\n< B > > :: Record as burn :: record :: Record < B >> :: Item < S > : burn ::\nserde :: Serialize + burn :: serde :: de :: DeserializeOwned, < < usize as\nburn :: module :: Module < B > > :: Record as burn :: record :: Record < B >>\n:: Item < S > : burn :: serde :: Serialize + burn :: serde :: de ::\nDeserializeOwned, < < [usize; 2] as burn :: module :: Module < B > > :: Record\nas burn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize\n+ burn :: serde :: de :: DeserializeOwned,"
        )]
        pub struct ConvTranspose1dRecordItem<
            B: Backend,
            S: burn::record::PrecisionSettings,
        > {
            /// Field to be serialized.
            pub weight: <<Param<
                Tensor<B, 3>,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub bias: <<Option<
                Param<Tensor<B, 1>>,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub stride: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub kernel_size: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub dilation: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub groups: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub padding: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub padding_out: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub channels: <<[usize; 2] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
        }
        #[doc(hidden)]
        #[allow(
            non_upper_case_globals,
            unused_attributes,
            unused_qualifications,
            clippy::absolute_paths,
        )]
        const _: () = {
            use burn::serde as _serde;
            #[automatically_derived]
            impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
            for ConvTranspose1dRecordItem<B, S>
            where
                <<Param<
                    Tensor<B, 3>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Option<
                    Param<Tensor<B, 1>>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            {
                fn serialize<__S>(
                    &self,
                    __serializer: __S,
                ) -> _serde::__private226::Result<__S::Ok, __S::Error>
                where
                    __S: _serde::Serializer,
                {
                    let mut __serde_state = _serde::Serializer::serialize_struct(
                        __serializer,
                        "ConvTranspose1dRecordItem",
                        false as usize + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "weight",
                        &self.weight,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "bias",
                        &self.bias,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "stride",
                        &self.stride,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "kernel_size",
                        &self.kernel_size,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "dilation",
                        &self.dilation,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "groups",
                        &self.groups,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "padding",
                        &self.padding,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "padding_out",
                        &self.padding_out,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "channels",
                        &self.channels,
                    )?;
                    _serde::ser::SerializeStruct::end(__serde_state)
                }
            }
        };
        #[doc(hidden)]
        #[allow(
            non_upper_case_globals,
            unused_attributes,
            unused_qualifications,
            clippy::absolute_paths,
        )]
        const _: () = {
            use burn::serde as _serde;
            #[automatically_derived]
            impl<
                'de,
                B: Backend,
                S: burn::record::PrecisionSettings,
            > _serde::Deserialize<'de> for ConvTranspose1dRecordItem<B, S>
            where
                <<Param<
                    Tensor<B, 3>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Option<
                    Param<Tensor<B, 1>>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            {
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private226::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    #[allow(non_camel_case_types)]
                    #[doc(hidden)]
                    enum __Field {
                        __field0,
                        __field1,
                        __field2,
                        __field3,
                        __field4,
                        __field5,
                        __field6,
                        __field7,
                        __field8,
                        __ignore,
                    }
                    #[doc(hidden)]
                    struct __FieldVisitor;
                    #[automatically_derived]
                    impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                        type Value = __Field;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private226::Formatter,
                        ) -> _serde::__private226::fmt::Result {
                            _serde::__private226::Formatter::write_str(
                                __formatter,
                                "field identifier",
                            )
                        }
                        fn visit_u64<__E>(
                            self,
                            __value: u64,
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                0u64 => _serde::__private226::Ok(__Field::__field0),
                                1u64 => _serde::__private226::Ok(__Field::__field1),
                                2u64 => _serde::__private226::Ok(__Field::__field2),
                                3u64 => _serde::__private226::Ok(__Field::__field3),
                                4u64 => _serde::__private226::Ok(__Field::__field4),
                                5u64 => _serde::__private226::Ok(__Field::__field5),
                                6u64 => _serde::__private226::Ok(__Field::__field6),
                                7u64 => _serde::__private226::Ok(__Field::__field7),
                                8u64 => _serde::__private226::Ok(__Field::__field8),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_str<__E>(
                            self,
                            __value: &str,
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                "weight" => _serde::__private226::Ok(__Field::__field0),
                                "bias" => _serde::__private226::Ok(__Field::__field1),
                                "stride" => _serde::__private226::Ok(__Field::__field2),
                                "kernel_size" => _serde::__private226::Ok(__Field::__field3),
                                "dilation" => _serde::__private226::Ok(__Field::__field4),
                                "groups" => _serde::__private226::Ok(__Field::__field5),
                                "padding" => _serde::__private226::Ok(__Field::__field6),
                                "padding_out" => _serde::__private226::Ok(__Field::__field7),
                                "channels" => _serde::__private226::Ok(__Field::__field8),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_bytes<__E>(
                            self,
                            __value: &[u8],
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                b"weight" => _serde::__private226::Ok(__Field::__field0),
                                b"bias" => _serde::__private226::Ok(__Field::__field1),
                                b"stride" => _serde::__private226::Ok(__Field::__field2),
                                b"kernel_size" => {
                                    _serde::__private226::Ok(__Field::__field3)
                                }
                                b"dilation" => _serde::__private226::Ok(__Field::__field4),
                                b"groups" => _serde::__private226::Ok(__Field::__field5),
                                b"padding" => _serde::__private226::Ok(__Field::__field6),
                                b"padding_out" => {
                                    _serde::__private226::Ok(__Field::__field7)
                                }
                                b"channels" => _serde::__private226::Ok(__Field::__field8),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                    }
                    #[automatically_derived]
                    impl<'de> _serde::Deserialize<'de> for __Field {
                        #[inline]
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private226::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            _serde::Deserializer::deserialize_identifier(
                                __deserializer,
                                __FieldVisitor,
                            )
                        }
                    }
                    #[doc(hidden)]
                    struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                    where
                        <<Param<
                            Tensor<B, 3>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Option<
                            Param<Tensor<B, 1>>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    {
                        marker: _serde::__private226::PhantomData<
                            ConvTranspose1dRecordItem<B, S>,
                        >,
                        lifetime: _serde::__private226::PhantomData<&'de ()>,
                    }
                    #[automatically_derived]
                    impl<
                        'de,
                        B: Backend,
                        S: burn::record::PrecisionSettings,
                    > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                    where
                        <<Param<
                            Tensor<B, 3>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Option<
                            Param<Tensor<B, 1>>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    {
                        type Value = ConvTranspose1dRecordItem<B, S>;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private226::Formatter,
                        ) -> _serde::__private226::fmt::Result {
                            _serde::__private226::Formatter::write_str(
                                __formatter,
                                "struct ConvTranspose1dRecordItem",
                            )
                        }
                        #[inline]
                        fn visit_seq<__A>(
                            self,
                            mut __seq: __A,
                        ) -> _serde::__private226::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::SeqAccess<'de>,
                        {
                            let __field0 = match _serde::de::SeqAccess::next_element::<
                                <<Param<
                                    Tensor<B, 3>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            0usize,
                                            &"struct ConvTranspose1dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field1 = match _serde::de::SeqAccess::next_element::<
                                <<Option<
                                    Param<Tensor<B, 1>>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            1usize,
                                            &"struct ConvTranspose1dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field2 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            2usize,
                                            &"struct ConvTranspose1dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field3 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            3usize,
                                            &"struct ConvTranspose1dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field4 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            4usize,
                                            &"struct ConvTranspose1dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field5 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            5usize,
                                            &"struct ConvTranspose1dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field6 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            6usize,
                                            &"struct ConvTranspose1dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field7 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            7usize,
                                            &"struct ConvTranspose1dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field8 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            8usize,
                                            &"struct ConvTranspose1dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            _serde::__private226::Ok(ConvTranspose1dRecordItem {
                                weight: __field0,
                                bias: __field1,
                                stride: __field2,
                                kernel_size: __field3,
                                dilation: __field4,
                                groups: __field5,
                                padding: __field6,
                                padding_out: __field7,
                                channels: __field8,
                            })
                        }
                        #[inline]
                        fn visit_map<__A>(
                            self,
                            mut __map: __A,
                        ) -> _serde::__private226::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::MapAccess<'de>,
                        {
                            let mut __field0: _serde::__private226::Option<
                                <<Param<
                                    Tensor<B, 3>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field1: _serde::__private226::Option<
                                <<Option<
                                    Param<Tensor<B, 1>>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field2: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field3: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field4: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field5: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field6: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field7: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field8: _serde::__private226::Option<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            while let _serde::__private226::Some(__key) = _serde::de::MapAccess::next_key::<
                                __Field,
                            >(&mut __map)? {
                                match __key {
                                    __Field::__field0 => {
                                        if _serde::__private226::Option::is_some(&__field0) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("weight"),
                                            );
                                        }
                                        __field0 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Param<
                                                    Tensor<B, 3>,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field1 => {
                                        if _serde::__private226::Option::is_some(&__field1) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("bias"),
                                            );
                                        }
                                        __field1 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Option<
                                                    Param<Tensor<B, 1>>,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field2 => {
                                        if _serde::__private226::Option::is_some(&__field2) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                            );
                                        }
                                        __field2 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field3 => {
                                        if _serde::__private226::Option::is_some(&__field3) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "kernel_size",
                                                ),
                                            );
                                        }
                                        __field3 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field4 => {
                                        if _serde::__private226::Option::is_some(&__field4) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "dilation",
                                                ),
                                            );
                                        }
                                        __field4 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field5 => {
                                        if _serde::__private226::Option::is_some(&__field5) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("groups"),
                                            );
                                        }
                                        __field5 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field6 => {
                                        if _serde::__private226::Option::is_some(&__field6) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "padding",
                                                ),
                                            );
                                        }
                                        __field6 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field7 => {
                                        if _serde::__private226::Option::is_some(&__field7) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "padding_out",
                                                ),
                                            );
                                        }
                                        __field7 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field8 => {
                                        if _serde::__private226::Option::is_some(&__field8) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "channels",
                                                ),
                                            );
                                        }
                                        __field8 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 2] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    _ => {
                                        let _ = _serde::de::MapAccess::next_value::<
                                            _serde::de::IgnoredAny,
                                        >(&mut __map)?;
                                    }
                                }
                            }
                            let __field0 = match __field0 {
                                _serde::__private226::Some(__field0) => __field0,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("weight")?
                                }
                            };
                            let __field1 = match __field1 {
                                _serde::__private226::Some(__field1) => __field1,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("bias")?
                                }
                            };
                            let __field2 = match __field2 {
                                _serde::__private226::Some(__field2) => __field2,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("stride")?
                                }
                            };
                            let __field3 = match __field3 {
                                _serde::__private226::Some(__field3) => __field3,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("kernel_size")?
                                }
                            };
                            let __field4 = match __field4 {
                                _serde::__private226::Some(__field4) => __field4,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("dilation")?
                                }
                            };
                            let __field5 = match __field5 {
                                _serde::__private226::Some(__field5) => __field5,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("groups")?
                                }
                            };
                            let __field6 = match __field6 {
                                _serde::__private226::Some(__field6) => __field6,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("padding")?
                                }
                            };
                            let __field7 = match __field7 {
                                _serde::__private226::Some(__field7) => __field7,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("padding_out")?
                                }
                            };
                            let __field8 = match __field8 {
                                _serde::__private226::Some(__field8) => __field8,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("channels")?
                                }
                            };
                            _serde::__private226::Ok(ConvTranspose1dRecordItem {
                                weight: __field0,
                                bias: __field1,
                                stride: __field2,
                                kernel_size: __field3,
                                dilation: __field4,
                                groups: __field5,
                                padding: __field6,
                                padding_out: __field7,
                                channels: __field8,
                            })
                        }
                    }
                    #[doc(hidden)]
                    const FIELDS: &'static [&'static str] = &[
                        "weight",
                        "bias",
                        "stride",
                        "kernel_size",
                        "dilation",
                        "groups",
                        "padding",
                        "padding_out",
                        "channels",
                    ];
                    _serde::Deserializer::deserialize_struct(
                        __deserializer,
                        "ConvTranspose1dRecordItem",
                        FIELDS,
                        __Visitor {
                            marker: _serde::__private226::PhantomData::<
                                ConvTranspose1dRecordItem<B, S>,
                            >,
                            lifetime: _serde::__private226::PhantomData,
                        },
                    )
                }
            }
        };
        impl<B: Backend, S: burn::record::PrecisionSettings> Clone
        for ConvTranspose1dRecordItem<B, S> {
            fn clone(&self) -> Self {
                Self {
                    weight: self.weight.clone(),
                    bias: self.bias.clone(),
                    stride: self.stride.clone(),
                    kernel_size: self.kernel_size.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                    padding_out: self.padding_out.clone(),
                    channels: self.channels.clone(),
                }
            }
        }
        impl<B: Backend> burn::record::Record<B> for ConvTranspose1dRecord<B> {
            type Item<S: burn::record::PrecisionSettings> = ConvTranspose1dRecordItem<
                B,
                S,
            >;
            fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
                ConvTranspose1dRecordItem {
                    weight: burn::record::Record::<B>::into_item::<S>(self.weight),
                    bias: burn::record::Record::<B>::into_item::<S>(self.bias),
                    stride: burn::record::Record::<B>::into_item::<S>(self.stride),
                    kernel_size: burn::record::Record::<
                        B,
                    >::into_item::<S>(self.kernel_size),
                    dilation: burn::record::Record::<B>::into_item::<S>(self.dilation),
                    groups: burn::record::Record::<B>::into_item::<S>(self.groups),
                    padding: burn::record::Record::<B>::into_item::<S>(self.padding),
                    padding_out: burn::record::Record::<
                        B,
                    >::into_item::<S>(self.padding_out),
                    channels: burn::record::Record::<B>::into_item::<S>(self.channels),
                }
            }
            fn from_item<S: burn::record::PrecisionSettings>(
                item: Self::Item<S>,
                device: &B::Device,
            ) -> Self {
                Self {
                    weight: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.weight, device),
                    bias: burn::record::Record::<B>::from_item::<S>(item.bias, device),
                    stride: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.stride, device),
                    kernel_size: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.kernel_size, device),
                    dilation: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.dilation, device),
                    groups: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.groups, device),
                    padding: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.padding, device),
                    padding_out: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.padding_out, device),
                    channels: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.channels, device),
                }
            }
        }
        #[automatically_derived]
        impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for ConvTranspose1d<B> {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                let names: &'static _ = &[
                    "weight",
                    "bias",
                    "stride",
                    "kernel_size",
                    "dilation",
                    "groups",
                    "padding",
                    "padding_out",
                    "channels",
                ];
                let values: &[&dyn ::core::fmt::Debug] = &[
                    &self.weight,
                    &self.bias,
                    &self.stride,
                    &self.kernel_size,
                    &self.dilation,
                    &self.groups,
                    &self.padding,
                    &self.padding_out,
                    &&self.channels,
                ];
                ::core::fmt::Formatter::debug_struct_fields_finish(
                    f,
                    "ConvTranspose1d",
                    names,
                    values,
                )
            }
        }
        impl<B: Backend> ModuleDisplay for ConvTranspose1d<B> {
            fn custom_settings(&self) -> Option<DisplaySettings> {
                DisplaySettings::new().with_new_line_after_attribute(false).optional()
            }
            fn custom_content(&self, content: Content) -> Option<Content> {
                content
                    .add(
                        "channels",
                        &::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("{0:?}", &self.channels))
                        }),
                    )
                    .add("stride", &self.stride)
                    .add("kernel_size", &self.kernel_size)
                    .add("dilation", &self.dilation)
                    .add("groups", &self.groups)
                    .add("padding", &self.padding)
                    .add("padding_out", &self.padding_out)
                    .optional()
            }
        }
        impl ConvTranspose1dConfig {
            /// Initialize a new [conv transpose 1d](ConvTranspose1d) module.
            pub fn init<B: Backend>(&self, device: &B::Device) -> ConvTranspose1d<B> {
                checks::checks_channels_div_groups(
                    self.channels[0],
                    self.channels[1],
                    self.groups,
                );
                let shape = [
                    self.channels[0],
                    self.channels[1] / self.groups,
                    self.kernel_size,
                ];
                let fan_in = self.channels[1] / self.groups * self.kernel_size;
                let weight = self
                    .initializer
                    .init_with(shape, Some(fan_in), None, device);
                let mut bias = None;
                if self.bias {
                    bias = Some(
                        self
                            .initializer
                            .init_with([self.channels[1]], Some(fan_in), None, device),
                    );
                }
                ConvTranspose1d {
                    weight,
                    bias,
                    stride: self.stride,
                    kernel_size: self.kernel_size,
                    dilation: self.dilation,
                    groups: self.groups,
                    padding: self.padding,
                    padding_out: self.padding_out,
                    channels: self.channels,
                }
            }
        }
        impl<B: Backend> ConvTranspose1d<B> {
            /// Applies the forward pass on the input tensor.
            ///
            /// See also [conv_transpose1d](burn::tensor::module::conv_transpose1d).
            ///
            /// # Shapes
            ///
            /// - input: `[batch_size, channels_in, length_in]`
            /// - output: `[batch_size, channels_out, length_out]`
            pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
                conv_transpose1d(
                    input,
                    self.weight.val(),
                    self.bias.as_ref().map(|bias| bias.val()),
                    ConvTransposeOptions::new(
                        [self.stride],
                        [self.padding],
                        [self.padding_out],
                        [self.dilation],
                        self.groups,
                    ),
                )
            }
        }
    }
    mod conv_transpose2d {
        use alloc::format;
        use burn_core as burn;
        use crate::conv::checks;
        use burn::config::Config;
        use burn::module::Content;
        use burn::module::DisplaySettings;
        use burn::module::Initializer;
        use burn::module::Module;
        use burn::module::ModuleDisplay;
        use burn::module::Param;
        use burn::tensor::Tensor;
        use burn::tensor::backend::Backend;
        use burn::tensor::module::conv_transpose2d;
        use burn::tensor::ops::ConvTransposeOptions;
        /// Configuration to create an [2D transposed convolution](ConvTranspose2d) layer
        /// using the [init function](ConvTranspose2dConfig::init).
        pub struct ConvTranspose2dConfig {
            /// The number of channels.
            pub channels: [usize; 2],
            /// The size of the kernel.
            pub kernel_size: [usize; 2],
            /// The stride of the convolution.
            #[config(default = "[1, 1]")]
            pub stride: [usize; 2],
            /// Spacing between kernel elements.
            #[config(default = "[1, 1]")]
            pub dilation: [usize; 2],
            /// Controls the connections between input and output channels.
            #[config(default = "1")]
            pub groups: usize,
            /// The padding configuration.
            #[config(default = "[0, 0]")]
            pub padding: [usize; 2],
            /// The padding output configuration.
            #[config(default = "[0, 0]")]
            pub padding_out: [usize; 2],
            /// If bias should be added to the output.
            #[config(default = true)]
            pub bias: bool,
            /// The type of function used to initialize neural network parameters
            #[config(
                default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
            )]
            pub initializer: Initializer,
        }
        impl burn::config::Config for ConvTranspose2dConfig {}
        impl ConvTranspose2dConfig {
            /// Create a new instance of the config.
            #[allow(clippy::too_many_arguments)]
            pub fn new(channels: [usize; 2], kernel_size: [usize; 2]) -> Self {
                Self {
                    channels: channels,
                    kernel_size: kernel_size,
                    stride: [1, 1],
                    dilation: [1, 1],
                    groups: 1,
                    padding: [0, 0],
                    padding_out: [0, 0],
                    bias: true,
                    initializer: Initializer::KaimingUniform {
                        gain: 1.0 / num_traits::Float::sqrt(3.0),
                        fan_out_only: false,
                    },
                }
            }
        }
        impl ConvTranspose2dConfig {
            /// The stride of the convolution.
            pub fn with_stride(mut self, stride: [usize; 2]) -> Self {
                self.stride = stride;
                self
            }
            /// Spacing between kernel elements.
            pub fn with_dilation(mut self, dilation: [usize; 2]) -> Self {
                self.dilation = dilation;
                self
            }
            /// Controls the connections between input and output channels.
            pub fn with_groups(mut self, groups: usize) -> Self {
                self.groups = groups;
                self
            }
            /// The padding configuration.
            pub fn with_padding(mut self, padding: [usize; 2]) -> Self {
                self.padding = padding;
                self
            }
            /// The padding output configuration.
            pub fn with_padding_out(mut self, padding_out: [usize; 2]) -> Self {
                self.padding_out = padding_out;
                self
            }
            /// If bias should be added to the output.
            pub fn with_bias(mut self, bias: bool) -> Self {
                self.bias = bias;
                self
            }
            /// The type of function used to initialize neural network parameters
            pub fn with_initializer(mut self, initializer: Initializer) -> Self {
                self.initializer = initializer;
                self
            }
        }
        impl burn::serde::Serialize for ConvTranspose2dConfig {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: burn::serde::Serializer,
            {
                #[serde(crate = "burn::serde")]
                struct ConvTranspose2dConfigSerde {
                    channels: [usize; 2],
                    kernel_size: [usize; 2],
                    stride: [usize; 2],
                    dilation: [usize; 2],
                    groups: usize,
                    padding: [usize; 2],
                    padding_out: [usize; 2],
                    bias: bool,
                    initializer: Initializer,
                }
                #[doc(hidden)]
                #[allow(
                    non_upper_case_globals,
                    unused_attributes,
                    unused_qualifications,
                    clippy::absolute_paths,
                )]
                const _: () = {
                    use burn::serde as _serde;
                    #[automatically_derived]
                    impl _serde::Serialize for ConvTranspose2dConfigSerde {
                        fn serialize<__S>(
                            &self,
                            __serializer: __S,
                        ) -> _serde::__private226::Result<__S::Ok, __S::Error>
                        where
                            __S: _serde::Serializer,
                        {
                            let mut __serde_state = _serde::Serializer::serialize_struct(
                                __serializer,
                                "ConvTranspose2dConfigSerde",
                                false as usize + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "channels",
                                &self.channels,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "kernel_size",
                                &self.kernel_size,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "stride",
                                &self.stride,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "dilation",
                                &self.dilation,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "groups",
                                &self.groups,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "padding",
                                &self.padding,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "padding_out",
                                &self.padding_out,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "bias",
                                &self.bias,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "initializer",
                                &self.initializer,
                            )?;
                            _serde::ser::SerializeStruct::end(__serde_state)
                        }
                    }
                };
                let serde_state = ConvTranspose2dConfigSerde {
                    channels: self.channels.clone(),
                    kernel_size: self.kernel_size.clone(),
                    stride: self.stride.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                    padding_out: self.padding_out.clone(),
                    bias: self.bias.clone(),
                    initializer: self.initializer.clone(),
                };
                serde_state.serialize(serializer)
            }
        }
        impl<'de> burn::serde::Deserialize<'de> for ConvTranspose2dConfig {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: burn::serde::Deserializer<'de>,
            {
                #[serde(crate = "burn::serde")]
                struct ConvTranspose2dConfigSerde {
                    channels: [usize; 2],
                    kernel_size: [usize; 2],
                    stride: [usize; 2],
                    dilation: [usize; 2],
                    groups: usize,
                    padding: [usize; 2],
                    padding_out: [usize; 2],
                    bias: bool,
                    initializer: Initializer,
                }
                #[doc(hidden)]
                #[allow(
                    non_upper_case_globals,
                    unused_attributes,
                    unused_qualifications,
                    clippy::absolute_paths,
                )]
                const _: () = {
                    use burn::serde as _serde;
                    #[automatically_derived]
                    impl<'de> _serde::Deserialize<'de> for ConvTranspose2dConfigSerde {
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private226::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            #[allow(non_camel_case_types)]
                            #[doc(hidden)]
                            enum __Field {
                                __field0,
                                __field1,
                                __field2,
                                __field3,
                                __field4,
                                __field5,
                                __field6,
                                __field7,
                                __field8,
                                __ignore,
                            }
                            #[doc(hidden)]
                            struct __FieldVisitor;
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                type Value = __Field;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private226::Formatter,
                                ) -> _serde::__private226::fmt::Result {
                                    _serde::__private226::Formatter::write_str(
                                        __formatter,
                                        "field identifier",
                                    )
                                }
                                fn visit_u64<__E>(
                                    self,
                                    __value: u64,
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        0u64 => _serde::__private226::Ok(__Field::__field0),
                                        1u64 => _serde::__private226::Ok(__Field::__field1),
                                        2u64 => _serde::__private226::Ok(__Field::__field2),
                                        3u64 => _serde::__private226::Ok(__Field::__field3),
                                        4u64 => _serde::__private226::Ok(__Field::__field4),
                                        5u64 => _serde::__private226::Ok(__Field::__field5),
                                        6u64 => _serde::__private226::Ok(__Field::__field6),
                                        7u64 => _serde::__private226::Ok(__Field::__field7),
                                        8u64 => _serde::__private226::Ok(__Field::__field8),
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_str<__E>(
                                    self,
                                    __value: &str,
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        "channels" => _serde::__private226::Ok(__Field::__field0),
                                        "kernel_size" => _serde::__private226::Ok(__Field::__field1),
                                        "stride" => _serde::__private226::Ok(__Field::__field2),
                                        "dilation" => _serde::__private226::Ok(__Field::__field3),
                                        "groups" => _serde::__private226::Ok(__Field::__field4),
                                        "padding" => _serde::__private226::Ok(__Field::__field5),
                                        "padding_out" => _serde::__private226::Ok(__Field::__field6),
                                        "bias" => _serde::__private226::Ok(__Field::__field7),
                                        "initializer" => _serde::__private226::Ok(__Field::__field8),
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_bytes<__E>(
                                    self,
                                    __value: &[u8],
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        b"channels" => _serde::__private226::Ok(__Field::__field0),
                                        b"kernel_size" => {
                                            _serde::__private226::Ok(__Field::__field1)
                                        }
                                        b"stride" => _serde::__private226::Ok(__Field::__field2),
                                        b"dilation" => _serde::__private226::Ok(__Field::__field3),
                                        b"groups" => _serde::__private226::Ok(__Field::__field4),
                                        b"padding" => _serde::__private226::Ok(__Field::__field5),
                                        b"padding_out" => {
                                            _serde::__private226::Ok(__Field::__field6)
                                        }
                                        b"bias" => _serde::__private226::Ok(__Field::__field7),
                                        b"initializer" => {
                                            _serde::__private226::Ok(__Field::__field8)
                                        }
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                            }
                            #[automatically_derived]
                            impl<'de> _serde::Deserialize<'de> for __Field {
                                #[inline]
                                fn deserialize<__D>(
                                    __deserializer: __D,
                                ) -> _serde::__private226::Result<Self, __D::Error>
                                where
                                    __D: _serde::Deserializer<'de>,
                                {
                                    _serde::Deserializer::deserialize_identifier(
                                        __deserializer,
                                        __FieldVisitor,
                                    )
                                }
                            }
                            #[doc(hidden)]
                            struct __Visitor<'de> {
                                marker: _serde::__private226::PhantomData<
                                    ConvTranspose2dConfigSerde,
                                >,
                                lifetime: _serde::__private226::PhantomData<&'de ()>,
                            }
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                type Value = ConvTranspose2dConfigSerde;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private226::Formatter,
                                ) -> _serde::__private226::fmt::Result {
                                    _serde::__private226::Formatter::write_str(
                                        __formatter,
                                        "struct ConvTranspose2dConfigSerde",
                                    )
                                }
                                #[inline]
                                fn visit_seq<__A>(
                                    self,
                                    mut __seq: __A,
                                ) -> _serde::__private226::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::SeqAccess<'de>,
                                {
                                    let __field0 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    0usize,
                                                    &"struct ConvTranspose2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field1 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    1usize,
                                                    &"struct ConvTranspose2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field2 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    2usize,
                                                    &"struct ConvTranspose2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field3 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    3usize,
                                                    &"struct ConvTranspose2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field4 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    4usize,
                                                    &"struct ConvTranspose2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field5 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    5usize,
                                                    &"struct ConvTranspose2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field6 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    6usize,
                                                    &"struct ConvTranspose2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field7 = match _serde::de::SeqAccess::next_element::<
                                        bool,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    7usize,
                                                    &"struct ConvTranspose2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field8 = match _serde::de::SeqAccess::next_element::<
                                        Initializer,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    8usize,
                                                    &"struct ConvTranspose2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    _serde::__private226::Ok(ConvTranspose2dConfigSerde {
                                        channels: __field0,
                                        kernel_size: __field1,
                                        stride: __field2,
                                        dilation: __field3,
                                        groups: __field4,
                                        padding: __field5,
                                        padding_out: __field6,
                                        bias: __field7,
                                        initializer: __field8,
                                    })
                                }
                                #[inline]
                                fn visit_map<__A>(
                                    self,
                                    mut __map: __A,
                                ) -> _serde::__private226::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::MapAccess<'de>,
                                {
                                    let mut __field0: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field1: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field2: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field3: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field4: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field5: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field6: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field7: _serde::__private226::Option<bool> = _serde::__private226::None;
                                    let mut __field8: _serde::__private226::Option<
                                        Initializer,
                                    > = _serde::__private226::None;
                                    while let _serde::__private226::Some(__key) = _serde::de::MapAccess::next_key::<
                                        __Field,
                                    >(&mut __map)? {
                                        match __key {
                                            __Field::__field0 => {
                                                if _serde::__private226::Option::is_some(&__field0) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "channels",
                                                        ),
                                                    );
                                                }
                                                __field0 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field1 => {
                                                if _serde::__private226::Option::is_some(&__field1) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "kernel_size",
                                                        ),
                                                    );
                                                }
                                                __field1 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field2 => {
                                                if _serde::__private226::Option::is_some(&__field2) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                    );
                                                }
                                                __field2 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field3 => {
                                                if _serde::__private226::Option::is_some(&__field3) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "dilation",
                                                        ),
                                                    );
                                                }
                                                __field3 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field4 => {
                                                if _serde::__private226::Option::is_some(&__field4) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("groups"),
                                                    );
                                                }
                                                __field4 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field5 => {
                                                if _serde::__private226::Option::is_some(&__field5) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "padding",
                                                        ),
                                                    );
                                                }
                                                __field5 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field6 => {
                                                if _serde::__private226::Option::is_some(&__field6) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "padding_out",
                                                        ),
                                                    );
                                                }
                                                __field6 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field7 => {
                                                if _serde::__private226::Option::is_some(&__field7) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("bias"),
                                                    );
                                                }
                                                __field7 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<bool>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field8 => {
                                                if _serde::__private226::Option::is_some(&__field8) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "initializer",
                                                        ),
                                                    );
                                                }
                                                __field8 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<
                                                        Initializer,
                                                    >(&mut __map)?,
                                                );
                                            }
                                            _ => {
                                                let _ = _serde::de::MapAccess::next_value::<
                                                    _serde::de::IgnoredAny,
                                                >(&mut __map)?;
                                            }
                                        }
                                    }
                                    let __field0 = match __field0 {
                                        _serde::__private226::Some(__field0) => __field0,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("channels")?
                                        }
                                    };
                                    let __field1 = match __field1 {
                                        _serde::__private226::Some(__field1) => __field1,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("kernel_size")?
                                        }
                                    };
                                    let __field2 = match __field2 {
                                        _serde::__private226::Some(__field2) => __field2,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("stride")?
                                        }
                                    };
                                    let __field3 = match __field3 {
                                        _serde::__private226::Some(__field3) => __field3,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("dilation")?
                                        }
                                    };
                                    let __field4 = match __field4 {
                                        _serde::__private226::Some(__field4) => __field4,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("groups")?
                                        }
                                    };
                                    let __field5 = match __field5 {
                                        _serde::__private226::Some(__field5) => __field5,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("padding")?
                                        }
                                    };
                                    let __field6 = match __field6 {
                                        _serde::__private226::Some(__field6) => __field6,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("padding_out")?
                                        }
                                    };
                                    let __field7 = match __field7 {
                                        _serde::__private226::Some(__field7) => __field7,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("bias")?
                                        }
                                    };
                                    let __field8 = match __field8 {
                                        _serde::__private226::Some(__field8) => __field8,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("initializer")?
                                        }
                                    };
                                    _serde::__private226::Ok(ConvTranspose2dConfigSerde {
                                        channels: __field0,
                                        kernel_size: __field1,
                                        stride: __field2,
                                        dilation: __field3,
                                        groups: __field4,
                                        padding: __field5,
                                        padding_out: __field6,
                                        bias: __field7,
                                        initializer: __field8,
                                    })
                                }
                            }
                            #[doc(hidden)]
                            const FIELDS: &'static [&'static str] = &[
                                "channels",
                                "kernel_size",
                                "stride",
                                "dilation",
                                "groups",
                                "padding",
                                "padding_out",
                                "bias",
                                "initializer",
                            ];
                            _serde::Deserializer::deserialize_struct(
                                __deserializer,
                                "ConvTranspose2dConfigSerde",
                                FIELDS,
                                __Visitor {
                                    marker: _serde::__private226::PhantomData::<
                                        ConvTranspose2dConfigSerde,
                                    >,
                                    lifetime: _serde::__private226::PhantomData,
                                },
                            )
                        }
                    }
                };
                let serde_state = ConvTranspose2dConfigSerde::deserialize(deserializer)?;
                Ok(ConvTranspose2dConfig {
                    channels: serde_state.channels,
                    kernel_size: serde_state.kernel_size,
                    stride: serde_state.stride,
                    dilation: serde_state.dilation,
                    groups: serde_state.groups,
                    padding: serde_state.padding,
                    padding_out: serde_state.padding_out,
                    bias: serde_state.bias,
                    initializer: serde_state.initializer,
                })
            }
        }
        impl Clone for ConvTranspose2dConfig {
            fn clone(&self) -> Self {
                Self {
                    channels: self.channels.clone(),
                    kernel_size: self.kernel_size.clone(),
                    stride: self.stride.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                    padding_out: self.padding_out.clone(),
                    bias: self.bias.clone(),
                    initializer: self.initializer.clone(),
                }
            }
        }
        impl core::fmt::Display for ConvTranspose2dConfig {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(&burn::config::config_to_json(self))
            }
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for ConvTranspose2dConfig {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                let names: &'static _ = &[
                    "channels",
                    "kernel_size",
                    "stride",
                    "dilation",
                    "groups",
                    "padding",
                    "padding_out",
                    "bias",
                    "initializer",
                ];
                let values: &[&dyn ::core::fmt::Debug] = &[
                    &self.channels,
                    &self.kernel_size,
                    &self.stride,
                    &self.dilation,
                    &self.groups,
                    &self.padding,
                    &self.padding_out,
                    &self.bias,
                    &&self.initializer,
                ];
                ::core::fmt::Formatter::debug_struct_fields_finish(
                    f,
                    "ConvTranspose2dConfig",
                    names,
                    values,
                )
            }
        }
        /// Applies a 2D transposed convolution over input tensors.
        #[module(custom_display)]
        pub struct ConvTranspose2d<B: Backend> {
            /// Tensor of shape `[channels_in, channels_out / groups, kernel_size_1, kernel_size_2]`
            pub weight: Param<Tensor<B, 4>>,
            /// Tensor of shape `[channels_out]`
            pub bias: Option<Param<Tensor<B, 1>>>,
            /// Stride of the convolution.
            pub stride: [usize; 2],
            /// Size of the kernel.
            pub kernel_size: [usize; 2],
            /// Spacing between kernel elements.
            pub dilation: [usize; 2],
            /// Controls the connections between input and output channels.
            pub groups: usize,
            /// Padding configuration.
            pub padding: [usize; 2],
            /// Padding output configuration.
            pub padding_out: [usize; 2],
            /// Number of channels.
            pub channels: [usize; 2],
        }
        impl<B: Backend> burn::module::Module<B> for ConvTranspose2d<B> {
            type Record = ConvTranspose2dRecord<B>;
            fn load_record(self, record: Self::Record) -> Self {
                Self {
                    weight: burn::module::Module::<
                        B,
                    >::load_record(self.weight, record.weight),
                    bias: burn::module::Module::<B>::load_record(self.bias, record.bias),
                    stride: burn::module::Module::<
                        B,
                    >::load_record(self.stride, record.stride),
                    kernel_size: burn::module::Module::<
                        B,
                    >::load_record(self.kernel_size, record.kernel_size),
                    dilation: burn::module::Module::<
                        B,
                    >::load_record(self.dilation, record.dilation),
                    groups: burn::module::Module::<
                        B,
                    >::load_record(self.groups, record.groups),
                    padding: burn::module::Module::<
                        B,
                    >::load_record(self.padding, record.padding),
                    padding_out: burn::module::Module::<
                        B,
                    >::load_record(self.padding_out, record.padding_out),
                    channels: burn::module::Module::<
                        B,
                    >::load_record(self.channels, record.channels),
                }
            }
            fn into_record(self) -> Self::Record {
                Self::Record {
                    weight: burn::module::Module::<B>::into_record(self.weight),
                    bias: burn::module::Module::<B>::into_record(self.bias),
                    stride: burn::module::Module::<B>::into_record(self.stride),
                    kernel_size: burn::module::Module::<
                        B,
                    >::into_record(self.kernel_size),
                    dilation: burn::module::Module::<B>::into_record(self.dilation),
                    groups: burn::module::Module::<B>::into_record(self.groups),
                    padding: burn::module::Module::<B>::into_record(self.padding),
                    padding_out: burn::module::Module::<
                        B,
                    >::into_record(self.padding_out),
                    channels: burn::module::Module::<B>::into_record(self.channels),
                }
            }
            fn num_params(&self) -> usize {
                let mut num_params = 0;
                num_params += burn::module::Module::<B>::num_params(&self.weight);
                num_params += burn::module::Module::<B>::num_params(&self.bias);
                num_params += burn::module::Module::<B>::num_params(&self.stride);
                num_params += burn::module::Module::<B>::num_params(&self.kernel_size);
                num_params += burn::module::Module::<B>::num_params(&self.dilation);
                num_params += burn::module::Module::<B>::num_params(&self.groups);
                num_params += burn::module::Module::<B>::num_params(&self.padding);
                num_params += burn::module::Module::<B>::num_params(&self.padding_out);
                num_params += burn::module::Module::<B>::num_params(&self.channels);
                num_params
            }
            fn visit<Visitor: burn::module::ModuleVisitor<B>>(
                &self,
                visitor: &mut Visitor,
            ) {
                visitor.enter_module("weight", "ConvTranspose2d");
                burn::module::Module::visit(&self.weight, visitor);
                visitor.exit_module("weight", "ConvTranspose2d");
                visitor.enter_module("bias", "ConvTranspose2d");
                burn::module::Module::visit(&self.bias, visitor);
                visitor.exit_module("bias", "ConvTranspose2d");
                visitor.enter_module("stride", "ConvTranspose2d");
                burn::module::Module::visit(&self.stride, visitor);
                visitor.exit_module("stride", "ConvTranspose2d");
                visitor.enter_module("kernel_size", "ConvTranspose2d");
                burn::module::Module::visit(&self.kernel_size, visitor);
                visitor.exit_module("kernel_size", "ConvTranspose2d");
                visitor.enter_module("dilation", "ConvTranspose2d");
                burn::module::Module::visit(&self.dilation, visitor);
                visitor.exit_module("dilation", "ConvTranspose2d");
                visitor.enter_module("groups", "ConvTranspose2d");
                burn::module::Module::visit(&self.groups, visitor);
                visitor.exit_module("groups", "ConvTranspose2d");
                visitor.enter_module("padding", "ConvTranspose2d");
                burn::module::Module::visit(&self.padding, visitor);
                visitor.exit_module("padding", "ConvTranspose2d");
                visitor.enter_module("padding_out", "ConvTranspose2d");
                burn::module::Module::visit(&self.padding_out, visitor);
                visitor.exit_module("padding_out", "ConvTranspose2d");
                visitor.enter_module("channels", "ConvTranspose2d");
                burn::module::Module::visit(&self.channels, visitor);
                visitor.exit_module("channels", "ConvTranspose2d");
            }
            fn map<Mapper: burn::module::ModuleMapper<B>>(
                self,
                mapper: &mut Mapper,
            ) -> Self {
                mapper.enter_module("weight", "ConvTranspose2d");
                let weight = burn::module::Module::<B>::map(self.weight, mapper);
                mapper.exit_module("weight", "ConvTranspose2d");
                mapper.enter_module("bias", "ConvTranspose2d");
                let bias = burn::module::Module::<B>::map(self.bias, mapper);
                mapper.exit_module("bias", "ConvTranspose2d");
                mapper.enter_module("stride", "ConvTranspose2d");
                let stride = burn::module::Module::<B>::map(self.stride, mapper);
                mapper.exit_module("stride", "ConvTranspose2d");
                mapper.enter_module("kernel_size", "ConvTranspose2d");
                let kernel_size = burn::module::Module::<
                    B,
                >::map(self.kernel_size, mapper);
                mapper.exit_module("kernel_size", "ConvTranspose2d");
                mapper.enter_module("dilation", "ConvTranspose2d");
                let dilation = burn::module::Module::<B>::map(self.dilation, mapper);
                mapper.exit_module("dilation", "ConvTranspose2d");
                mapper.enter_module("groups", "ConvTranspose2d");
                let groups = burn::module::Module::<B>::map(self.groups, mapper);
                mapper.exit_module("groups", "ConvTranspose2d");
                mapper.enter_module("padding", "ConvTranspose2d");
                let padding = burn::module::Module::<B>::map(self.padding, mapper);
                mapper.exit_module("padding", "ConvTranspose2d");
                mapper.enter_module("padding_out", "ConvTranspose2d");
                let padding_out = burn::module::Module::<
                    B,
                >::map(self.padding_out, mapper);
                mapper.exit_module("padding_out", "ConvTranspose2d");
                mapper.enter_module("channels", "ConvTranspose2d");
                let channels = burn::module::Module::<B>::map(self.channels, mapper);
                mapper.exit_module("channels", "ConvTranspose2d");
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                    padding_out,
                    channels,
                }
            }
            fn collect_devices(
                &self,
                devices: burn::module::Devices<B>,
            ) -> burn::module::Devices<B> {
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.weight, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.bias, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.stride, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.kernel_size, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.dilation, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.groups, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.padding, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.padding_out, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.channels, devices);
                devices
            }
            fn to_device(self, device: &B::Device) -> Self {
                let weight = burn::module::Module::<B>::to_device(self.weight, device);
                let bias = burn::module::Module::<B>::to_device(self.bias, device);
                let stride = burn::module::Module::<B>::to_device(self.stride, device);
                let kernel_size = burn::module::Module::<
                    B,
                >::to_device(self.kernel_size, device);
                let dilation = burn::module::Module::<
                    B,
                >::to_device(self.dilation, device);
                let groups = burn::module::Module::<B>::to_device(self.groups, device);
                let padding = burn::module::Module::<B>::to_device(self.padding, device);
                let padding_out = burn::module::Module::<
                    B,
                >::to_device(self.padding_out, device);
                let channels = burn::module::Module::<
                    B,
                >::to_device(self.channels, device);
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                    padding_out,
                    channels,
                }
            }
            fn fork(self, device: &B::Device) -> Self {
                let weight = burn::module::Module::<B>::fork(self.weight, device);
                let bias = burn::module::Module::<B>::fork(self.bias, device);
                let stride = burn::module::Module::<B>::fork(self.stride, device);
                let kernel_size = burn::module::Module::<
                    B,
                >::fork(self.kernel_size, device);
                let dilation = burn::module::Module::<B>::fork(self.dilation, device);
                let groups = burn::module::Module::<B>::fork(self.groups, device);
                let padding = burn::module::Module::<B>::fork(self.padding, device);
                let padding_out = burn::module::Module::<
                    B,
                >::fork(self.padding_out, device);
                let channels = burn::module::Module::<B>::fork(self.channels, device);
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                    padding_out,
                    channels,
                }
            }
        }
        impl<B: Backend> burn::module::AutodiffModule<B> for ConvTranspose2d<B>
        where
            B: burn::tensor::backend::AutodiffBackend,
            <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
        {
            type InnerModule = ConvTranspose2d<B::InnerBackend>;
            fn valid(&self) -> Self::InnerModule {
                let weight = burn::module::AutodiffModule::<B>::valid(&self.weight);
                let bias = burn::module::AutodiffModule::<B>::valid(&self.bias);
                let stride = burn::module::AutodiffModule::<B>::valid(&self.stride);
                let kernel_size = burn::module::AutodiffModule::<
                    B,
                >::valid(&self.kernel_size);
                let dilation = burn::module::AutodiffModule::<B>::valid(&self.dilation);
                let groups = burn::module::AutodiffModule::<B>::valid(&self.groups);
                let padding = burn::module::AutodiffModule::<B>::valid(&self.padding);
                let padding_out = burn::module::AutodiffModule::<
                    B,
                >::valid(&self.padding_out);
                let channels = burn::module::AutodiffModule::<B>::valid(&self.channels);
                Self::InnerModule {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                    padding_out,
                    channels,
                }
            }
        }
        impl<B: Backend> core::fmt::Display for ConvTranspose2d<B> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let formatted = burn::module::ModuleDisplay::format(
                    self,
                    Default::default(),
                );
                f.write_fmt(format_args!("{0}", formatted))
            }
        }
        impl<B: Backend> burn::module::ModuleDisplayDefault for ConvTranspose2d<B> {
            fn content(
                &self,
                mut content: burn::module::Content,
            ) -> Option<burn::module::Content> {
                content
                    .set_top_level_type(&"ConvTranspose2d")
                    .add("weight", &self.weight)
                    .add("bias", &self.bias)
                    .add("stride", &self.stride)
                    .add("kernel_size", &self.kernel_size)
                    .add("dilation", &self.dilation)
                    .add("groups", &self.groups)
                    .add("padding", &self.padding)
                    .add("padding_out", &self.padding_out)
                    .add("channels", &self.channels)
                    .optional()
            }
            fn num_params(&self) -> usize {
                burn::module::Module::num_params(self)
            }
        }
        impl<B: Backend> Clone for ConvTranspose2d<B> {
            fn clone(&self) -> Self {
                let weight = self.weight.clone();
                let bias = self.bias.clone();
                let stride = self.stride.clone();
                let kernel_size = self.kernel_size.clone();
                let dilation = self.dilation.clone();
                let groups = self.groups.clone();
                let padding = self.padding.clone();
                let padding_out = self.padding_out.clone();
                let channels = self.channels.clone();
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                    padding_out,
                    channels,
                }
            }
        }
        /// The record type for the module.
        pub struct ConvTranspose2dRecord<B: Backend> {
            /// The module record associative type.
            pub weight: <Param<Tensor<B, 4>> as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub bias: <Option<Param<Tensor<B, 1>>> as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub stride: <[usize; 2] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub kernel_size: <[usize; 2] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub dilation: <[usize; 2] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub groups: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub padding: <[usize; 2] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub padding_out: <[usize; 2] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub channels: <[usize; 2] as burn::module::Module<B>>::Record,
        }
        /// The record item type for the module.
        #[serde(crate = "burn::serde")]
        #[serde(
            bound = "< < Param < Tensor < B, 4 > > as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize +\nburn :: serde :: de :: DeserializeOwned, < < Option < Param < Tensor < B, 1 >\n> > as burn :: module :: Module < B > > :: Record as burn :: record :: Record\n< B >> :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de ::\nDeserializeOwned, < < [usize; 2] as burn :: module :: Module < B > > :: Record\nas burn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize\n+ burn :: serde :: de :: DeserializeOwned, < < [usize; 2] as burn :: module ::\nModule < B > > :: Record as burn :: record :: Record < B >> :: Item < S > :\nburn :: serde :: Serialize + burn :: serde :: de :: DeserializeOwned, < <\n[usize; 2] as burn :: module :: Module < B > > :: Record as burn :: record ::\nRecord < B >> :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de\n:: DeserializeOwned, < < usize as burn :: module :: Module < B > > :: Record\nas burn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize\n+ burn :: serde :: de :: DeserializeOwned, < < [usize; 2] as burn :: module ::\nModule < B > > :: Record as burn :: record :: Record < B >> :: Item < S > :\nburn :: serde :: Serialize + burn :: serde :: de :: DeserializeOwned, < <\n[usize; 2] as burn :: module :: Module < B > > :: Record as burn :: record ::\nRecord < B >> :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de\n:: DeserializeOwned, < < [usize; 2] as burn :: module :: Module < B > > ::\nRecord as burn :: record :: Record < B >> :: Item < S > : burn :: serde ::\nSerialize + burn :: serde :: de :: DeserializeOwned,"
        )]
        pub struct ConvTranspose2dRecordItem<
            B: Backend,
            S: burn::record::PrecisionSettings,
        > {
            /// Field to be serialized.
            pub weight: <<Param<
                Tensor<B, 4>,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub bias: <<Option<
                Param<Tensor<B, 1>>,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub stride: <<[usize; 2] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub kernel_size: <<[usize; 2] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub dilation: <<[usize; 2] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub groups: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub padding: <<[usize; 2] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub padding_out: <<[usize; 2] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub channels: <<[usize; 2] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
        }
        #[doc(hidden)]
        #[allow(
            non_upper_case_globals,
            unused_attributes,
            unused_qualifications,
            clippy::absolute_paths,
        )]
        const _: () = {
            use burn::serde as _serde;
            #[automatically_derived]
            impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
            for ConvTranspose2dRecordItem<B, S>
            where
                <<Param<
                    Tensor<B, 4>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Option<
                    Param<Tensor<B, 1>>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            {
                fn serialize<__S>(
                    &self,
                    __serializer: __S,
                ) -> _serde::__private226::Result<__S::Ok, __S::Error>
                where
                    __S: _serde::Serializer,
                {
                    let mut __serde_state = _serde::Serializer::serialize_struct(
                        __serializer,
                        "ConvTranspose2dRecordItem",
                        false as usize + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "weight",
                        &self.weight,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "bias",
                        &self.bias,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "stride",
                        &self.stride,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "kernel_size",
                        &self.kernel_size,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "dilation",
                        &self.dilation,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "groups",
                        &self.groups,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "padding",
                        &self.padding,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "padding_out",
                        &self.padding_out,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "channels",
                        &self.channels,
                    )?;
                    _serde::ser::SerializeStruct::end(__serde_state)
                }
            }
        };
        #[doc(hidden)]
        #[allow(
            non_upper_case_globals,
            unused_attributes,
            unused_qualifications,
            clippy::absolute_paths,
        )]
        const _: () = {
            use burn::serde as _serde;
            #[automatically_derived]
            impl<
                'de,
                B: Backend,
                S: burn::record::PrecisionSettings,
            > _serde::Deserialize<'de> for ConvTranspose2dRecordItem<B, S>
            where
                <<Param<
                    Tensor<B, 4>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Option<
                    Param<Tensor<B, 1>>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            {
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private226::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    #[allow(non_camel_case_types)]
                    #[doc(hidden)]
                    enum __Field {
                        __field0,
                        __field1,
                        __field2,
                        __field3,
                        __field4,
                        __field5,
                        __field6,
                        __field7,
                        __field8,
                        __ignore,
                    }
                    #[doc(hidden)]
                    struct __FieldVisitor;
                    #[automatically_derived]
                    impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                        type Value = __Field;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private226::Formatter,
                        ) -> _serde::__private226::fmt::Result {
                            _serde::__private226::Formatter::write_str(
                                __formatter,
                                "field identifier",
                            )
                        }
                        fn visit_u64<__E>(
                            self,
                            __value: u64,
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                0u64 => _serde::__private226::Ok(__Field::__field0),
                                1u64 => _serde::__private226::Ok(__Field::__field1),
                                2u64 => _serde::__private226::Ok(__Field::__field2),
                                3u64 => _serde::__private226::Ok(__Field::__field3),
                                4u64 => _serde::__private226::Ok(__Field::__field4),
                                5u64 => _serde::__private226::Ok(__Field::__field5),
                                6u64 => _serde::__private226::Ok(__Field::__field6),
                                7u64 => _serde::__private226::Ok(__Field::__field7),
                                8u64 => _serde::__private226::Ok(__Field::__field8),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_str<__E>(
                            self,
                            __value: &str,
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                "weight" => _serde::__private226::Ok(__Field::__field0),
                                "bias" => _serde::__private226::Ok(__Field::__field1),
                                "stride" => _serde::__private226::Ok(__Field::__field2),
                                "kernel_size" => _serde::__private226::Ok(__Field::__field3),
                                "dilation" => _serde::__private226::Ok(__Field::__field4),
                                "groups" => _serde::__private226::Ok(__Field::__field5),
                                "padding" => _serde::__private226::Ok(__Field::__field6),
                                "padding_out" => _serde::__private226::Ok(__Field::__field7),
                                "channels" => _serde::__private226::Ok(__Field::__field8),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_bytes<__E>(
                            self,
                            __value: &[u8],
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                b"weight" => _serde::__private226::Ok(__Field::__field0),
                                b"bias" => _serde::__private226::Ok(__Field::__field1),
                                b"stride" => _serde::__private226::Ok(__Field::__field2),
                                b"kernel_size" => {
                                    _serde::__private226::Ok(__Field::__field3)
                                }
                                b"dilation" => _serde::__private226::Ok(__Field::__field4),
                                b"groups" => _serde::__private226::Ok(__Field::__field5),
                                b"padding" => _serde::__private226::Ok(__Field::__field6),
                                b"padding_out" => {
                                    _serde::__private226::Ok(__Field::__field7)
                                }
                                b"channels" => _serde::__private226::Ok(__Field::__field8),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                    }
                    #[automatically_derived]
                    impl<'de> _serde::Deserialize<'de> for __Field {
                        #[inline]
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private226::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            _serde::Deserializer::deserialize_identifier(
                                __deserializer,
                                __FieldVisitor,
                            )
                        }
                    }
                    #[doc(hidden)]
                    struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                    where
                        <<Param<
                            Tensor<B, 4>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Option<
                            Param<Tensor<B, 1>>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    {
                        marker: _serde::__private226::PhantomData<
                            ConvTranspose2dRecordItem<B, S>,
                        >,
                        lifetime: _serde::__private226::PhantomData<&'de ()>,
                    }
                    #[automatically_derived]
                    impl<
                        'de,
                        B: Backend,
                        S: burn::record::PrecisionSettings,
                    > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                    where
                        <<Param<
                            Tensor<B, 4>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Option<
                            Param<Tensor<B, 1>>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    {
                        type Value = ConvTranspose2dRecordItem<B, S>;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private226::Formatter,
                        ) -> _serde::__private226::fmt::Result {
                            _serde::__private226::Formatter::write_str(
                                __formatter,
                                "struct ConvTranspose2dRecordItem",
                            )
                        }
                        #[inline]
                        fn visit_seq<__A>(
                            self,
                            mut __seq: __A,
                        ) -> _serde::__private226::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::SeqAccess<'de>,
                        {
                            let __field0 = match _serde::de::SeqAccess::next_element::<
                                <<Param<
                                    Tensor<B, 4>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            0usize,
                                            &"struct ConvTranspose2dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field1 = match _serde::de::SeqAccess::next_element::<
                                <<Option<
                                    Param<Tensor<B, 1>>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            1usize,
                                            &"struct ConvTranspose2dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field2 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            2usize,
                                            &"struct ConvTranspose2dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field3 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            3usize,
                                            &"struct ConvTranspose2dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field4 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            4usize,
                                            &"struct ConvTranspose2dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field5 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            5usize,
                                            &"struct ConvTranspose2dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field6 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            6usize,
                                            &"struct ConvTranspose2dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field7 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            7usize,
                                            &"struct ConvTranspose2dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field8 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            8usize,
                                            &"struct ConvTranspose2dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            _serde::__private226::Ok(ConvTranspose2dRecordItem {
                                weight: __field0,
                                bias: __field1,
                                stride: __field2,
                                kernel_size: __field3,
                                dilation: __field4,
                                groups: __field5,
                                padding: __field6,
                                padding_out: __field7,
                                channels: __field8,
                            })
                        }
                        #[inline]
                        fn visit_map<__A>(
                            self,
                            mut __map: __A,
                        ) -> _serde::__private226::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::MapAccess<'de>,
                        {
                            let mut __field0: _serde::__private226::Option<
                                <<Param<
                                    Tensor<B, 4>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field1: _serde::__private226::Option<
                                <<Option<
                                    Param<Tensor<B, 1>>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field2: _serde::__private226::Option<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field3: _serde::__private226::Option<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field4: _serde::__private226::Option<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field5: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field6: _serde::__private226::Option<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field7: _serde::__private226::Option<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field8: _serde::__private226::Option<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            while let _serde::__private226::Some(__key) = _serde::de::MapAccess::next_key::<
                                __Field,
                            >(&mut __map)? {
                                match __key {
                                    __Field::__field0 => {
                                        if _serde::__private226::Option::is_some(&__field0) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("weight"),
                                            );
                                        }
                                        __field0 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Param<
                                                    Tensor<B, 4>,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field1 => {
                                        if _serde::__private226::Option::is_some(&__field1) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("bias"),
                                            );
                                        }
                                        __field1 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Option<
                                                    Param<Tensor<B, 1>>,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field2 => {
                                        if _serde::__private226::Option::is_some(&__field2) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                            );
                                        }
                                        __field2 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 2] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field3 => {
                                        if _serde::__private226::Option::is_some(&__field3) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "kernel_size",
                                                ),
                                            );
                                        }
                                        __field3 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 2] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field4 => {
                                        if _serde::__private226::Option::is_some(&__field4) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "dilation",
                                                ),
                                            );
                                        }
                                        __field4 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 2] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field5 => {
                                        if _serde::__private226::Option::is_some(&__field5) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("groups"),
                                            );
                                        }
                                        __field5 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field6 => {
                                        if _serde::__private226::Option::is_some(&__field6) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "padding",
                                                ),
                                            );
                                        }
                                        __field6 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 2] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field7 => {
                                        if _serde::__private226::Option::is_some(&__field7) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "padding_out",
                                                ),
                                            );
                                        }
                                        __field7 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 2] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field8 => {
                                        if _serde::__private226::Option::is_some(&__field8) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "channels",
                                                ),
                                            );
                                        }
                                        __field8 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 2] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    _ => {
                                        let _ = _serde::de::MapAccess::next_value::<
                                            _serde::de::IgnoredAny,
                                        >(&mut __map)?;
                                    }
                                }
                            }
                            let __field0 = match __field0 {
                                _serde::__private226::Some(__field0) => __field0,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("weight")?
                                }
                            };
                            let __field1 = match __field1 {
                                _serde::__private226::Some(__field1) => __field1,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("bias")?
                                }
                            };
                            let __field2 = match __field2 {
                                _serde::__private226::Some(__field2) => __field2,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("stride")?
                                }
                            };
                            let __field3 = match __field3 {
                                _serde::__private226::Some(__field3) => __field3,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("kernel_size")?
                                }
                            };
                            let __field4 = match __field4 {
                                _serde::__private226::Some(__field4) => __field4,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("dilation")?
                                }
                            };
                            let __field5 = match __field5 {
                                _serde::__private226::Some(__field5) => __field5,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("groups")?
                                }
                            };
                            let __field6 = match __field6 {
                                _serde::__private226::Some(__field6) => __field6,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("padding")?
                                }
                            };
                            let __field7 = match __field7 {
                                _serde::__private226::Some(__field7) => __field7,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("padding_out")?
                                }
                            };
                            let __field8 = match __field8 {
                                _serde::__private226::Some(__field8) => __field8,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("channels")?
                                }
                            };
                            _serde::__private226::Ok(ConvTranspose2dRecordItem {
                                weight: __field0,
                                bias: __field1,
                                stride: __field2,
                                kernel_size: __field3,
                                dilation: __field4,
                                groups: __field5,
                                padding: __field6,
                                padding_out: __field7,
                                channels: __field8,
                            })
                        }
                    }
                    #[doc(hidden)]
                    const FIELDS: &'static [&'static str] = &[
                        "weight",
                        "bias",
                        "stride",
                        "kernel_size",
                        "dilation",
                        "groups",
                        "padding",
                        "padding_out",
                        "channels",
                    ];
                    _serde::Deserializer::deserialize_struct(
                        __deserializer,
                        "ConvTranspose2dRecordItem",
                        FIELDS,
                        __Visitor {
                            marker: _serde::__private226::PhantomData::<
                                ConvTranspose2dRecordItem<B, S>,
                            >,
                            lifetime: _serde::__private226::PhantomData,
                        },
                    )
                }
            }
        };
        impl<B: Backend, S: burn::record::PrecisionSettings> Clone
        for ConvTranspose2dRecordItem<B, S> {
            fn clone(&self) -> Self {
                Self {
                    weight: self.weight.clone(),
                    bias: self.bias.clone(),
                    stride: self.stride.clone(),
                    kernel_size: self.kernel_size.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                    padding_out: self.padding_out.clone(),
                    channels: self.channels.clone(),
                }
            }
        }
        impl<B: Backend> burn::record::Record<B> for ConvTranspose2dRecord<B> {
            type Item<S: burn::record::PrecisionSettings> = ConvTranspose2dRecordItem<
                B,
                S,
            >;
            fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
                ConvTranspose2dRecordItem {
                    weight: burn::record::Record::<B>::into_item::<S>(self.weight),
                    bias: burn::record::Record::<B>::into_item::<S>(self.bias),
                    stride: burn::record::Record::<B>::into_item::<S>(self.stride),
                    kernel_size: burn::record::Record::<
                        B,
                    >::into_item::<S>(self.kernel_size),
                    dilation: burn::record::Record::<B>::into_item::<S>(self.dilation),
                    groups: burn::record::Record::<B>::into_item::<S>(self.groups),
                    padding: burn::record::Record::<B>::into_item::<S>(self.padding),
                    padding_out: burn::record::Record::<
                        B,
                    >::into_item::<S>(self.padding_out),
                    channels: burn::record::Record::<B>::into_item::<S>(self.channels),
                }
            }
            fn from_item<S: burn::record::PrecisionSettings>(
                item: Self::Item<S>,
                device: &B::Device,
            ) -> Self {
                Self {
                    weight: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.weight, device),
                    bias: burn::record::Record::<B>::from_item::<S>(item.bias, device),
                    stride: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.stride, device),
                    kernel_size: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.kernel_size, device),
                    dilation: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.dilation, device),
                    groups: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.groups, device),
                    padding: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.padding, device),
                    padding_out: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.padding_out, device),
                    channels: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.channels, device),
                }
            }
        }
        #[automatically_derived]
        impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for ConvTranspose2d<B> {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                let names: &'static _ = &[
                    "weight",
                    "bias",
                    "stride",
                    "kernel_size",
                    "dilation",
                    "groups",
                    "padding",
                    "padding_out",
                    "channels",
                ];
                let values: &[&dyn ::core::fmt::Debug] = &[
                    &self.weight,
                    &self.bias,
                    &self.stride,
                    &self.kernel_size,
                    &self.dilation,
                    &self.groups,
                    &self.padding,
                    &self.padding_out,
                    &&self.channels,
                ];
                ::core::fmt::Formatter::debug_struct_fields_finish(
                    f,
                    "ConvTranspose2d",
                    names,
                    values,
                )
            }
        }
        impl<B: Backend> ModuleDisplay for ConvTranspose2d<B> {
            fn custom_settings(&self) -> Option<DisplaySettings> {
                DisplaySettings::new().with_new_line_after_attribute(false).optional()
            }
            fn custom_content(&self, content: Content) -> Option<Content> {
                content
                    .add(
                        "channels",
                        &::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("{0:?}", &self.channels))
                        }),
                    )
                    .add(
                        "stride",
                        &::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("{0:?}", &self.stride))
                        }),
                    )
                    .add(
                        "kernel_size",
                        &::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!("{0:?}", &self.kernel_size),
                            )
                        }),
                    )
                    .add(
                        "dilation",
                        &::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("{0:?}", &self.dilation))
                        }),
                    )
                    .add("groups", &self.groups)
                    .add(
                        "padding",
                        &::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("{0:?}", &self.padding))
                        }),
                    )
                    .add(
                        "padding_out",
                        &::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!("{0:?}", &self.padding_out),
                            )
                        }),
                    )
                    .optional()
            }
        }
        impl ConvTranspose2dConfig {
            /// Initialize a new [conv transpose 2d](ConvTranspose2d) module.
            pub fn init<B: Backend>(&self, device: &B::Device) -> ConvTranspose2d<B> {
                checks::checks_channels_div_groups(
                    self.channels[0],
                    self.channels[1],
                    self.groups,
                );
                let shape = [
                    self.channels[0],
                    self.channels[1] / self.groups,
                    self.kernel_size[0],
                    self.kernel_size[1],
                ];
                let fan_in = self.channels[1] / self.groups
                    * self.kernel_size.iter().product::<usize>();
                let weight = self
                    .initializer
                    .init_with(shape, Some(fan_in), None, device);
                let mut bias = None;
                if self.bias {
                    bias = Some(
                        self
                            .initializer
                            .init_with([self.channels[1]], Some(fan_in), None, device),
                    );
                }
                ConvTranspose2d {
                    weight,
                    bias,
                    stride: self.stride,
                    kernel_size: self.kernel_size,
                    dilation: self.dilation,
                    groups: self.groups,
                    padding: self.padding,
                    padding_out: self.padding_out,
                    channels: self.channels,
                }
            }
        }
        impl<B: Backend> ConvTranspose2d<B> {
            /// Applies the forward pass on the input tensor.
            ///
            /// See also [conv_transpose2d](burn::tensor::module::conv_transpose2d).
            ///
            /// # Shapes
            ///
            /// - input: `[batch_size, channels_in, height_in, width_in]`
            /// - output: `[batch_size, channels_out, height_out, width_out]`
            pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                conv_transpose2d(
                    input,
                    self.weight.val(),
                    self.bias.as_ref().map(|bias| bias.val()),
                    ConvTransposeOptions::new(
                        self.stride,
                        self.padding,
                        self.padding_out,
                        self.dilation,
                        self.groups,
                    ),
                )
            }
        }
    }
    mod conv_transpose3d {
        use alloc::format;
        use burn_core as burn;
        use crate::conv::checks;
        use burn::config::Config;
        use burn::module::Content;
        use burn::module::DisplaySettings;
        use burn::module::Initializer;
        use burn::module::Module;
        use burn::module::ModuleDisplay;
        use burn::module::Param;
        use burn::tensor::Tensor;
        use burn::tensor::backend::Backend;
        use burn::tensor::module::conv_transpose3d;
        use burn::tensor::ops::ConvTransposeOptions;
        /// Configuration to create an [3D transposed convolution](ConvTranspose3d) layer
        /// using the [init function](ConvTranspose3dConfig::init).
        pub struct ConvTranspose3dConfig {
            /// The number of channels.
            pub channels: [usize; 2],
            /// The size of the kernel.
            pub kernel_size: [usize; 3],
            /// The stride of the convolution.
            #[config(default = "[1, 1, 1]")]
            pub stride: [usize; 3],
            /// Spacing between kernel elements.
            #[config(default = "[1, 1, 1]")]
            pub dilation: [usize; 3],
            /// Controls the connections between input and output channels.
            #[config(default = "1")]
            pub groups: usize,
            /// The padding configuration.
            #[config(default = "[0, 0, 0]")]
            pub padding: [usize; 3],
            /// The padding output configuration.
            #[config(default = "[0, 0, 0]")]
            pub padding_out: [usize; 3],
            /// If bias should be added to the output.
            #[config(default = true)]
            pub bias: bool,
            /// The type of function used to initialize neural network parameters
            #[config(
                default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
            )]
            pub initializer: Initializer,
        }
        impl burn::config::Config for ConvTranspose3dConfig {}
        impl ConvTranspose3dConfig {
            /// Create a new instance of the config.
            #[allow(clippy::too_many_arguments)]
            pub fn new(channels: [usize; 2], kernel_size: [usize; 3]) -> Self {
                Self {
                    channels: channels,
                    kernel_size: kernel_size,
                    stride: [1, 1, 1],
                    dilation: [1, 1, 1],
                    groups: 1,
                    padding: [0, 0, 0],
                    padding_out: [0, 0, 0],
                    bias: true,
                    initializer: Initializer::KaimingUniform {
                        gain: 1.0 / num_traits::Float::sqrt(3.0),
                        fan_out_only: false,
                    },
                }
            }
        }
        impl ConvTranspose3dConfig {
            /// The stride of the convolution.
            pub fn with_stride(mut self, stride: [usize; 3]) -> Self {
                self.stride = stride;
                self
            }
            /// Spacing between kernel elements.
            pub fn with_dilation(mut self, dilation: [usize; 3]) -> Self {
                self.dilation = dilation;
                self
            }
            /// Controls the connections between input and output channels.
            pub fn with_groups(mut self, groups: usize) -> Self {
                self.groups = groups;
                self
            }
            /// The padding configuration.
            pub fn with_padding(mut self, padding: [usize; 3]) -> Self {
                self.padding = padding;
                self
            }
            /// The padding output configuration.
            pub fn with_padding_out(mut self, padding_out: [usize; 3]) -> Self {
                self.padding_out = padding_out;
                self
            }
            /// If bias should be added to the output.
            pub fn with_bias(mut self, bias: bool) -> Self {
                self.bias = bias;
                self
            }
            /// The type of function used to initialize neural network parameters
            pub fn with_initializer(mut self, initializer: Initializer) -> Self {
                self.initializer = initializer;
                self
            }
        }
        impl burn::serde::Serialize for ConvTranspose3dConfig {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: burn::serde::Serializer,
            {
                #[serde(crate = "burn::serde")]
                struct ConvTranspose3dConfigSerde {
                    channels: [usize; 2],
                    kernel_size: [usize; 3],
                    stride: [usize; 3],
                    dilation: [usize; 3],
                    groups: usize,
                    padding: [usize; 3],
                    padding_out: [usize; 3],
                    bias: bool,
                    initializer: Initializer,
                }
                #[doc(hidden)]
                #[allow(
                    non_upper_case_globals,
                    unused_attributes,
                    unused_qualifications,
                    clippy::absolute_paths,
                )]
                const _: () = {
                    use burn::serde as _serde;
                    #[automatically_derived]
                    impl _serde::Serialize for ConvTranspose3dConfigSerde {
                        fn serialize<__S>(
                            &self,
                            __serializer: __S,
                        ) -> _serde::__private226::Result<__S::Ok, __S::Error>
                        where
                            __S: _serde::Serializer,
                        {
                            let mut __serde_state = _serde::Serializer::serialize_struct(
                                __serializer,
                                "ConvTranspose3dConfigSerde",
                                false as usize + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "channels",
                                &self.channels,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "kernel_size",
                                &self.kernel_size,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "stride",
                                &self.stride,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "dilation",
                                &self.dilation,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "groups",
                                &self.groups,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "padding",
                                &self.padding,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "padding_out",
                                &self.padding_out,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "bias",
                                &self.bias,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "initializer",
                                &self.initializer,
                            )?;
                            _serde::ser::SerializeStruct::end(__serde_state)
                        }
                    }
                };
                let serde_state = ConvTranspose3dConfigSerde {
                    channels: self.channels.clone(),
                    kernel_size: self.kernel_size.clone(),
                    stride: self.stride.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                    padding_out: self.padding_out.clone(),
                    bias: self.bias.clone(),
                    initializer: self.initializer.clone(),
                };
                serde_state.serialize(serializer)
            }
        }
        impl<'de> burn::serde::Deserialize<'de> for ConvTranspose3dConfig {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: burn::serde::Deserializer<'de>,
            {
                #[serde(crate = "burn::serde")]
                struct ConvTranspose3dConfigSerde {
                    channels: [usize; 2],
                    kernel_size: [usize; 3],
                    stride: [usize; 3],
                    dilation: [usize; 3],
                    groups: usize,
                    padding: [usize; 3],
                    padding_out: [usize; 3],
                    bias: bool,
                    initializer: Initializer,
                }
                #[doc(hidden)]
                #[allow(
                    non_upper_case_globals,
                    unused_attributes,
                    unused_qualifications,
                    clippy::absolute_paths,
                )]
                const _: () = {
                    use burn::serde as _serde;
                    #[automatically_derived]
                    impl<'de> _serde::Deserialize<'de> for ConvTranspose3dConfigSerde {
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private226::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            #[allow(non_camel_case_types)]
                            #[doc(hidden)]
                            enum __Field {
                                __field0,
                                __field1,
                                __field2,
                                __field3,
                                __field4,
                                __field5,
                                __field6,
                                __field7,
                                __field8,
                                __ignore,
                            }
                            #[doc(hidden)]
                            struct __FieldVisitor;
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                type Value = __Field;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private226::Formatter,
                                ) -> _serde::__private226::fmt::Result {
                                    _serde::__private226::Formatter::write_str(
                                        __formatter,
                                        "field identifier",
                                    )
                                }
                                fn visit_u64<__E>(
                                    self,
                                    __value: u64,
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        0u64 => _serde::__private226::Ok(__Field::__field0),
                                        1u64 => _serde::__private226::Ok(__Field::__field1),
                                        2u64 => _serde::__private226::Ok(__Field::__field2),
                                        3u64 => _serde::__private226::Ok(__Field::__field3),
                                        4u64 => _serde::__private226::Ok(__Field::__field4),
                                        5u64 => _serde::__private226::Ok(__Field::__field5),
                                        6u64 => _serde::__private226::Ok(__Field::__field6),
                                        7u64 => _serde::__private226::Ok(__Field::__field7),
                                        8u64 => _serde::__private226::Ok(__Field::__field8),
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_str<__E>(
                                    self,
                                    __value: &str,
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        "channels" => _serde::__private226::Ok(__Field::__field0),
                                        "kernel_size" => _serde::__private226::Ok(__Field::__field1),
                                        "stride" => _serde::__private226::Ok(__Field::__field2),
                                        "dilation" => _serde::__private226::Ok(__Field::__field3),
                                        "groups" => _serde::__private226::Ok(__Field::__field4),
                                        "padding" => _serde::__private226::Ok(__Field::__field5),
                                        "padding_out" => _serde::__private226::Ok(__Field::__field6),
                                        "bias" => _serde::__private226::Ok(__Field::__field7),
                                        "initializer" => _serde::__private226::Ok(__Field::__field8),
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_bytes<__E>(
                                    self,
                                    __value: &[u8],
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        b"channels" => _serde::__private226::Ok(__Field::__field0),
                                        b"kernel_size" => {
                                            _serde::__private226::Ok(__Field::__field1)
                                        }
                                        b"stride" => _serde::__private226::Ok(__Field::__field2),
                                        b"dilation" => _serde::__private226::Ok(__Field::__field3),
                                        b"groups" => _serde::__private226::Ok(__Field::__field4),
                                        b"padding" => _serde::__private226::Ok(__Field::__field5),
                                        b"padding_out" => {
                                            _serde::__private226::Ok(__Field::__field6)
                                        }
                                        b"bias" => _serde::__private226::Ok(__Field::__field7),
                                        b"initializer" => {
                                            _serde::__private226::Ok(__Field::__field8)
                                        }
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                            }
                            #[automatically_derived]
                            impl<'de> _serde::Deserialize<'de> for __Field {
                                #[inline]
                                fn deserialize<__D>(
                                    __deserializer: __D,
                                ) -> _serde::__private226::Result<Self, __D::Error>
                                where
                                    __D: _serde::Deserializer<'de>,
                                {
                                    _serde::Deserializer::deserialize_identifier(
                                        __deserializer,
                                        __FieldVisitor,
                                    )
                                }
                            }
                            #[doc(hidden)]
                            struct __Visitor<'de> {
                                marker: _serde::__private226::PhantomData<
                                    ConvTranspose3dConfigSerde,
                                >,
                                lifetime: _serde::__private226::PhantomData<&'de ()>,
                            }
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                type Value = ConvTranspose3dConfigSerde;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private226::Formatter,
                                ) -> _serde::__private226::fmt::Result {
                                    _serde::__private226::Formatter::write_str(
                                        __formatter,
                                        "struct ConvTranspose3dConfigSerde",
                                    )
                                }
                                #[inline]
                                fn visit_seq<__A>(
                                    self,
                                    mut __seq: __A,
                                ) -> _serde::__private226::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::SeqAccess<'de>,
                                {
                                    let __field0 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    0usize,
                                                    &"struct ConvTranspose3dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field1 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 3],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    1usize,
                                                    &"struct ConvTranspose3dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field2 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 3],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    2usize,
                                                    &"struct ConvTranspose3dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field3 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 3],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    3usize,
                                                    &"struct ConvTranspose3dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field4 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    4usize,
                                                    &"struct ConvTranspose3dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field5 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 3],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    5usize,
                                                    &"struct ConvTranspose3dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field6 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 3],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    6usize,
                                                    &"struct ConvTranspose3dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field7 = match _serde::de::SeqAccess::next_element::<
                                        bool,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    7usize,
                                                    &"struct ConvTranspose3dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field8 = match _serde::de::SeqAccess::next_element::<
                                        Initializer,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    8usize,
                                                    &"struct ConvTranspose3dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    _serde::__private226::Ok(ConvTranspose3dConfigSerde {
                                        channels: __field0,
                                        kernel_size: __field1,
                                        stride: __field2,
                                        dilation: __field3,
                                        groups: __field4,
                                        padding: __field5,
                                        padding_out: __field6,
                                        bias: __field7,
                                        initializer: __field8,
                                    })
                                }
                                #[inline]
                                fn visit_map<__A>(
                                    self,
                                    mut __map: __A,
                                ) -> _serde::__private226::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::MapAccess<'de>,
                                {
                                    let mut __field0: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field1: _serde::__private226::Option<
                                        [usize; 3],
                                    > = _serde::__private226::None;
                                    let mut __field2: _serde::__private226::Option<
                                        [usize; 3],
                                    > = _serde::__private226::None;
                                    let mut __field3: _serde::__private226::Option<
                                        [usize; 3],
                                    > = _serde::__private226::None;
                                    let mut __field4: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field5: _serde::__private226::Option<
                                        [usize; 3],
                                    > = _serde::__private226::None;
                                    let mut __field6: _serde::__private226::Option<
                                        [usize; 3],
                                    > = _serde::__private226::None;
                                    let mut __field7: _serde::__private226::Option<bool> = _serde::__private226::None;
                                    let mut __field8: _serde::__private226::Option<
                                        Initializer,
                                    > = _serde::__private226::None;
                                    while let _serde::__private226::Some(__key) = _serde::de::MapAccess::next_key::<
                                        __Field,
                                    >(&mut __map)? {
                                        match __key {
                                            __Field::__field0 => {
                                                if _serde::__private226::Option::is_some(&__field0) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "channels",
                                                        ),
                                                    );
                                                }
                                                __field0 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field1 => {
                                                if _serde::__private226::Option::is_some(&__field1) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "kernel_size",
                                                        ),
                                                    );
                                                }
                                                __field1 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 3]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field2 => {
                                                if _serde::__private226::Option::is_some(&__field2) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                    );
                                                }
                                                __field2 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 3]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field3 => {
                                                if _serde::__private226::Option::is_some(&__field3) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "dilation",
                                                        ),
                                                    );
                                                }
                                                __field3 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 3]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field4 => {
                                                if _serde::__private226::Option::is_some(&__field4) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("groups"),
                                                    );
                                                }
                                                __field4 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field5 => {
                                                if _serde::__private226::Option::is_some(&__field5) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "padding",
                                                        ),
                                                    );
                                                }
                                                __field5 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 3]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field6 => {
                                                if _serde::__private226::Option::is_some(&__field6) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "padding_out",
                                                        ),
                                                    );
                                                }
                                                __field6 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 3]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field7 => {
                                                if _serde::__private226::Option::is_some(&__field7) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("bias"),
                                                    );
                                                }
                                                __field7 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<bool>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field8 => {
                                                if _serde::__private226::Option::is_some(&__field8) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "initializer",
                                                        ),
                                                    );
                                                }
                                                __field8 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<
                                                        Initializer,
                                                    >(&mut __map)?,
                                                );
                                            }
                                            _ => {
                                                let _ = _serde::de::MapAccess::next_value::<
                                                    _serde::de::IgnoredAny,
                                                >(&mut __map)?;
                                            }
                                        }
                                    }
                                    let __field0 = match __field0 {
                                        _serde::__private226::Some(__field0) => __field0,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("channels")?
                                        }
                                    };
                                    let __field1 = match __field1 {
                                        _serde::__private226::Some(__field1) => __field1,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("kernel_size")?
                                        }
                                    };
                                    let __field2 = match __field2 {
                                        _serde::__private226::Some(__field2) => __field2,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("stride")?
                                        }
                                    };
                                    let __field3 = match __field3 {
                                        _serde::__private226::Some(__field3) => __field3,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("dilation")?
                                        }
                                    };
                                    let __field4 = match __field4 {
                                        _serde::__private226::Some(__field4) => __field4,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("groups")?
                                        }
                                    };
                                    let __field5 = match __field5 {
                                        _serde::__private226::Some(__field5) => __field5,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("padding")?
                                        }
                                    };
                                    let __field6 = match __field6 {
                                        _serde::__private226::Some(__field6) => __field6,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("padding_out")?
                                        }
                                    };
                                    let __field7 = match __field7 {
                                        _serde::__private226::Some(__field7) => __field7,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("bias")?
                                        }
                                    };
                                    let __field8 = match __field8 {
                                        _serde::__private226::Some(__field8) => __field8,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("initializer")?
                                        }
                                    };
                                    _serde::__private226::Ok(ConvTranspose3dConfigSerde {
                                        channels: __field0,
                                        kernel_size: __field1,
                                        stride: __field2,
                                        dilation: __field3,
                                        groups: __field4,
                                        padding: __field5,
                                        padding_out: __field6,
                                        bias: __field7,
                                        initializer: __field8,
                                    })
                                }
                            }
                            #[doc(hidden)]
                            const FIELDS: &'static [&'static str] = &[
                                "channels",
                                "kernel_size",
                                "stride",
                                "dilation",
                                "groups",
                                "padding",
                                "padding_out",
                                "bias",
                                "initializer",
                            ];
                            _serde::Deserializer::deserialize_struct(
                                __deserializer,
                                "ConvTranspose3dConfigSerde",
                                FIELDS,
                                __Visitor {
                                    marker: _serde::__private226::PhantomData::<
                                        ConvTranspose3dConfigSerde,
                                    >,
                                    lifetime: _serde::__private226::PhantomData,
                                },
                            )
                        }
                    }
                };
                let serde_state = ConvTranspose3dConfigSerde::deserialize(deserializer)?;
                Ok(ConvTranspose3dConfig {
                    channels: serde_state.channels,
                    kernel_size: serde_state.kernel_size,
                    stride: serde_state.stride,
                    dilation: serde_state.dilation,
                    groups: serde_state.groups,
                    padding: serde_state.padding,
                    padding_out: serde_state.padding_out,
                    bias: serde_state.bias,
                    initializer: serde_state.initializer,
                })
            }
        }
        impl Clone for ConvTranspose3dConfig {
            fn clone(&self) -> Self {
                Self {
                    channels: self.channels.clone(),
                    kernel_size: self.kernel_size.clone(),
                    stride: self.stride.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                    padding_out: self.padding_out.clone(),
                    bias: self.bias.clone(),
                    initializer: self.initializer.clone(),
                }
            }
        }
        impl core::fmt::Display for ConvTranspose3dConfig {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(&burn::config::config_to_json(self))
            }
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for ConvTranspose3dConfig {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                let names: &'static _ = &[
                    "channels",
                    "kernel_size",
                    "stride",
                    "dilation",
                    "groups",
                    "padding",
                    "padding_out",
                    "bias",
                    "initializer",
                ];
                let values: &[&dyn ::core::fmt::Debug] = &[
                    &self.channels,
                    &self.kernel_size,
                    &self.stride,
                    &self.dilation,
                    &self.groups,
                    &self.padding,
                    &self.padding_out,
                    &self.bias,
                    &&self.initializer,
                ];
                ::core::fmt::Formatter::debug_struct_fields_finish(
                    f,
                    "ConvTranspose3dConfig",
                    names,
                    values,
                )
            }
        }
        /// Applies a 3D transposed convolution over input tensors.
        #[module(custom_display)]
        pub struct ConvTranspose3d<B: Backend> {
            /// Tensor of shape `[channels_in, channels_out / groups, kernel_size_1, kernel_size_2, kernel_size_3]`
            pub weight: Param<Tensor<B, 5>>,
            /// Tensor of shape `[channels_out]`
            pub bias: Option<Param<Tensor<B, 1>>>,
            /// Stride of the convolution.
            pub stride: [usize; 3],
            /// Size of the kernel.
            pub kernel_size: [usize; 3],
            /// Spacing between kernel elements.
            pub dilation: [usize; 3],
            /// Controls the connections between input and output channels.
            pub groups: usize,
            /// Padding configuration.
            pub padding: [usize; 3],
            /// Padding output configuration.
            pub padding_out: [usize; 3],
            /// Number of channels.
            pub channels: [usize; 2],
        }
        impl<B: Backend> burn::module::Module<B> for ConvTranspose3d<B> {
            type Record = ConvTranspose3dRecord<B>;
            fn load_record(self, record: Self::Record) -> Self {
                Self {
                    weight: burn::module::Module::<
                        B,
                    >::load_record(self.weight, record.weight),
                    bias: burn::module::Module::<B>::load_record(self.bias, record.bias),
                    stride: burn::module::Module::<
                        B,
                    >::load_record(self.stride, record.stride),
                    kernel_size: burn::module::Module::<
                        B,
                    >::load_record(self.kernel_size, record.kernel_size),
                    dilation: burn::module::Module::<
                        B,
                    >::load_record(self.dilation, record.dilation),
                    groups: burn::module::Module::<
                        B,
                    >::load_record(self.groups, record.groups),
                    padding: burn::module::Module::<
                        B,
                    >::load_record(self.padding, record.padding),
                    padding_out: burn::module::Module::<
                        B,
                    >::load_record(self.padding_out, record.padding_out),
                    channels: burn::module::Module::<
                        B,
                    >::load_record(self.channels, record.channels),
                }
            }
            fn into_record(self) -> Self::Record {
                Self::Record {
                    weight: burn::module::Module::<B>::into_record(self.weight),
                    bias: burn::module::Module::<B>::into_record(self.bias),
                    stride: burn::module::Module::<B>::into_record(self.stride),
                    kernel_size: burn::module::Module::<
                        B,
                    >::into_record(self.kernel_size),
                    dilation: burn::module::Module::<B>::into_record(self.dilation),
                    groups: burn::module::Module::<B>::into_record(self.groups),
                    padding: burn::module::Module::<B>::into_record(self.padding),
                    padding_out: burn::module::Module::<
                        B,
                    >::into_record(self.padding_out),
                    channels: burn::module::Module::<B>::into_record(self.channels),
                }
            }
            fn num_params(&self) -> usize {
                let mut num_params = 0;
                num_params += burn::module::Module::<B>::num_params(&self.weight);
                num_params += burn::module::Module::<B>::num_params(&self.bias);
                num_params += burn::module::Module::<B>::num_params(&self.stride);
                num_params += burn::module::Module::<B>::num_params(&self.kernel_size);
                num_params += burn::module::Module::<B>::num_params(&self.dilation);
                num_params += burn::module::Module::<B>::num_params(&self.groups);
                num_params += burn::module::Module::<B>::num_params(&self.padding);
                num_params += burn::module::Module::<B>::num_params(&self.padding_out);
                num_params += burn::module::Module::<B>::num_params(&self.channels);
                num_params
            }
            fn visit<Visitor: burn::module::ModuleVisitor<B>>(
                &self,
                visitor: &mut Visitor,
            ) {
                visitor.enter_module("weight", "ConvTranspose3d");
                burn::module::Module::visit(&self.weight, visitor);
                visitor.exit_module("weight", "ConvTranspose3d");
                visitor.enter_module("bias", "ConvTranspose3d");
                burn::module::Module::visit(&self.bias, visitor);
                visitor.exit_module("bias", "ConvTranspose3d");
                visitor.enter_module("stride", "ConvTranspose3d");
                burn::module::Module::visit(&self.stride, visitor);
                visitor.exit_module("stride", "ConvTranspose3d");
                visitor.enter_module("kernel_size", "ConvTranspose3d");
                burn::module::Module::visit(&self.kernel_size, visitor);
                visitor.exit_module("kernel_size", "ConvTranspose3d");
                visitor.enter_module("dilation", "ConvTranspose3d");
                burn::module::Module::visit(&self.dilation, visitor);
                visitor.exit_module("dilation", "ConvTranspose3d");
                visitor.enter_module("groups", "ConvTranspose3d");
                burn::module::Module::visit(&self.groups, visitor);
                visitor.exit_module("groups", "ConvTranspose3d");
                visitor.enter_module("padding", "ConvTranspose3d");
                burn::module::Module::visit(&self.padding, visitor);
                visitor.exit_module("padding", "ConvTranspose3d");
                visitor.enter_module("padding_out", "ConvTranspose3d");
                burn::module::Module::visit(&self.padding_out, visitor);
                visitor.exit_module("padding_out", "ConvTranspose3d");
                visitor.enter_module("channels", "ConvTranspose3d");
                burn::module::Module::visit(&self.channels, visitor);
                visitor.exit_module("channels", "ConvTranspose3d");
            }
            fn map<Mapper: burn::module::ModuleMapper<B>>(
                self,
                mapper: &mut Mapper,
            ) -> Self {
                mapper.enter_module("weight", "ConvTranspose3d");
                let weight = burn::module::Module::<B>::map(self.weight, mapper);
                mapper.exit_module("weight", "ConvTranspose3d");
                mapper.enter_module("bias", "ConvTranspose3d");
                let bias = burn::module::Module::<B>::map(self.bias, mapper);
                mapper.exit_module("bias", "ConvTranspose3d");
                mapper.enter_module("stride", "ConvTranspose3d");
                let stride = burn::module::Module::<B>::map(self.stride, mapper);
                mapper.exit_module("stride", "ConvTranspose3d");
                mapper.enter_module("kernel_size", "ConvTranspose3d");
                let kernel_size = burn::module::Module::<
                    B,
                >::map(self.kernel_size, mapper);
                mapper.exit_module("kernel_size", "ConvTranspose3d");
                mapper.enter_module("dilation", "ConvTranspose3d");
                let dilation = burn::module::Module::<B>::map(self.dilation, mapper);
                mapper.exit_module("dilation", "ConvTranspose3d");
                mapper.enter_module("groups", "ConvTranspose3d");
                let groups = burn::module::Module::<B>::map(self.groups, mapper);
                mapper.exit_module("groups", "ConvTranspose3d");
                mapper.enter_module("padding", "ConvTranspose3d");
                let padding = burn::module::Module::<B>::map(self.padding, mapper);
                mapper.exit_module("padding", "ConvTranspose3d");
                mapper.enter_module("padding_out", "ConvTranspose3d");
                let padding_out = burn::module::Module::<
                    B,
                >::map(self.padding_out, mapper);
                mapper.exit_module("padding_out", "ConvTranspose3d");
                mapper.enter_module("channels", "ConvTranspose3d");
                let channels = burn::module::Module::<B>::map(self.channels, mapper);
                mapper.exit_module("channels", "ConvTranspose3d");
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                    padding_out,
                    channels,
                }
            }
            fn collect_devices(
                &self,
                devices: burn::module::Devices<B>,
            ) -> burn::module::Devices<B> {
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.weight, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.bias, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.stride, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.kernel_size, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.dilation, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.groups, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.padding, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.padding_out, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.channels, devices);
                devices
            }
            fn to_device(self, device: &B::Device) -> Self {
                let weight = burn::module::Module::<B>::to_device(self.weight, device);
                let bias = burn::module::Module::<B>::to_device(self.bias, device);
                let stride = burn::module::Module::<B>::to_device(self.stride, device);
                let kernel_size = burn::module::Module::<
                    B,
                >::to_device(self.kernel_size, device);
                let dilation = burn::module::Module::<
                    B,
                >::to_device(self.dilation, device);
                let groups = burn::module::Module::<B>::to_device(self.groups, device);
                let padding = burn::module::Module::<B>::to_device(self.padding, device);
                let padding_out = burn::module::Module::<
                    B,
                >::to_device(self.padding_out, device);
                let channels = burn::module::Module::<
                    B,
                >::to_device(self.channels, device);
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                    padding_out,
                    channels,
                }
            }
            fn fork(self, device: &B::Device) -> Self {
                let weight = burn::module::Module::<B>::fork(self.weight, device);
                let bias = burn::module::Module::<B>::fork(self.bias, device);
                let stride = burn::module::Module::<B>::fork(self.stride, device);
                let kernel_size = burn::module::Module::<
                    B,
                >::fork(self.kernel_size, device);
                let dilation = burn::module::Module::<B>::fork(self.dilation, device);
                let groups = burn::module::Module::<B>::fork(self.groups, device);
                let padding = burn::module::Module::<B>::fork(self.padding, device);
                let padding_out = burn::module::Module::<
                    B,
                >::fork(self.padding_out, device);
                let channels = burn::module::Module::<B>::fork(self.channels, device);
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                    padding_out,
                    channels,
                }
            }
        }
        impl<B: Backend> burn::module::AutodiffModule<B> for ConvTranspose3d<B>
        where
            B: burn::tensor::backend::AutodiffBackend,
            <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
        {
            type InnerModule = ConvTranspose3d<B::InnerBackend>;
            fn valid(&self) -> Self::InnerModule {
                let weight = burn::module::AutodiffModule::<B>::valid(&self.weight);
                let bias = burn::module::AutodiffModule::<B>::valid(&self.bias);
                let stride = burn::module::AutodiffModule::<B>::valid(&self.stride);
                let kernel_size = burn::module::AutodiffModule::<
                    B,
                >::valid(&self.kernel_size);
                let dilation = burn::module::AutodiffModule::<B>::valid(&self.dilation);
                let groups = burn::module::AutodiffModule::<B>::valid(&self.groups);
                let padding = burn::module::AutodiffModule::<B>::valid(&self.padding);
                let padding_out = burn::module::AutodiffModule::<
                    B,
                >::valid(&self.padding_out);
                let channels = burn::module::AutodiffModule::<B>::valid(&self.channels);
                Self::InnerModule {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                    padding_out,
                    channels,
                }
            }
        }
        impl<B: Backend> core::fmt::Display for ConvTranspose3d<B> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let formatted = burn::module::ModuleDisplay::format(
                    self,
                    Default::default(),
                );
                f.write_fmt(format_args!("{0}", formatted))
            }
        }
        impl<B: Backend> burn::module::ModuleDisplayDefault for ConvTranspose3d<B> {
            fn content(
                &self,
                mut content: burn::module::Content,
            ) -> Option<burn::module::Content> {
                content
                    .set_top_level_type(&"ConvTranspose3d")
                    .add("weight", &self.weight)
                    .add("bias", &self.bias)
                    .add("stride", &self.stride)
                    .add("kernel_size", &self.kernel_size)
                    .add("dilation", &self.dilation)
                    .add("groups", &self.groups)
                    .add("padding", &self.padding)
                    .add("padding_out", &self.padding_out)
                    .add("channels", &self.channels)
                    .optional()
            }
            fn num_params(&self) -> usize {
                burn::module::Module::num_params(self)
            }
        }
        impl<B: Backend> Clone for ConvTranspose3d<B> {
            fn clone(&self) -> Self {
                let weight = self.weight.clone();
                let bias = self.bias.clone();
                let stride = self.stride.clone();
                let kernel_size = self.kernel_size.clone();
                let dilation = self.dilation.clone();
                let groups = self.groups.clone();
                let padding = self.padding.clone();
                let padding_out = self.padding_out.clone();
                let channels = self.channels.clone();
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    groups,
                    padding,
                    padding_out,
                    channels,
                }
            }
        }
        /// The record type for the module.
        pub struct ConvTranspose3dRecord<B: Backend> {
            /// The module record associative type.
            pub weight: <Param<Tensor<B, 5>> as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub bias: <Option<Param<Tensor<B, 1>>> as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub stride: <[usize; 3] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub kernel_size: <[usize; 3] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub dilation: <[usize; 3] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub groups: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub padding: <[usize; 3] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub padding_out: <[usize; 3] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub channels: <[usize; 2] as burn::module::Module<B>>::Record,
        }
        /// The record item type for the module.
        #[serde(crate = "burn::serde")]
        #[serde(
            bound = "< < Param < Tensor < B, 5 > > as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize +\nburn :: serde :: de :: DeserializeOwned, < < Option < Param < Tensor < B, 1 >\n> > as burn :: module :: Module < B > > :: Record as burn :: record :: Record\n< B >> :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de ::\nDeserializeOwned, < < [usize; 3] as burn :: module :: Module < B > > :: Record\nas burn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize\n+ burn :: serde :: de :: DeserializeOwned, < < [usize; 3] as burn :: module ::\nModule < B > > :: Record as burn :: record :: Record < B >> :: Item < S > :\nburn :: serde :: Serialize + burn :: serde :: de :: DeserializeOwned, < <\n[usize; 3] as burn :: module :: Module < B > > :: Record as burn :: record ::\nRecord < B >> :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de\n:: DeserializeOwned, < < usize as burn :: module :: Module < B > > :: Record\nas burn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize\n+ burn :: serde :: de :: DeserializeOwned, < < [usize; 3] as burn :: module ::\nModule < B > > :: Record as burn :: record :: Record < B >> :: Item < S > :\nburn :: serde :: Serialize + burn :: serde :: de :: DeserializeOwned, < <\n[usize; 3] as burn :: module :: Module < B > > :: Record as burn :: record ::\nRecord < B >> :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de\n:: DeserializeOwned, < < [usize; 2] as burn :: module :: Module < B > > ::\nRecord as burn :: record :: Record < B >> :: Item < S > : burn :: serde ::\nSerialize + burn :: serde :: de :: DeserializeOwned,"
        )]
        pub struct ConvTranspose3dRecordItem<
            B: Backend,
            S: burn::record::PrecisionSettings,
        > {
            /// Field to be serialized.
            pub weight: <<Param<
                Tensor<B, 5>,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub bias: <<Option<
                Param<Tensor<B, 1>>,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub stride: <<[usize; 3] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub kernel_size: <<[usize; 3] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub dilation: <<[usize; 3] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub groups: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub padding: <<[usize; 3] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub padding_out: <<[usize; 3] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub channels: <<[usize; 2] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
        }
        #[doc(hidden)]
        #[allow(
            non_upper_case_globals,
            unused_attributes,
            unused_qualifications,
            clippy::absolute_paths,
        )]
        const _: () = {
            use burn::serde as _serde;
            #[automatically_derived]
            impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
            for ConvTranspose3dRecordItem<B, S>
            where
                <<Param<
                    Tensor<B, 5>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Option<
                    Param<Tensor<B, 1>>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            {
                fn serialize<__S>(
                    &self,
                    __serializer: __S,
                ) -> _serde::__private226::Result<__S::Ok, __S::Error>
                where
                    __S: _serde::Serializer,
                {
                    let mut __serde_state = _serde::Serializer::serialize_struct(
                        __serializer,
                        "ConvTranspose3dRecordItem",
                        false as usize + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "weight",
                        &self.weight,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "bias",
                        &self.bias,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "stride",
                        &self.stride,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "kernel_size",
                        &self.kernel_size,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "dilation",
                        &self.dilation,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "groups",
                        &self.groups,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "padding",
                        &self.padding,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "padding_out",
                        &self.padding_out,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "channels",
                        &self.channels,
                    )?;
                    _serde::ser::SerializeStruct::end(__serde_state)
                }
            }
        };
        #[doc(hidden)]
        #[allow(
            non_upper_case_globals,
            unused_attributes,
            unused_qualifications,
            clippy::absolute_paths,
        )]
        const _: () = {
            use burn::serde as _serde;
            #[automatically_derived]
            impl<
                'de,
                B: Backend,
                S: burn::record::PrecisionSettings,
            > _serde::Deserialize<'de> for ConvTranspose3dRecordItem<B, S>
            where
                <<Param<
                    Tensor<B, 5>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Option<
                    Param<Tensor<B, 1>>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 3] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            {
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private226::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    #[allow(non_camel_case_types)]
                    #[doc(hidden)]
                    enum __Field {
                        __field0,
                        __field1,
                        __field2,
                        __field3,
                        __field4,
                        __field5,
                        __field6,
                        __field7,
                        __field8,
                        __ignore,
                    }
                    #[doc(hidden)]
                    struct __FieldVisitor;
                    #[automatically_derived]
                    impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                        type Value = __Field;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private226::Formatter,
                        ) -> _serde::__private226::fmt::Result {
                            _serde::__private226::Formatter::write_str(
                                __formatter,
                                "field identifier",
                            )
                        }
                        fn visit_u64<__E>(
                            self,
                            __value: u64,
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                0u64 => _serde::__private226::Ok(__Field::__field0),
                                1u64 => _serde::__private226::Ok(__Field::__field1),
                                2u64 => _serde::__private226::Ok(__Field::__field2),
                                3u64 => _serde::__private226::Ok(__Field::__field3),
                                4u64 => _serde::__private226::Ok(__Field::__field4),
                                5u64 => _serde::__private226::Ok(__Field::__field5),
                                6u64 => _serde::__private226::Ok(__Field::__field6),
                                7u64 => _serde::__private226::Ok(__Field::__field7),
                                8u64 => _serde::__private226::Ok(__Field::__field8),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_str<__E>(
                            self,
                            __value: &str,
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                "weight" => _serde::__private226::Ok(__Field::__field0),
                                "bias" => _serde::__private226::Ok(__Field::__field1),
                                "stride" => _serde::__private226::Ok(__Field::__field2),
                                "kernel_size" => _serde::__private226::Ok(__Field::__field3),
                                "dilation" => _serde::__private226::Ok(__Field::__field4),
                                "groups" => _serde::__private226::Ok(__Field::__field5),
                                "padding" => _serde::__private226::Ok(__Field::__field6),
                                "padding_out" => _serde::__private226::Ok(__Field::__field7),
                                "channels" => _serde::__private226::Ok(__Field::__field8),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_bytes<__E>(
                            self,
                            __value: &[u8],
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                b"weight" => _serde::__private226::Ok(__Field::__field0),
                                b"bias" => _serde::__private226::Ok(__Field::__field1),
                                b"stride" => _serde::__private226::Ok(__Field::__field2),
                                b"kernel_size" => {
                                    _serde::__private226::Ok(__Field::__field3)
                                }
                                b"dilation" => _serde::__private226::Ok(__Field::__field4),
                                b"groups" => _serde::__private226::Ok(__Field::__field5),
                                b"padding" => _serde::__private226::Ok(__Field::__field6),
                                b"padding_out" => {
                                    _serde::__private226::Ok(__Field::__field7)
                                }
                                b"channels" => _serde::__private226::Ok(__Field::__field8),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                    }
                    #[automatically_derived]
                    impl<'de> _serde::Deserialize<'de> for __Field {
                        #[inline]
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private226::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            _serde::Deserializer::deserialize_identifier(
                                __deserializer,
                                __FieldVisitor,
                            )
                        }
                    }
                    #[doc(hidden)]
                    struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                    where
                        <<Param<
                            Tensor<B, 5>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Option<
                            Param<Tensor<B, 1>>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    {
                        marker: _serde::__private226::PhantomData<
                            ConvTranspose3dRecordItem<B, S>,
                        >,
                        lifetime: _serde::__private226::PhantomData<&'de ()>,
                    }
                    #[automatically_derived]
                    impl<
                        'de,
                        B: Backend,
                        S: burn::record::PrecisionSettings,
                    > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                    where
                        <<Param<
                            Tensor<B, 5>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Option<
                            Param<Tensor<B, 1>>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 3] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    {
                        type Value = ConvTranspose3dRecordItem<B, S>;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private226::Formatter,
                        ) -> _serde::__private226::fmt::Result {
                            _serde::__private226::Formatter::write_str(
                                __formatter,
                                "struct ConvTranspose3dRecordItem",
                            )
                        }
                        #[inline]
                        fn visit_seq<__A>(
                            self,
                            mut __seq: __A,
                        ) -> _serde::__private226::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::SeqAccess<'de>,
                        {
                            let __field0 = match _serde::de::SeqAccess::next_element::<
                                <<Param<
                                    Tensor<B, 5>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            0usize,
                                            &"struct ConvTranspose3dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field1 = match _serde::de::SeqAccess::next_element::<
                                <<Option<
                                    Param<Tensor<B, 1>>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            1usize,
                                            &"struct ConvTranspose3dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field2 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            2usize,
                                            &"struct ConvTranspose3dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field3 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            3usize,
                                            &"struct ConvTranspose3dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field4 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            4usize,
                                            &"struct ConvTranspose3dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field5 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            5usize,
                                            &"struct ConvTranspose3dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field6 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            6usize,
                                            &"struct ConvTranspose3dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field7 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            7usize,
                                            &"struct ConvTranspose3dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            let __field8 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            8usize,
                                            &"struct ConvTranspose3dRecordItem with 9 elements",
                                        ),
                                    );
                                }
                            };
                            _serde::__private226::Ok(ConvTranspose3dRecordItem {
                                weight: __field0,
                                bias: __field1,
                                stride: __field2,
                                kernel_size: __field3,
                                dilation: __field4,
                                groups: __field5,
                                padding: __field6,
                                padding_out: __field7,
                                channels: __field8,
                            })
                        }
                        #[inline]
                        fn visit_map<__A>(
                            self,
                            mut __map: __A,
                        ) -> _serde::__private226::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::MapAccess<'de>,
                        {
                            let mut __field0: _serde::__private226::Option<
                                <<Param<
                                    Tensor<B, 5>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field1: _serde::__private226::Option<
                                <<Option<
                                    Param<Tensor<B, 1>>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field2: _serde::__private226::Option<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field3: _serde::__private226::Option<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field4: _serde::__private226::Option<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field5: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field6: _serde::__private226::Option<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field7: _serde::__private226::Option<
                                <<[usize; 3] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field8: _serde::__private226::Option<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            while let _serde::__private226::Some(__key) = _serde::de::MapAccess::next_key::<
                                __Field,
                            >(&mut __map)? {
                                match __key {
                                    __Field::__field0 => {
                                        if _serde::__private226::Option::is_some(&__field0) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("weight"),
                                            );
                                        }
                                        __field0 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Param<
                                                    Tensor<B, 5>,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field1 => {
                                        if _serde::__private226::Option::is_some(&__field1) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("bias"),
                                            );
                                        }
                                        __field1 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Option<
                                                    Param<Tensor<B, 1>>,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field2 => {
                                        if _serde::__private226::Option::is_some(&__field2) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                            );
                                        }
                                        __field2 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 3] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field3 => {
                                        if _serde::__private226::Option::is_some(&__field3) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "kernel_size",
                                                ),
                                            );
                                        }
                                        __field3 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 3] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field4 => {
                                        if _serde::__private226::Option::is_some(&__field4) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "dilation",
                                                ),
                                            );
                                        }
                                        __field4 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 3] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field5 => {
                                        if _serde::__private226::Option::is_some(&__field5) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("groups"),
                                            );
                                        }
                                        __field5 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field6 => {
                                        if _serde::__private226::Option::is_some(&__field6) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "padding",
                                                ),
                                            );
                                        }
                                        __field6 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 3] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field7 => {
                                        if _serde::__private226::Option::is_some(&__field7) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "padding_out",
                                                ),
                                            );
                                        }
                                        __field7 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 3] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field8 => {
                                        if _serde::__private226::Option::is_some(&__field8) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "channels",
                                                ),
                                            );
                                        }
                                        __field8 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 2] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    _ => {
                                        let _ = _serde::de::MapAccess::next_value::<
                                            _serde::de::IgnoredAny,
                                        >(&mut __map)?;
                                    }
                                }
                            }
                            let __field0 = match __field0 {
                                _serde::__private226::Some(__field0) => __field0,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("weight")?
                                }
                            };
                            let __field1 = match __field1 {
                                _serde::__private226::Some(__field1) => __field1,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("bias")?
                                }
                            };
                            let __field2 = match __field2 {
                                _serde::__private226::Some(__field2) => __field2,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("stride")?
                                }
                            };
                            let __field3 = match __field3 {
                                _serde::__private226::Some(__field3) => __field3,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("kernel_size")?
                                }
                            };
                            let __field4 = match __field4 {
                                _serde::__private226::Some(__field4) => __field4,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("dilation")?
                                }
                            };
                            let __field5 = match __field5 {
                                _serde::__private226::Some(__field5) => __field5,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("groups")?
                                }
                            };
                            let __field6 = match __field6 {
                                _serde::__private226::Some(__field6) => __field6,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("padding")?
                                }
                            };
                            let __field7 = match __field7 {
                                _serde::__private226::Some(__field7) => __field7,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("padding_out")?
                                }
                            };
                            let __field8 = match __field8 {
                                _serde::__private226::Some(__field8) => __field8,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("channels")?
                                }
                            };
                            _serde::__private226::Ok(ConvTranspose3dRecordItem {
                                weight: __field0,
                                bias: __field1,
                                stride: __field2,
                                kernel_size: __field3,
                                dilation: __field4,
                                groups: __field5,
                                padding: __field6,
                                padding_out: __field7,
                                channels: __field8,
                            })
                        }
                    }
                    #[doc(hidden)]
                    const FIELDS: &'static [&'static str] = &[
                        "weight",
                        "bias",
                        "stride",
                        "kernel_size",
                        "dilation",
                        "groups",
                        "padding",
                        "padding_out",
                        "channels",
                    ];
                    _serde::Deserializer::deserialize_struct(
                        __deserializer,
                        "ConvTranspose3dRecordItem",
                        FIELDS,
                        __Visitor {
                            marker: _serde::__private226::PhantomData::<
                                ConvTranspose3dRecordItem<B, S>,
                            >,
                            lifetime: _serde::__private226::PhantomData,
                        },
                    )
                }
            }
        };
        impl<B: Backend, S: burn::record::PrecisionSettings> Clone
        for ConvTranspose3dRecordItem<B, S> {
            fn clone(&self) -> Self {
                Self {
                    weight: self.weight.clone(),
                    bias: self.bias.clone(),
                    stride: self.stride.clone(),
                    kernel_size: self.kernel_size.clone(),
                    dilation: self.dilation.clone(),
                    groups: self.groups.clone(),
                    padding: self.padding.clone(),
                    padding_out: self.padding_out.clone(),
                    channels: self.channels.clone(),
                }
            }
        }
        impl<B: Backend> burn::record::Record<B> for ConvTranspose3dRecord<B> {
            type Item<S: burn::record::PrecisionSettings> = ConvTranspose3dRecordItem<
                B,
                S,
            >;
            fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
                ConvTranspose3dRecordItem {
                    weight: burn::record::Record::<B>::into_item::<S>(self.weight),
                    bias: burn::record::Record::<B>::into_item::<S>(self.bias),
                    stride: burn::record::Record::<B>::into_item::<S>(self.stride),
                    kernel_size: burn::record::Record::<
                        B,
                    >::into_item::<S>(self.kernel_size),
                    dilation: burn::record::Record::<B>::into_item::<S>(self.dilation),
                    groups: burn::record::Record::<B>::into_item::<S>(self.groups),
                    padding: burn::record::Record::<B>::into_item::<S>(self.padding),
                    padding_out: burn::record::Record::<
                        B,
                    >::into_item::<S>(self.padding_out),
                    channels: burn::record::Record::<B>::into_item::<S>(self.channels),
                }
            }
            fn from_item<S: burn::record::PrecisionSettings>(
                item: Self::Item<S>,
                device: &B::Device,
            ) -> Self {
                Self {
                    weight: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.weight, device),
                    bias: burn::record::Record::<B>::from_item::<S>(item.bias, device),
                    stride: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.stride, device),
                    kernel_size: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.kernel_size, device),
                    dilation: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.dilation, device),
                    groups: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.groups, device),
                    padding: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.padding, device),
                    padding_out: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.padding_out, device),
                    channels: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.channels, device),
                }
            }
        }
        #[automatically_derived]
        impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for ConvTranspose3d<B> {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                let names: &'static _ = &[
                    "weight",
                    "bias",
                    "stride",
                    "kernel_size",
                    "dilation",
                    "groups",
                    "padding",
                    "padding_out",
                    "channels",
                ];
                let values: &[&dyn ::core::fmt::Debug] = &[
                    &self.weight,
                    &self.bias,
                    &self.stride,
                    &self.kernel_size,
                    &self.dilation,
                    &self.groups,
                    &self.padding,
                    &self.padding_out,
                    &&self.channels,
                ];
                ::core::fmt::Formatter::debug_struct_fields_finish(
                    f,
                    "ConvTranspose3d",
                    names,
                    values,
                )
            }
        }
        impl<B: Backend> ModuleDisplay for ConvTranspose3d<B> {
            fn custom_settings(&self) -> Option<DisplaySettings> {
                DisplaySettings::new().with_new_line_after_attribute(false).optional()
            }
            fn custom_content(&self, content: Content) -> Option<Content> {
                content
                    .add(
                        "channels",
                        &::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("{0:?}", &self.channels))
                        }),
                    )
                    .add(
                        "stride",
                        &::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("{0:?}", &self.stride))
                        }),
                    )
                    .add(
                        "kernel_size",
                        &::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!("{0:?}", &self.kernel_size),
                            )
                        }),
                    )
                    .add(
                        "dilation",
                        &::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("{0:?}", &self.dilation))
                        }),
                    )
                    .add("groups", &self.groups)
                    .add(
                        "padding",
                        &::alloc::__export::must_use({
                            ::alloc::fmt::format(format_args!("{0:?}", &self.padding))
                        }),
                    )
                    .add(
                        "padding_out",
                        &::alloc::__export::must_use({
                            ::alloc::fmt::format(
                                format_args!("{0:?}", &self.padding_out),
                            )
                        }),
                    )
                    .optional()
            }
        }
        impl ConvTranspose3dConfig {
            /// Initialize a new [conv transpose 2d](ConvTranspose3d) module.
            pub fn init<B: Backend>(&self, device: &B::Device) -> ConvTranspose3d<B> {
                checks::checks_channels_div_groups(
                    self.channels[0],
                    self.channels[1],
                    self.groups,
                );
                let shape = [
                    self.channels[0],
                    self.channels[1] / self.groups,
                    self.kernel_size[0],
                    self.kernel_size[1],
                    self.kernel_size[2],
                ];
                let fan_in = self.channels[1] / self.groups
                    * self.kernel_size.iter().product::<usize>();
                let weight = self
                    .initializer
                    .init_with(shape, Some(fan_in), None, device);
                let mut bias = None;
                if self.bias {
                    bias = Some(
                        self
                            .initializer
                            .init_with([self.channels[1]], Some(fan_in), None, device),
                    );
                }
                ConvTranspose3d {
                    weight,
                    bias,
                    stride: self.stride,
                    kernel_size: self.kernel_size,
                    dilation: self.dilation,
                    groups: self.groups,
                    padding: self.padding,
                    padding_out: self.padding_out,
                    channels: self.channels,
                }
            }
        }
        impl<B: Backend> ConvTranspose3d<B> {
            /// Applies the forward pass on the input tensor.
            ///
            /// See also [conv_transpose3d](burn::tensor::module::conv_transpose3d).
            ///
            /// # Shapes
            ///
            /// - input: `[batch_size, channels_in, depth_in, height_in, width_in]`
            /// - output: `[batch_size, channels_out, depth_out, height_out, width_out]`
            pub fn forward(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
                conv_transpose3d(
                    input,
                    self.weight.val(),
                    self.bias.as_ref().map(|bias| bias.val()),
                    ConvTransposeOptions::new(
                        self.stride,
                        self.padding,
                        self.padding_out,
                        self.dilation,
                        self.groups,
                    ),
                )
            }
        }
    }
    mod deform_conv2d {
        use alloc::format;
        use burn::tensor::ops::DeformConvOptions;
        use burn_core as burn;
        use crate::PaddingConfig2d;
        use burn::config::Config;
        use burn::module::Initializer;
        use burn::module::{
            Content, DisplaySettings, Ignored, Module, ModuleDisplay, Param,
        };
        use burn::tensor::Tensor;
        use burn::tensor::backend::Backend;
        use burn::tensor::module::deform_conv2d;
        use crate::conv::checks;
        /// Configuration to create a [deformable 2D convolution](DeformConv2d) layer, using the [init function](DeformConv2dConfig::init).
        pub struct DeformConv2dConfig {
            /// The number of channels.
            pub channels: [usize; 2],
            /// The size of the kernel.
            pub kernel_size: [usize; 2],
            /// The stride of the convolution.
            #[config(default = "[1, 1]")]
            pub stride: [usize; 2],
            /// Spacing between kernel elements.
            #[config(default = "[1, 1]")]
            pub dilation: [usize; 2],
            /// Controls the connections between input and output channels.
            #[config(default = "1")]
            pub weight_groups: usize,
            /// Offset groups.
            #[config(default = "1")]
            pub offset_groups: usize,
            /// The padding configuration.
            ///
            /// ### Warning
            /// Only symmetric padding is currently supported. As such, using `Same` padding with an even kernel
            /// size is not supported as it will not produce the same output size.
            #[config(default = "PaddingConfig2d::Valid")]
            pub padding: PaddingConfig2d,
            /// If bias should be added to the output.
            #[config(default = true)]
            pub bias: bool,
            /// The type of function used to initialize neural network parameters
            #[config(
                default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
            )]
            pub initializer: Initializer,
        }
        impl burn::config::Config for DeformConv2dConfig {}
        impl DeformConv2dConfig {
            /// Create a new instance of the config.
            #[allow(clippy::too_many_arguments)]
            pub fn new(channels: [usize; 2], kernel_size: [usize; 2]) -> Self {
                Self {
                    channels: channels,
                    kernel_size: kernel_size,
                    stride: [1, 1],
                    dilation: [1, 1],
                    weight_groups: 1,
                    offset_groups: 1,
                    padding: PaddingConfig2d::Valid,
                    bias: true,
                    initializer: Initializer::KaimingUniform {
                        gain: 1.0 / num_traits::Float::sqrt(3.0),
                        fan_out_only: false,
                    },
                }
            }
        }
        impl DeformConv2dConfig {
            /// The stride of the convolution.
            pub fn with_stride(mut self, stride: [usize; 2]) -> Self {
                self.stride = stride;
                self
            }
            /// Spacing between kernel elements.
            pub fn with_dilation(mut self, dilation: [usize; 2]) -> Self {
                self.dilation = dilation;
                self
            }
            /// Controls the connections between input and output channels.
            pub fn with_weight_groups(mut self, weight_groups: usize) -> Self {
                self.weight_groups = weight_groups;
                self
            }
            /// Offset groups.
            pub fn with_offset_groups(mut self, offset_groups: usize) -> Self {
                self.offset_groups = offset_groups;
                self
            }
            /// The padding configuration.
            pub fn with_padding(mut self, padding: PaddingConfig2d) -> Self {
                self.padding = padding;
                self
            }
            /// If bias should be added to the output.
            pub fn with_bias(mut self, bias: bool) -> Self {
                self.bias = bias;
                self
            }
            /// The type of function used to initialize neural network parameters
            pub fn with_initializer(mut self, initializer: Initializer) -> Self {
                self.initializer = initializer;
                self
            }
        }
        impl burn::serde::Serialize for DeformConv2dConfig {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: burn::serde::Serializer,
            {
                #[serde(crate = "burn::serde")]
                struct DeformConv2dConfigSerde {
                    channels: [usize; 2],
                    kernel_size: [usize; 2],
                    stride: [usize; 2],
                    dilation: [usize; 2],
                    weight_groups: usize,
                    offset_groups: usize,
                    padding: PaddingConfig2d,
                    bias: bool,
                    initializer: Initializer,
                }
                #[doc(hidden)]
                #[allow(
                    non_upper_case_globals,
                    unused_attributes,
                    unused_qualifications,
                    clippy::absolute_paths,
                )]
                const _: () = {
                    use burn::serde as _serde;
                    #[automatically_derived]
                    impl _serde::Serialize for DeformConv2dConfigSerde {
                        fn serialize<__S>(
                            &self,
                            __serializer: __S,
                        ) -> _serde::__private226::Result<__S::Ok, __S::Error>
                        where
                            __S: _serde::Serializer,
                        {
                            let mut __serde_state = _serde::Serializer::serialize_struct(
                                __serializer,
                                "DeformConv2dConfigSerde",
                                false as usize + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "channels",
                                &self.channels,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "kernel_size",
                                &self.kernel_size,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "stride",
                                &self.stride,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "dilation",
                                &self.dilation,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "weight_groups",
                                &self.weight_groups,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "offset_groups",
                                &self.offset_groups,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "padding",
                                &self.padding,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "bias",
                                &self.bias,
                            )?;
                            _serde::ser::SerializeStruct::serialize_field(
                                &mut __serde_state,
                                "initializer",
                                &self.initializer,
                            )?;
                            _serde::ser::SerializeStruct::end(__serde_state)
                        }
                    }
                };
                let serde_state = DeformConv2dConfigSerde {
                    channels: self.channels.clone(),
                    kernel_size: self.kernel_size.clone(),
                    stride: self.stride.clone(),
                    dilation: self.dilation.clone(),
                    weight_groups: self.weight_groups.clone(),
                    offset_groups: self.offset_groups.clone(),
                    padding: self.padding.clone(),
                    bias: self.bias.clone(),
                    initializer: self.initializer.clone(),
                };
                serde_state.serialize(serializer)
            }
        }
        impl<'de> burn::serde::Deserialize<'de> for DeformConv2dConfig {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: burn::serde::Deserializer<'de>,
            {
                #[serde(crate = "burn::serde")]
                struct DeformConv2dConfigSerde {
                    channels: [usize; 2],
                    kernel_size: [usize; 2],
                    stride: [usize; 2],
                    dilation: [usize; 2],
                    weight_groups: usize,
                    offset_groups: usize,
                    padding: PaddingConfig2d,
                    bias: bool,
                    initializer: Initializer,
                }
                #[doc(hidden)]
                #[allow(
                    non_upper_case_globals,
                    unused_attributes,
                    unused_qualifications,
                    clippy::absolute_paths,
                )]
                const _: () = {
                    use burn::serde as _serde;
                    #[automatically_derived]
                    impl<'de> _serde::Deserialize<'de> for DeformConv2dConfigSerde {
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private226::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            #[allow(non_camel_case_types)]
                            #[doc(hidden)]
                            enum __Field {
                                __field0,
                                __field1,
                                __field2,
                                __field3,
                                __field4,
                                __field5,
                                __field6,
                                __field7,
                                __field8,
                                __ignore,
                            }
                            #[doc(hidden)]
                            struct __FieldVisitor;
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                type Value = __Field;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private226::Formatter,
                                ) -> _serde::__private226::fmt::Result {
                                    _serde::__private226::Formatter::write_str(
                                        __formatter,
                                        "field identifier",
                                    )
                                }
                                fn visit_u64<__E>(
                                    self,
                                    __value: u64,
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        0u64 => _serde::__private226::Ok(__Field::__field0),
                                        1u64 => _serde::__private226::Ok(__Field::__field1),
                                        2u64 => _serde::__private226::Ok(__Field::__field2),
                                        3u64 => _serde::__private226::Ok(__Field::__field3),
                                        4u64 => _serde::__private226::Ok(__Field::__field4),
                                        5u64 => _serde::__private226::Ok(__Field::__field5),
                                        6u64 => _serde::__private226::Ok(__Field::__field6),
                                        7u64 => _serde::__private226::Ok(__Field::__field7),
                                        8u64 => _serde::__private226::Ok(__Field::__field8),
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_str<__E>(
                                    self,
                                    __value: &str,
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        "channels" => _serde::__private226::Ok(__Field::__field0),
                                        "kernel_size" => _serde::__private226::Ok(__Field::__field1),
                                        "stride" => _serde::__private226::Ok(__Field::__field2),
                                        "dilation" => _serde::__private226::Ok(__Field::__field3),
                                        "weight_groups" => {
                                            _serde::__private226::Ok(__Field::__field4)
                                        }
                                        "offset_groups" => {
                                            _serde::__private226::Ok(__Field::__field5)
                                        }
                                        "padding" => _serde::__private226::Ok(__Field::__field6),
                                        "bias" => _serde::__private226::Ok(__Field::__field7),
                                        "initializer" => _serde::__private226::Ok(__Field::__field8),
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                                fn visit_bytes<__E>(
                                    self,
                                    __value: &[u8],
                                ) -> _serde::__private226::Result<Self::Value, __E>
                                where
                                    __E: _serde::de::Error,
                                {
                                    match __value {
                                        b"channels" => _serde::__private226::Ok(__Field::__field0),
                                        b"kernel_size" => {
                                            _serde::__private226::Ok(__Field::__field1)
                                        }
                                        b"stride" => _serde::__private226::Ok(__Field::__field2),
                                        b"dilation" => _serde::__private226::Ok(__Field::__field3),
                                        b"weight_groups" => {
                                            _serde::__private226::Ok(__Field::__field4)
                                        }
                                        b"offset_groups" => {
                                            _serde::__private226::Ok(__Field::__field5)
                                        }
                                        b"padding" => _serde::__private226::Ok(__Field::__field6),
                                        b"bias" => _serde::__private226::Ok(__Field::__field7),
                                        b"initializer" => {
                                            _serde::__private226::Ok(__Field::__field8)
                                        }
                                        _ => _serde::__private226::Ok(__Field::__ignore),
                                    }
                                }
                            }
                            #[automatically_derived]
                            impl<'de> _serde::Deserialize<'de> for __Field {
                                #[inline]
                                fn deserialize<__D>(
                                    __deserializer: __D,
                                ) -> _serde::__private226::Result<Self, __D::Error>
                                where
                                    __D: _serde::Deserializer<'de>,
                                {
                                    _serde::Deserializer::deserialize_identifier(
                                        __deserializer,
                                        __FieldVisitor,
                                    )
                                }
                            }
                            #[doc(hidden)]
                            struct __Visitor<'de> {
                                marker: _serde::__private226::PhantomData<
                                    DeformConv2dConfigSerde,
                                >,
                                lifetime: _serde::__private226::PhantomData<&'de ()>,
                            }
                            #[automatically_derived]
                            impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                type Value = DeformConv2dConfigSerde;
                                fn expecting(
                                    &self,
                                    __formatter: &mut _serde::__private226::Formatter,
                                ) -> _serde::__private226::fmt::Result {
                                    _serde::__private226::Formatter::write_str(
                                        __formatter,
                                        "struct DeformConv2dConfigSerde",
                                    )
                                }
                                #[inline]
                                fn visit_seq<__A>(
                                    self,
                                    mut __seq: __A,
                                ) -> _serde::__private226::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::SeqAccess<'de>,
                                {
                                    let __field0 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    0usize,
                                                    &"struct DeformConv2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field1 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    1usize,
                                                    &"struct DeformConv2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field2 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    2usize,
                                                    &"struct DeformConv2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field3 = match _serde::de::SeqAccess::next_element::<
                                        [usize; 2],
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    3usize,
                                                    &"struct DeformConv2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field4 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    4usize,
                                                    &"struct DeformConv2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field5 = match _serde::de::SeqAccess::next_element::<
                                        usize,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    5usize,
                                                    &"struct DeformConv2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field6 = match _serde::de::SeqAccess::next_element::<
                                        PaddingConfig2d,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    6usize,
                                                    &"struct DeformConv2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field7 = match _serde::de::SeqAccess::next_element::<
                                        bool,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    7usize,
                                                    &"struct DeformConv2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    let __field8 = match _serde::de::SeqAccess::next_element::<
                                        Initializer,
                                    >(&mut __seq)? {
                                        _serde::__private226::Some(__value) => __value,
                                        _serde::__private226::None => {
                                            return _serde::__private226::Err(
                                                _serde::de::Error::invalid_length(
                                                    8usize,
                                                    &"struct DeformConv2dConfigSerde with 9 elements",
                                                ),
                                            );
                                        }
                                    };
                                    _serde::__private226::Ok(DeformConv2dConfigSerde {
                                        channels: __field0,
                                        kernel_size: __field1,
                                        stride: __field2,
                                        dilation: __field3,
                                        weight_groups: __field4,
                                        offset_groups: __field5,
                                        padding: __field6,
                                        bias: __field7,
                                        initializer: __field8,
                                    })
                                }
                                #[inline]
                                fn visit_map<__A>(
                                    self,
                                    mut __map: __A,
                                ) -> _serde::__private226::Result<Self::Value, __A::Error>
                                where
                                    __A: _serde::de::MapAccess<'de>,
                                {
                                    let mut __field0: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field1: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field2: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field3: _serde::__private226::Option<
                                        [usize; 2],
                                    > = _serde::__private226::None;
                                    let mut __field4: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field5: _serde::__private226::Option<usize> = _serde::__private226::None;
                                    let mut __field6: _serde::__private226::Option<
                                        PaddingConfig2d,
                                    > = _serde::__private226::None;
                                    let mut __field7: _serde::__private226::Option<bool> = _serde::__private226::None;
                                    let mut __field8: _serde::__private226::Option<
                                        Initializer,
                                    > = _serde::__private226::None;
                                    while let _serde::__private226::Some(__key) = _serde::de::MapAccess::next_key::<
                                        __Field,
                                    >(&mut __map)? {
                                        match __key {
                                            __Field::__field0 => {
                                                if _serde::__private226::Option::is_some(&__field0) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "channels",
                                                        ),
                                                    );
                                                }
                                                __field0 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field1 => {
                                                if _serde::__private226::Option::is_some(&__field1) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "kernel_size",
                                                        ),
                                                    );
                                                }
                                                __field1 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field2 => {
                                                if _serde::__private226::Option::is_some(&__field2) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                                    );
                                                }
                                                __field2 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field3 => {
                                                if _serde::__private226::Option::is_some(&__field3) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "dilation",
                                                        ),
                                                    );
                                                }
                                                __field3 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<[usize; 2]>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field4 => {
                                                if _serde::__private226::Option::is_some(&__field4) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "weight_groups",
                                                        ),
                                                    );
                                                }
                                                __field4 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field5 => {
                                                if _serde::__private226::Option::is_some(&__field5) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "offset_groups",
                                                        ),
                                                    );
                                                }
                                                __field5 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<usize>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field6 => {
                                                if _serde::__private226::Option::is_some(&__field6) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "padding",
                                                        ),
                                                    );
                                                }
                                                __field6 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<
                                                        PaddingConfig2d,
                                                    >(&mut __map)?,
                                                );
                                            }
                                            __Field::__field7 => {
                                                if _serde::__private226::Option::is_some(&__field7) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field("bias"),
                                                    );
                                                }
                                                __field7 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<bool>(&mut __map)?,
                                                );
                                            }
                                            __Field::__field8 => {
                                                if _serde::__private226::Option::is_some(&__field8) {
                                                    return _serde::__private226::Err(
                                                        <__A::Error as _serde::de::Error>::duplicate_field(
                                                            "initializer",
                                                        ),
                                                    );
                                                }
                                                __field8 = _serde::__private226::Some(
                                                    _serde::de::MapAccess::next_value::<
                                                        Initializer,
                                                    >(&mut __map)?,
                                                );
                                            }
                                            _ => {
                                                let _ = _serde::de::MapAccess::next_value::<
                                                    _serde::de::IgnoredAny,
                                                >(&mut __map)?;
                                            }
                                        }
                                    }
                                    let __field0 = match __field0 {
                                        _serde::__private226::Some(__field0) => __field0,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("channels")?
                                        }
                                    };
                                    let __field1 = match __field1 {
                                        _serde::__private226::Some(__field1) => __field1,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("kernel_size")?
                                        }
                                    };
                                    let __field2 = match __field2 {
                                        _serde::__private226::Some(__field2) => __field2,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("stride")?
                                        }
                                    };
                                    let __field3 = match __field3 {
                                        _serde::__private226::Some(__field3) => __field3,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("dilation")?
                                        }
                                    };
                                    let __field4 = match __field4 {
                                        _serde::__private226::Some(__field4) => __field4,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("weight_groups")?
                                        }
                                    };
                                    let __field5 = match __field5 {
                                        _serde::__private226::Some(__field5) => __field5,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("offset_groups")?
                                        }
                                    };
                                    let __field6 = match __field6 {
                                        _serde::__private226::Some(__field6) => __field6,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("padding")?
                                        }
                                    };
                                    let __field7 = match __field7 {
                                        _serde::__private226::Some(__field7) => __field7,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("bias")?
                                        }
                                    };
                                    let __field8 = match __field8 {
                                        _serde::__private226::Some(__field8) => __field8,
                                        _serde::__private226::None => {
                                            _serde::__private226::de::missing_field("initializer")?
                                        }
                                    };
                                    _serde::__private226::Ok(DeformConv2dConfigSerde {
                                        channels: __field0,
                                        kernel_size: __field1,
                                        stride: __field2,
                                        dilation: __field3,
                                        weight_groups: __field4,
                                        offset_groups: __field5,
                                        padding: __field6,
                                        bias: __field7,
                                        initializer: __field8,
                                    })
                                }
                            }
                            #[doc(hidden)]
                            const FIELDS: &'static [&'static str] = &[
                                "channels",
                                "kernel_size",
                                "stride",
                                "dilation",
                                "weight_groups",
                                "offset_groups",
                                "padding",
                                "bias",
                                "initializer",
                            ];
                            _serde::Deserializer::deserialize_struct(
                                __deserializer,
                                "DeformConv2dConfigSerde",
                                FIELDS,
                                __Visitor {
                                    marker: _serde::__private226::PhantomData::<
                                        DeformConv2dConfigSerde,
                                    >,
                                    lifetime: _serde::__private226::PhantomData,
                                },
                            )
                        }
                    }
                };
                let serde_state = DeformConv2dConfigSerde::deserialize(deserializer)?;
                Ok(DeformConv2dConfig {
                    channels: serde_state.channels,
                    kernel_size: serde_state.kernel_size,
                    stride: serde_state.stride,
                    dilation: serde_state.dilation,
                    weight_groups: serde_state.weight_groups,
                    offset_groups: serde_state.offset_groups,
                    padding: serde_state.padding,
                    bias: serde_state.bias,
                    initializer: serde_state.initializer,
                })
            }
        }
        impl Clone for DeformConv2dConfig {
            fn clone(&self) -> Self {
                Self {
                    channels: self.channels.clone(),
                    kernel_size: self.kernel_size.clone(),
                    stride: self.stride.clone(),
                    dilation: self.dilation.clone(),
                    weight_groups: self.weight_groups.clone(),
                    offset_groups: self.offset_groups.clone(),
                    padding: self.padding.clone(),
                    bias: self.bias.clone(),
                    initializer: self.initializer.clone(),
                }
            }
        }
        impl core::fmt::Display for DeformConv2dConfig {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(&burn::config::config_to_json(self))
            }
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for DeformConv2dConfig {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                let names: &'static _ = &[
                    "channels",
                    "kernel_size",
                    "stride",
                    "dilation",
                    "weight_groups",
                    "offset_groups",
                    "padding",
                    "bias",
                    "initializer",
                ];
                let values: &[&dyn ::core::fmt::Debug] = &[
                    &self.channels,
                    &self.kernel_size,
                    &self.stride,
                    &self.dilation,
                    &self.weight_groups,
                    &self.offset_groups,
                    &self.padding,
                    &self.bias,
                    &&self.initializer,
                ];
                ::core::fmt::Formatter::debug_struct_fields_finish(
                    f,
                    "DeformConv2dConfig",
                    names,
                    values,
                )
            }
        }
        /// Applies a deformable 2D convolution over input tensors.
        ///
        /// Should be created with [DeformConv2dConfig].
        #[module(custom_display)]
        pub struct DeformConv2d<B: Backend> {
            /// Tensor of shape `[channels_out, channels_in / groups, kernel_size_1, kernel_size_2]`
            pub weight: Param<Tensor<B, 4>>,
            /// Tensor of shape `[channels_out]`
            pub bias: Option<Param<Tensor<B, 1>>>,
            /// Stride of the convolution.
            pub stride: [usize; 2],
            /// Size of the kernel.
            pub kernel_size: [usize; 2],
            /// Spacing between kernel elements.
            pub dilation: [usize; 2],
            /// Controls the connections between input and output channels.
            pub weight_groups: usize,
            /// Offset groups.
            pub offset_groups: usize,
            /// The padding configuration.
            pub padding: Ignored<PaddingConfig2d>,
        }
        impl<B: Backend> burn::module::Module<B> for DeformConv2d<B> {
            type Record = DeformConv2dRecord<B>;
            fn load_record(self, record: Self::Record) -> Self {
                Self {
                    weight: burn::module::Module::<
                        B,
                    >::load_record(self.weight, record.weight),
                    bias: burn::module::Module::<B>::load_record(self.bias, record.bias),
                    stride: burn::module::Module::<
                        B,
                    >::load_record(self.stride, record.stride),
                    kernel_size: burn::module::Module::<
                        B,
                    >::load_record(self.kernel_size, record.kernel_size),
                    dilation: burn::module::Module::<
                        B,
                    >::load_record(self.dilation, record.dilation),
                    weight_groups: burn::module::Module::<
                        B,
                    >::load_record(self.weight_groups, record.weight_groups),
                    offset_groups: burn::module::Module::<
                        B,
                    >::load_record(self.offset_groups, record.offset_groups),
                    padding: burn::module::Module::<
                        B,
                    >::load_record(self.padding, record.padding),
                }
            }
            fn into_record(self) -> Self::Record {
                Self::Record {
                    weight: burn::module::Module::<B>::into_record(self.weight),
                    bias: burn::module::Module::<B>::into_record(self.bias),
                    stride: burn::module::Module::<B>::into_record(self.stride),
                    kernel_size: burn::module::Module::<
                        B,
                    >::into_record(self.kernel_size),
                    dilation: burn::module::Module::<B>::into_record(self.dilation),
                    weight_groups: burn::module::Module::<
                        B,
                    >::into_record(self.weight_groups),
                    offset_groups: burn::module::Module::<
                        B,
                    >::into_record(self.offset_groups),
                    padding: burn::module::Module::<B>::into_record(self.padding),
                }
            }
            fn num_params(&self) -> usize {
                let mut num_params = 0;
                num_params += burn::module::Module::<B>::num_params(&self.weight);
                num_params += burn::module::Module::<B>::num_params(&self.bias);
                num_params += burn::module::Module::<B>::num_params(&self.stride);
                num_params += burn::module::Module::<B>::num_params(&self.kernel_size);
                num_params += burn::module::Module::<B>::num_params(&self.dilation);
                num_params += burn::module::Module::<B>::num_params(&self.weight_groups);
                num_params += burn::module::Module::<B>::num_params(&self.offset_groups);
                num_params += burn::module::Module::<B>::num_params(&self.padding);
                num_params
            }
            fn visit<Visitor: burn::module::ModuleVisitor<B>>(
                &self,
                visitor: &mut Visitor,
            ) {
                visitor.enter_module("weight", "DeformConv2d");
                burn::module::Module::visit(&self.weight, visitor);
                visitor.exit_module("weight", "DeformConv2d");
                visitor.enter_module("bias", "DeformConv2d");
                burn::module::Module::visit(&self.bias, visitor);
                visitor.exit_module("bias", "DeformConv2d");
                visitor.enter_module("stride", "DeformConv2d");
                burn::module::Module::visit(&self.stride, visitor);
                visitor.exit_module("stride", "DeformConv2d");
                visitor.enter_module("kernel_size", "DeformConv2d");
                burn::module::Module::visit(&self.kernel_size, visitor);
                visitor.exit_module("kernel_size", "DeformConv2d");
                visitor.enter_module("dilation", "DeformConv2d");
                burn::module::Module::visit(&self.dilation, visitor);
                visitor.exit_module("dilation", "DeformConv2d");
                visitor.enter_module("weight_groups", "DeformConv2d");
                burn::module::Module::visit(&self.weight_groups, visitor);
                visitor.exit_module("weight_groups", "DeformConv2d");
                visitor.enter_module("offset_groups", "DeformConv2d");
                burn::module::Module::visit(&self.offset_groups, visitor);
                visitor.exit_module("offset_groups", "DeformConv2d");
                visitor.enter_module("padding", "DeformConv2d");
                burn::module::Module::visit(&self.padding, visitor);
                visitor.exit_module("padding", "DeformConv2d");
            }
            fn map<Mapper: burn::module::ModuleMapper<B>>(
                self,
                mapper: &mut Mapper,
            ) -> Self {
                mapper.enter_module("weight", "DeformConv2d");
                let weight = burn::module::Module::<B>::map(self.weight, mapper);
                mapper.exit_module("weight", "DeformConv2d");
                mapper.enter_module("bias", "DeformConv2d");
                let bias = burn::module::Module::<B>::map(self.bias, mapper);
                mapper.exit_module("bias", "DeformConv2d");
                mapper.enter_module("stride", "DeformConv2d");
                let stride = burn::module::Module::<B>::map(self.stride, mapper);
                mapper.exit_module("stride", "DeformConv2d");
                mapper.enter_module("kernel_size", "DeformConv2d");
                let kernel_size = burn::module::Module::<
                    B,
                >::map(self.kernel_size, mapper);
                mapper.exit_module("kernel_size", "DeformConv2d");
                mapper.enter_module("dilation", "DeformConv2d");
                let dilation = burn::module::Module::<B>::map(self.dilation, mapper);
                mapper.exit_module("dilation", "DeformConv2d");
                mapper.enter_module("weight_groups", "DeformConv2d");
                let weight_groups = burn::module::Module::<
                    B,
                >::map(self.weight_groups, mapper);
                mapper.exit_module("weight_groups", "DeformConv2d");
                mapper.enter_module("offset_groups", "DeformConv2d");
                let offset_groups = burn::module::Module::<
                    B,
                >::map(self.offset_groups, mapper);
                mapper.exit_module("offset_groups", "DeformConv2d");
                mapper.enter_module("padding", "DeformConv2d");
                let padding = burn::module::Module::<B>::map(self.padding, mapper);
                mapper.exit_module("padding", "DeformConv2d");
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    weight_groups,
                    offset_groups,
                    padding,
                }
            }
            fn collect_devices(
                &self,
                devices: burn::module::Devices<B>,
            ) -> burn::module::Devices<B> {
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.weight, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.bias, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.stride, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.kernel_size, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.dilation, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.weight_groups, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.offset_groups, devices);
                let devices = burn::module::Module::<
                    B,
                >::collect_devices(&self.padding, devices);
                devices
            }
            fn to_device(self, device: &B::Device) -> Self {
                let weight = burn::module::Module::<B>::to_device(self.weight, device);
                let bias = burn::module::Module::<B>::to_device(self.bias, device);
                let stride = burn::module::Module::<B>::to_device(self.stride, device);
                let kernel_size = burn::module::Module::<
                    B,
                >::to_device(self.kernel_size, device);
                let dilation = burn::module::Module::<
                    B,
                >::to_device(self.dilation, device);
                let weight_groups = burn::module::Module::<
                    B,
                >::to_device(self.weight_groups, device);
                let offset_groups = burn::module::Module::<
                    B,
                >::to_device(self.offset_groups, device);
                let padding = burn::module::Module::<B>::to_device(self.padding, device);
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    weight_groups,
                    offset_groups,
                    padding,
                }
            }
            fn fork(self, device: &B::Device) -> Self {
                let weight = burn::module::Module::<B>::fork(self.weight, device);
                let bias = burn::module::Module::<B>::fork(self.bias, device);
                let stride = burn::module::Module::<B>::fork(self.stride, device);
                let kernel_size = burn::module::Module::<
                    B,
                >::fork(self.kernel_size, device);
                let dilation = burn::module::Module::<B>::fork(self.dilation, device);
                let weight_groups = burn::module::Module::<
                    B,
                >::fork(self.weight_groups, device);
                let offset_groups = burn::module::Module::<
                    B,
                >::fork(self.offset_groups, device);
                let padding = burn::module::Module::<B>::fork(self.padding, device);
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    weight_groups,
                    offset_groups,
                    padding,
                }
            }
        }
        impl<B: Backend> burn::module::AutodiffModule<B> for DeformConv2d<B>
        where
            B: burn::tensor::backend::AutodiffBackend,
            <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
        {
            type InnerModule = DeformConv2d<B::InnerBackend>;
            fn valid(&self) -> Self::InnerModule {
                let weight = burn::module::AutodiffModule::<B>::valid(&self.weight);
                let bias = burn::module::AutodiffModule::<B>::valid(&self.bias);
                let stride = burn::module::AutodiffModule::<B>::valid(&self.stride);
                let kernel_size = burn::module::AutodiffModule::<
                    B,
                >::valid(&self.kernel_size);
                let dilation = burn::module::AutodiffModule::<B>::valid(&self.dilation);
                let weight_groups = burn::module::AutodiffModule::<
                    B,
                >::valid(&self.weight_groups);
                let offset_groups = burn::module::AutodiffModule::<
                    B,
                >::valid(&self.offset_groups);
                let padding = burn::module::AutodiffModule::<B>::valid(&self.padding);
                Self::InnerModule {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    weight_groups,
                    offset_groups,
                    padding,
                }
            }
        }
        impl<B: Backend> core::fmt::Display for DeformConv2d<B> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let formatted = burn::module::ModuleDisplay::format(
                    self,
                    Default::default(),
                );
                f.write_fmt(format_args!("{0}", formatted))
            }
        }
        impl<B: Backend> burn::module::ModuleDisplayDefault for DeformConv2d<B> {
            fn content(
                &self,
                mut content: burn::module::Content,
            ) -> Option<burn::module::Content> {
                content
                    .set_top_level_type(&"DeformConv2d")
                    .add("weight", &self.weight)
                    .add("bias", &self.bias)
                    .add("stride", &self.stride)
                    .add("kernel_size", &self.kernel_size)
                    .add("dilation", &self.dilation)
                    .add("weight_groups", &self.weight_groups)
                    .add("offset_groups", &self.offset_groups)
                    .add("padding", &self.padding)
                    .optional()
            }
            fn num_params(&self) -> usize {
                burn::module::Module::num_params(self)
            }
        }
        impl<B: Backend> Clone for DeformConv2d<B> {
            fn clone(&self) -> Self {
                let weight = self.weight.clone();
                let bias = self.bias.clone();
                let stride = self.stride.clone();
                let kernel_size = self.kernel_size.clone();
                let dilation = self.dilation.clone();
                let weight_groups = self.weight_groups.clone();
                let offset_groups = self.offset_groups.clone();
                let padding = self.padding.clone();
                Self {
                    weight,
                    bias,
                    stride,
                    kernel_size,
                    dilation,
                    weight_groups,
                    offset_groups,
                    padding,
                }
            }
        }
        /// The record type for the module.
        pub struct DeformConv2dRecord<B: Backend> {
            /// The module record associative type.
            pub weight: <Param<Tensor<B, 4>> as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub bias: <Option<Param<Tensor<B, 1>>> as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub stride: <[usize; 2] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub kernel_size: <[usize; 2] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub dilation: <[usize; 2] as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub weight_groups: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub offset_groups: <usize as burn::module::Module<B>>::Record,
            /// The module record associative type.
            pub padding: <Ignored<PaddingConfig2d> as burn::module::Module<B>>::Record,
        }
        /// The record item type for the module.
        #[serde(crate = "burn::serde")]
        #[serde(
            bound = "< < Param < Tensor < B, 4 > > as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize +\nburn :: serde :: de :: DeserializeOwned, < < Option < Param < Tensor < B, 1 >\n> > as burn :: module :: Module < B > > :: Record as burn :: record :: Record\n< B >> :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de ::\nDeserializeOwned, < < [usize; 2] as burn :: module :: Module < B > > :: Record\nas burn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize\n+ burn :: serde :: de :: DeserializeOwned, < < [usize; 2] as burn :: module ::\nModule < B > > :: Record as burn :: record :: Record < B >> :: Item < S > :\nburn :: serde :: Serialize + burn :: serde :: de :: DeserializeOwned, < <\n[usize; 2] as burn :: module :: Module < B > > :: Record as burn :: record ::\nRecord < B >> :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de\n:: DeserializeOwned, < < usize as burn :: module :: Module < B > > :: Record\nas burn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize\n+ burn :: serde :: de :: DeserializeOwned, < < usize as burn :: module ::\nModule < B > > :: Record as burn :: record :: Record < B >> :: Item < S > :\nburn :: serde :: Serialize + burn :: serde :: de :: DeserializeOwned, < <\nIgnored < PaddingConfig2d > as burn :: module :: Module < B > > :: Record as\nburn :: record :: Record < B >> :: Item < S > : burn :: serde :: Serialize +\nburn :: serde :: de :: DeserializeOwned,"
        )]
        pub struct DeformConv2dRecordItem<
            B: Backend,
            S: burn::record::PrecisionSettings,
        > {
            /// Field to be serialized.
            pub weight: <<Param<
                Tensor<B, 4>,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub bias: <<Option<
                Param<Tensor<B, 1>>,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub stride: <<[usize; 2] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub kernel_size: <<[usize; 2] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub dilation: <<[usize; 2] as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub weight_groups: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub offset_groups: <<usize as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
            /// Field to be serialized.
            pub padding: <<Ignored<
                PaddingConfig2d,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
        }
        #[doc(hidden)]
        #[allow(
            non_upper_case_globals,
            unused_attributes,
            unused_qualifications,
            clippy::absolute_paths,
        )]
        const _: () = {
            use burn::serde as _serde;
            #[automatically_derived]
            impl<B: Backend, S: burn::record::PrecisionSettings> _serde::Serialize
            for DeformConv2dRecordItem<B, S>
            where
                <<Param<
                    Tensor<B, 4>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Option<
                    Param<Tensor<B, 1>>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Ignored<
                    PaddingConfig2d,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            {
                fn serialize<__S>(
                    &self,
                    __serializer: __S,
                ) -> _serde::__private226::Result<__S::Ok, __S::Error>
                where
                    __S: _serde::Serializer,
                {
                    let mut __serde_state = _serde::Serializer::serialize_struct(
                        __serializer,
                        "DeformConv2dRecordItem",
                        false as usize + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "weight",
                        &self.weight,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "bias",
                        &self.bias,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "stride",
                        &self.stride,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "kernel_size",
                        &self.kernel_size,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "dilation",
                        &self.dilation,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "weight_groups",
                        &self.weight_groups,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "offset_groups",
                        &self.offset_groups,
                    )?;
                    _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "padding",
                        &self.padding,
                    )?;
                    _serde::ser::SerializeStruct::end(__serde_state)
                }
            }
        };
        #[doc(hidden)]
        #[allow(
            non_upper_case_globals,
            unused_attributes,
            unused_qualifications,
            clippy::absolute_paths,
        )]
        const _: () = {
            use burn::serde as _serde;
            #[automatically_derived]
            impl<
                'de,
                B: Backend,
                S: burn::record::PrecisionSettings,
            > _serde::Deserialize<'de> for DeformConv2dRecordItem<B, S>
            where
                <<Param<
                    Tensor<B, 4>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Option<
                    Param<Tensor<B, 1>>,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<[usize; 2] as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<usize as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                <<Ignored<
                    PaddingConfig2d,
                > as burn::module::Module<
                    B,
                >>::Record as burn::record::Record<
                    B,
                >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            {
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private226::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    #[allow(non_camel_case_types)]
                    #[doc(hidden)]
                    enum __Field {
                        __field0,
                        __field1,
                        __field2,
                        __field3,
                        __field4,
                        __field5,
                        __field6,
                        __field7,
                        __ignore,
                    }
                    #[doc(hidden)]
                    struct __FieldVisitor;
                    #[automatically_derived]
                    impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                        type Value = __Field;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private226::Formatter,
                        ) -> _serde::__private226::fmt::Result {
                            _serde::__private226::Formatter::write_str(
                                __formatter,
                                "field identifier",
                            )
                        }
                        fn visit_u64<__E>(
                            self,
                            __value: u64,
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                0u64 => _serde::__private226::Ok(__Field::__field0),
                                1u64 => _serde::__private226::Ok(__Field::__field1),
                                2u64 => _serde::__private226::Ok(__Field::__field2),
                                3u64 => _serde::__private226::Ok(__Field::__field3),
                                4u64 => _serde::__private226::Ok(__Field::__field4),
                                5u64 => _serde::__private226::Ok(__Field::__field5),
                                6u64 => _serde::__private226::Ok(__Field::__field6),
                                7u64 => _serde::__private226::Ok(__Field::__field7),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_str<__E>(
                            self,
                            __value: &str,
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                "weight" => _serde::__private226::Ok(__Field::__field0),
                                "bias" => _serde::__private226::Ok(__Field::__field1),
                                "stride" => _serde::__private226::Ok(__Field::__field2),
                                "kernel_size" => _serde::__private226::Ok(__Field::__field3),
                                "dilation" => _serde::__private226::Ok(__Field::__field4),
                                "weight_groups" => {
                                    _serde::__private226::Ok(__Field::__field5)
                                }
                                "offset_groups" => {
                                    _serde::__private226::Ok(__Field::__field6)
                                }
                                "padding" => _serde::__private226::Ok(__Field::__field7),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_bytes<__E>(
                            self,
                            __value: &[u8],
                        ) -> _serde::__private226::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                b"weight" => _serde::__private226::Ok(__Field::__field0),
                                b"bias" => _serde::__private226::Ok(__Field::__field1),
                                b"stride" => _serde::__private226::Ok(__Field::__field2),
                                b"kernel_size" => {
                                    _serde::__private226::Ok(__Field::__field3)
                                }
                                b"dilation" => _serde::__private226::Ok(__Field::__field4),
                                b"weight_groups" => {
                                    _serde::__private226::Ok(__Field::__field5)
                                }
                                b"offset_groups" => {
                                    _serde::__private226::Ok(__Field::__field6)
                                }
                                b"padding" => _serde::__private226::Ok(__Field::__field7),
                                _ => _serde::__private226::Ok(__Field::__ignore),
                            }
                        }
                    }
                    #[automatically_derived]
                    impl<'de> _serde::Deserialize<'de> for __Field {
                        #[inline]
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private226::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            _serde::Deserializer::deserialize_identifier(
                                __deserializer,
                                __FieldVisitor,
                            )
                        }
                    }
                    #[doc(hidden)]
                    struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                    where
                        <<Param<
                            Tensor<B, 4>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Option<
                            Param<Tensor<B, 1>>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Ignored<
                            PaddingConfig2d,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    {
                        marker: _serde::__private226::PhantomData<
                            DeformConv2dRecordItem<B, S>,
                        >,
                        lifetime: _serde::__private226::PhantomData<&'de ()>,
                    }
                    #[automatically_derived]
                    impl<
                        'de,
                        B: Backend,
                        S: burn::record::PrecisionSettings,
                    > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                    where
                        <<Param<
                            Tensor<B, 4>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Option<
                            Param<Tensor<B, 1>>,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<[usize; 2] as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<usize as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                        <<Ignored<
                            PaddingConfig2d,
                        > as burn::module::Module<
                            B,
                        >>::Record as burn::record::Record<
                            B,
                        >>::Item<
                            S,
                        >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    {
                        type Value = DeformConv2dRecordItem<B, S>;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private226::Formatter,
                        ) -> _serde::__private226::fmt::Result {
                            _serde::__private226::Formatter::write_str(
                                __formatter,
                                "struct DeformConv2dRecordItem",
                            )
                        }
                        #[inline]
                        fn visit_seq<__A>(
                            self,
                            mut __seq: __A,
                        ) -> _serde::__private226::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::SeqAccess<'de>,
                        {
                            let __field0 = match _serde::de::SeqAccess::next_element::<
                                <<Param<
                                    Tensor<B, 4>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            0usize,
                                            &"struct DeformConv2dRecordItem with 8 elements",
                                        ),
                                    );
                                }
                            };
                            let __field1 = match _serde::de::SeqAccess::next_element::<
                                <<Option<
                                    Param<Tensor<B, 1>>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            1usize,
                                            &"struct DeformConv2dRecordItem with 8 elements",
                                        ),
                                    );
                                }
                            };
                            let __field2 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            2usize,
                                            &"struct DeformConv2dRecordItem with 8 elements",
                                        ),
                                    );
                                }
                            };
                            let __field3 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            3usize,
                                            &"struct DeformConv2dRecordItem with 8 elements",
                                        ),
                                    );
                                }
                            };
                            let __field4 = match _serde::de::SeqAccess::next_element::<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            4usize,
                                            &"struct DeformConv2dRecordItem with 8 elements",
                                        ),
                                    );
                                }
                            };
                            let __field5 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            5usize,
                                            &"struct DeformConv2dRecordItem with 8 elements",
                                        ),
                                    );
                                }
                            };
                            let __field6 = match _serde::de::SeqAccess::next_element::<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            6usize,
                                            &"struct DeformConv2dRecordItem with 8 elements",
                                        ),
                                    );
                                }
                            };
                            let __field7 = match _serde::de::SeqAccess::next_element::<
                                <<Ignored<
                                    PaddingConfig2d,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            >(&mut __seq)? {
                                _serde::__private226::Some(__value) => __value,
                                _serde::__private226::None => {
                                    return _serde::__private226::Err(
                                        _serde::de::Error::invalid_length(
                                            7usize,
                                            &"struct DeformConv2dRecordItem with 8 elements",
                                        ),
                                    );
                                }
                            };
                            _serde::__private226::Ok(DeformConv2dRecordItem {
                                weight: __field0,
                                bias: __field1,
                                stride: __field2,
                                kernel_size: __field3,
                                dilation: __field4,
                                weight_groups: __field5,
                                offset_groups: __field6,
                                padding: __field7,
                            })
                        }
                        #[inline]
                        fn visit_map<__A>(
                            self,
                            mut __map: __A,
                        ) -> _serde::__private226::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::MapAccess<'de>,
                        {
                            let mut __field0: _serde::__private226::Option<
                                <<Param<
                                    Tensor<B, 4>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field1: _serde::__private226::Option<
                                <<Option<
                                    Param<Tensor<B, 1>>,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field2: _serde::__private226::Option<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field3: _serde::__private226::Option<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field4: _serde::__private226::Option<
                                <<[usize; 2] as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field5: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field6: _serde::__private226::Option<
                                <<usize as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            let mut __field7: _serde::__private226::Option<
                                <<Ignored<
                                    PaddingConfig2d,
                                > as burn::module::Module<
                                    B,
                                >>::Record as burn::record::Record<B>>::Item<S>,
                            > = _serde::__private226::None;
                            while let _serde::__private226::Some(__key) = _serde::de::MapAccess::next_key::<
                                __Field,
                            >(&mut __map)? {
                                match __key {
                                    __Field::__field0 => {
                                        if _serde::__private226::Option::is_some(&__field0) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("weight"),
                                            );
                                        }
                                        __field0 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Param<
                                                    Tensor<B, 4>,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field1 => {
                                        if _serde::__private226::Option::is_some(&__field1) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("bias"),
                                            );
                                        }
                                        __field1 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Option<
                                                    Param<Tensor<B, 1>>,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field2 => {
                                        if _serde::__private226::Option::is_some(&__field2) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field("stride"),
                                            );
                                        }
                                        __field2 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 2] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field3 => {
                                        if _serde::__private226::Option::is_some(&__field3) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "kernel_size",
                                                ),
                                            );
                                        }
                                        __field3 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 2] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field4 => {
                                        if _serde::__private226::Option::is_some(&__field4) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "dilation",
                                                ),
                                            );
                                        }
                                        __field4 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<[usize; 2] as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field5 => {
                                        if _serde::__private226::Option::is_some(&__field5) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "weight_groups",
                                                ),
                                            );
                                        }
                                        __field5 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field6 => {
                                        if _serde::__private226::Option::is_some(&__field6) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "offset_groups",
                                                ),
                                            );
                                        }
                                        __field6 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<usize as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    __Field::__field7 => {
                                        if _serde::__private226::Option::is_some(&__field7) {
                                            return _serde::__private226::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "padding",
                                                ),
                                            );
                                        }
                                        __field7 = _serde::__private226::Some(
                                            _serde::de::MapAccess::next_value::<
                                                <<Ignored<
                                                    PaddingConfig2d,
                                                > as burn::module::Module<
                                                    B,
                                                >>::Record as burn::record::Record<B>>::Item<S>,
                                            >(&mut __map)?,
                                        );
                                    }
                                    _ => {
                                        let _ = _serde::de::MapAccess::next_value::<
                                            _serde::de::IgnoredAny,
                                        >(&mut __map)?;
                                    }
                                }
                            }
                            let __field0 = match __field0 {
                                _serde::__private226::Some(__field0) => __field0,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("weight")?
                                }
                            };
                            let __field1 = match __field1 {
                                _serde::__private226::Some(__field1) => __field1,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("bias")?
                                }
                            };
                            let __field2 = match __field2 {
                                _serde::__private226::Some(__field2) => __field2,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("stride")?
                                }
                            };
                            let __field3 = match __field3 {
                                _serde::__private226::Some(__field3) => __field3,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("kernel_size")?
                                }
                            };
                            let __field4 = match __field4 {
                                _serde::__private226::Some(__field4) => __field4,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("dilation")?
                                }
                            };
                            let __field5 = match __field5 {
                                _serde::__private226::Some(__field5) => __field5,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("weight_groups")?
                                }
                            };
                            let __field6 = match __field6 {
                                _serde::__private226::Some(__field6) => __field6,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("offset_groups")?
                                }
                            };
                            let __field7 = match __field7 {
                                _serde::__private226::Some(__field7) => __field7,
                                _serde::__private226::None => {
                                    _serde::__private226::de::missing_field("padding")?
                                }
                            };
                            _serde::__private226::Ok(DeformConv2dRecordItem {
                                weight: __field0,
                                bias: __field1,
                                stride: __field2,
                                kernel_size: __field3,
                                dilation: __field4,
                                weight_groups: __field5,
                                offset_groups: __field6,
                                padding: __field7,
                            })
                        }
                    }
                    #[doc(hidden)]
                    const FIELDS: &'static [&'static str] = &[
                        "weight",
                        "bias",
                        "stride",
                        "kernel_size",
                        "dilation",
                        "weight_groups",
                        "offset_groups",
                        "padding",
                    ];
                    _serde::Deserializer::deserialize_struct(
                        __deserializer,
                        "DeformConv2dRecordItem",
                        FIELDS,
                        __Visitor {
                            marker: _serde::__private226::PhantomData::<
                                DeformConv2dRecordItem<B, S>,
                            >,
                            lifetime: _serde::__private226::PhantomData,
                        },
                    )
                }
            }
        };
        impl<B: Backend, S: burn::record::PrecisionSettings> Clone
        for DeformConv2dRecordItem<B, S> {
            fn clone(&self) -> Self {
                Self {
                    weight: self.weight.clone(),
                    bias: self.bias.clone(),
                    stride: self.stride.clone(),
                    kernel_size: self.kernel_size.clone(),
                    dilation: self.dilation.clone(),
                    weight_groups: self.weight_groups.clone(),
                    offset_groups: self.offset_groups.clone(),
                    padding: self.padding.clone(),
                }
            }
        }
        impl<B: Backend> burn::record::Record<B> for DeformConv2dRecord<B> {
            type Item<S: burn::record::PrecisionSettings> = DeformConv2dRecordItem<B, S>;
            fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
                DeformConv2dRecordItem {
                    weight: burn::record::Record::<B>::into_item::<S>(self.weight),
                    bias: burn::record::Record::<B>::into_item::<S>(self.bias),
                    stride: burn::record::Record::<B>::into_item::<S>(self.stride),
                    kernel_size: burn::record::Record::<
                        B,
                    >::into_item::<S>(self.kernel_size),
                    dilation: burn::record::Record::<B>::into_item::<S>(self.dilation),
                    weight_groups: burn::record::Record::<
                        B,
                    >::into_item::<S>(self.weight_groups),
                    offset_groups: burn::record::Record::<
                        B,
                    >::into_item::<S>(self.offset_groups),
                    padding: burn::record::Record::<B>::into_item::<S>(self.padding),
                }
            }
            fn from_item<S: burn::record::PrecisionSettings>(
                item: Self::Item<S>,
                device: &B::Device,
            ) -> Self {
                Self {
                    weight: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.weight, device),
                    bias: burn::record::Record::<B>::from_item::<S>(item.bias, device),
                    stride: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.stride, device),
                    kernel_size: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.kernel_size, device),
                    dilation: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.dilation, device),
                    weight_groups: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.weight_groups, device),
                    offset_groups: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.offset_groups, device),
                    padding: burn::record::Record::<
                        B,
                    >::from_item::<S>(item.padding, device),
                }
            }
        }
        #[automatically_derived]
        impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for DeformConv2d<B> {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                let names: &'static _ = &[
                    "weight",
                    "bias",
                    "stride",
                    "kernel_size",
                    "dilation",
                    "weight_groups",
                    "offset_groups",
                    "padding",
                ];
                let values: &[&dyn ::core::fmt::Debug] = &[
                    &self.weight,
                    &self.bias,
                    &self.stride,
                    &self.kernel_size,
                    &self.dilation,
                    &self.weight_groups,
                    &self.offset_groups,
                    &&self.padding,
                ];
                ::core::fmt::Formatter::debug_struct_fields_finish(
                    f,
                    "DeformConv2d",
                    names,
                    values,
                )
            }
        }
        impl DeformConv2dConfig {
            /// Initialize a new [DeformConv2d](DeformConv2d) module.
            pub fn init<B: Backend>(&self, device: &B::Device) -> DeformConv2d<B> {
                checks::checks_channels_div_groups(
                    self.channels[0],
                    self.channels[1],
                    self.weight_groups,
                );
                if self.padding == PaddingConfig2d::Same {
                    checks::check_same_padding_support(&self.kernel_size);
                }
                let shape = [
                    self.channels[1],
                    self.channels[0] / self.weight_groups,
                    self.kernel_size[0],
                    self.kernel_size[1],
                ];
                let k = self.kernel_size.iter().product::<usize>();
                let fan_in = self.channels[0] / self.weight_groups * k;
                let fan_out = self.channels[1] / self.weight_groups * k;
                let weight = self
                    .initializer
                    .init_with(shape, Some(fan_in), Some(fan_out), device);
                let mut bias = None;
                if self.bias {
                    bias = Some(
                        self
                            .initializer
                            .init_with(
                                [self.channels[1]],
                                Some(fan_in),
                                Some(fan_out),
                                device,
                            ),
                    );
                }
                DeformConv2d {
                    weight,
                    bias,
                    stride: self.stride,
                    kernel_size: self.kernel_size,
                    dilation: self.dilation,
                    padding: Ignored(self.padding.clone()),
                    weight_groups: self.weight_groups,
                    offset_groups: self.weight_groups,
                }
            }
        }
        impl<B: Backend> ModuleDisplay for DeformConv2d<B> {
            fn custom_settings(&self) -> Option<DisplaySettings> {
                DisplaySettings::new().with_new_line_after_attribute(false).optional()
            }
            fn custom_content(&self, content: Content) -> Option<Content> {
                let padding_formatted = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0}", &self.padding))
                });
                let stride = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", self.stride))
                });
                let kernel_size = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", self.kernel_size))
                });
                let dilation = ::alloc::__export::must_use({
                    ::alloc::fmt::format(format_args!("{0:?}", self.dilation))
                });
                content
                    .add("stride", &stride)
                    .add("kernel_size", &kernel_size)
                    .add("dilation", &dilation)
                    .add("weight_groups", &self.weight_groups)
                    .add("offset_groups", &self.offset_groups)
                    .add("padding", &padding_formatted)
                    .optional()
            }
        }
        impl<B: Backend> DeformConv2d<B> {
            /// Applies the forward pass on the input tensor.
            ///
            /// See [deform_conv2d](burn::tensor::module::deform_conv2d) for more information.
            ///
            /// # Shapes
            ///
            /// - input: `[batch_size, channels_in, height_in, width_in]`
            /// - offset: `[batch_size, 2 * offset_groups * kernel_height * kernel_width, height_out, width_out]`
            /// - mask: `[batch_size, offset_groups * kernel_height * kernel_width, height_out, width_out]`
            /// - output: `[batch_size, channels_out, height_out, width_out]`
            pub fn forward(
                &self,
                input: Tensor<B, 4>,
                offset: Tensor<B, 4>,
                mask: Option<Tensor<B, 4>>,
            ) -> Tensor<B, 4> {
                let [_batch_size, _channels_in, height_in, width_in] = input.dims();
                let padding = self
                    .padding
                    .calculate_padding_2d(
                        height_in,
                        width_in,
                        &self.kernel_size,
                        &self.stride,
                    );
                deform_conv2d(
                    input,
                    offset,
                    self.weight.val(),
                    mask,
                    self.bias.as_ref().map(|bias| bias.val()),
                    DeformConvOptions::new(
                        self.stride,
                        padding,
                        self.dilation,
                        self.weight_groups,
                        self.offset_groups,
                    ),
                )
            }
        }
    }
    pub(crate) mod checks {
        pub(crate) fn checks_channels_div_groups(
            channels_in: usize,
            channels_out: usize,
            groups: usize,
        ) {
            let channels_in_div_by_group = channels_in.is_multiple_of(groups);
            let channels_out_div_by_group = channels_out.is_multiple_of(groups);
            if !channels_in_div_by_group || !channels_out_div_by_group {
                {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "Both channels must be divisible by the number of groups. Got channels_in={0}, channels_out={1}, groups={2}",
                            channels_in,
                            channels_out,
                            groups,
                        ),
                    );
                };
            }
        }
        /// Only symmetric padding is currently supported. As such, using `Same` padding with an even kernel
        /// size is not supported as it will not produce the same output size.
        pub(crate) fn check_same_padding_support(kernel_size: &[usize]) {
            for k in kernel_size.iter() {
                if k % 2 == 0 {
                    {
                        ::core::panicking::panic_fmt(
                            format_args!(
                                "not implemented: {0}",
                                format_args!(
                                    "Same padding with an even kernel size is not supported",
                                ),
                            ),
                        );
                    };
                }
            }
        }
    }
    pub use conv_transpose1d::*;
    pub use conv_transpose2d::*;
    pub use conv_transpose3d::*;
    pub use conv1d::*;
    pub use conv2d::*;
    pub use conv3d::*;
    pub use deform_conv2d::*;
}
