package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

# If set, use YDF own copy of TensorFlow proto.
config_setting(
    name = "use_ydf_tensorflow_proto",
    values = {"define": "use_ydf_tensorflow_proto=1"},
)

# Use the TensorFlow IO library instead of the default one (relying on std).
# This option is required for < c++17.
config_setting(
    name = "use_tensorflow_io",
    values = {"define": "use_tensorflow_io=1"},
)

# TF-nightly uses absl::anyinvocable, while stable TF uses an Abseil version that does not use it.
config_setting(
    name = "use_absl_anyinvocable",
    values = {"define": "use_absl_anyinvocable=1"},
)

# TensorFlow is used for reading TFRecord dataset, and serializing models (if
# TFRecord is selected as the format container). If
# tensorflow_with_header_lib=1, TensorFlow is linked with "head only"
# dependencies. This option make sense when injecting Yggdrasil as a dynamic
# library in TensorFlow.
config_setting(
    name = "tensorflow_with_header_lib",
    values = {"define": "tensorflow_with_header_lib=1"},
)

# If not set (default), use absl concurrency primitives (e.g. absl::Mutex).
# If set, use the std concurrency primitives (e.g. std::mutex)
config_setting(
    name = "std_synchronization_primitives",
    values = {"define": "std_synchronization_primitives=1"},
)

# Webassembly compilation.
config_setting(
    name = "wasm_target",
    values = {"cpu": "wasm"},
)

# Number of bits required to represent the number of examples in a training
# dataset for "{Unsigned,Signed}ExampleIdx" in "learner/types.h".
#
# Learning algorithms are not required to depend on this type. In such case,
# this parameter has no effects.
#
# For learning algorithm that depend on this type, a smaller value can reduce
# the RAM usage and increase the speed of training speed. If the number of
# training examples exceed this value, the training with fail with a
# "absl::InvalidArgumentError" error. In some cases, a learning algorithm that
# depends on {}ExampleIdx can support more (but not less) training examples
# (e.g. if the algorithm use some sampling, or if the training algorithm
# contains zero-weighted examples).
#
# Possible values:
#   (default) --define=ydf_example_idx_num_bits=32 => Up to 2'147'483'647 training
#     examples.
#   --define=ydf_example_idx_num_bits=64 => Up to 10^19 training examples.
config_setting(
    name = "example_idx_32bits",
    values = {"define": "ydf_example_idx_num_bits=32"},
)

config_setting(
    name = "example_idx_64bits",
    values = {"define": "ydf_example_idx_num_bits=64"},
)
