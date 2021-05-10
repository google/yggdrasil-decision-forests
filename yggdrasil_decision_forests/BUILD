
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

# Use the TensorFlow IO library instead of the default one (relying on std).
# This option is required for < c++17.
config_setting(
    name = "use_tensorflow_io",
    values = {"define": "use_tensorflow_io=1"},
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

# If set, don't use absl's version of StatusOr.
config_setting(
    name = "no_absl_statusor",
    values = {"define": "no_absl_statusor=1"},
)
