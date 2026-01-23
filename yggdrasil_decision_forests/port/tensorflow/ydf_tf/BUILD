load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# TF Ops
# ======

# Op+kernel
cc_library(
    name = "kernel_and_op",
    visibility = ["//visibility:public"],
    deps = [
        ":kernel",
        ":op",
        "@ydf_cc//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/model/multitasker",
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest",
        "@ydf_cc//yggdrasil_decision_forests/serving/decision_forest:register_engines",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "inference.so",
    linkshared = 1,
    deps = [":kernel_and_op"],
)

py_library(
    name = "ydf_tf",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [":api"],
)

# User-facing library to run models in TensorFlow.
py_library(
    name = "api",
    srcs = ["api.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":op_py",
        # absl/logging dep,
        # six dep,
        # tensorflow dep,
        "@tensorflow//tensorflow/python/eager:context",
        "@tensorflow//tensorflow/python/framework:ops",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
    ],
)

# Declaration of the ops.
OP_SRCS = ["op.cc"]

OP_DEPS = select({
    "@ydf_cc//yggdrasil_decision_forests:tensorflow_with_header_lib": [
        "@local_config_tf//:tensorflow",
    ],
    "//conditions:default": ["@tensorflow//tensorflow/core:framework"],
})

cc_library(
    name = "op",
    srcs = OP_SRCS,
    linkstatic = 1,
    deps = OP_DEPS + [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/utility:if_constexpr",
        "@ydf_cc//yggdrasil_decision_forests/utils:status_macros",
    ],
    alwayslink = 1,
)

# Definition of the ops i.e. the kernels.
cc_library(
    name = "kernel",
    srcs = ["kernel.cc"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:vertical_dataset",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model",
        "@ydf_cc//yggdrasil_decision_forests/model:model_library",
        "@ydf_cc//yggdrasil_decision_forests/model/decision_tree:decision_forest_interface",
        "@ydf_cc//yggdrasil_decision_forests/utils:compatibility",
        "@ydf_cc//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:filesystem_tensorflow_impl",  # keep
        "@ydf_cc//yggdrasil_decision_forests/utils:status_macros",
    ] + select({
        "@ydf_cc//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@local_config_tf//:tensorflow",
        ],
        "//conditions:default": ["@tensorflow//tensorflow/core:framework"],
    }),
    alwayslink = 1,
)
