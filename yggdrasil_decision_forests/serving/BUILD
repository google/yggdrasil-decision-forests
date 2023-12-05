load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Engine groups
# ==============

# All the canonical inference engines.
cc_library_ydf(
    name = "all_inference_engines",
    deps = ["//yggdrasil_decision_forests/serving/decision_forest:register_engines"],
)

# Libraries
# =========

cc_library_ydf(
    name = "fast_engine",
    hdrs = [
        "fast_engine.h",
    ],
    deps = [
        ":example_set",
    ],
)

# TODO: Remove dependency to TF in this rule.
cc_library_ydf(
    name = "example_set",
    srcs = ["example_set.cc"],
    hdrs = ["example_set.h"],
    deps = [
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library_ydf(
    name = "tf_example",
    srcs = ["tf_example.cc"],
    hdrs = ["tf_example.h"],
    deps = [
        ":example_set",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset/tensorflow_no_dep:tf_example",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ] + select({
        "//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ],
        "//conditions:default": [],
    }),
)

cc_library_ydf(
    name = "example_set_model_wrapper",
    hdrs = [
        "example_set_model_wrapper.h",
    ],
    deps = [
        ":example_set",
        ":fast_engine",
        "//yggdrasil_decision_forests/model:abstract_model",
        "@com_google_absl//absl/status",
    ],
)

cc_library_ydf(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "utils.h",
    ],
    deps = [
        ":serving_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/utils:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

# Proto
# =====

all_proto_library(
    name = "serving_proto",
    srcs = ["serving.proto"],
)

# Tests
# =====

cc_test(
    name = "example_set_test",
    srcs = ["example_set_test.cc"],
    deps = [
        ":example_set",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tf_example_test",
    srcs = ["tf_example_test.cc"],
    deps = [
        ":example_set",
        ":tf_example",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset/tensorflow_no_dep:tf_example",
        "//yggdrasil_decision_forests/utils:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    data = ["//yggdrasil_decision_forests/test_data"],
    deps = [
        ":utils",
        "//yggdrasil_decision_forests/dataset:csv_example_reader",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/serving/decision_forest",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
        "@com_google_googletest//:gtest_main",
    ],
)
