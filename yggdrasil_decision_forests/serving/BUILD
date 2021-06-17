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

cc_library_ydf(
    name = "example_set",
    srcs = [
        "example_set.cc",
    ],
    hdrs = [
        "example_set.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
)

cc_library_ydf(
    name = "example_set_model_wrapper",
    hdrs = [
        "example_set_model_wrapper.h",
    ],
    deps = [
        ":example_set",
        ":fast_engine",
        "@com_google_absl//absl/status",
        "//yggdrasil_decision_forests/model:abstract_model",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/utils:logging",
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
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core/example:feature_util",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/utils:test",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    data = ["//yggdrasil_decision_forests/test_data"],
    deps = [
        ":utils",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/dataset:csv_example_reader",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/serving/decision_forest",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
    ],
)
