load("//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Library
# =======

cc_library_ydf(
    name = "inference",
    srcs = ["inference.cc"],
    hdrs = ["inference.h"],
    deps = [
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/serving:example_set",
        "//yggdrasil_decision_forests/serving:fast_engine",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
    ],
)

# Tests
# =====

cc_test(
    name = "inference_test",
    srcs = ["inference_test.cc"],
    data = ["//yggdrasil_decision_forests/test_data"],
    deps = [
        ":inference",
        "//yggdrasil_decision_forests/dataset:csv_example_reader",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/model/random_forest",
        "//yggdrasil_decision_forests/serving/decision_forest:register_engines",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)
