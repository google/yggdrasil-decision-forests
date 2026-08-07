load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Proto
# ========

all_proto_library(
    name = "smoothed_pav_calibrator_proto",
    srcs = ["smoothed_pav_calibrator.proto"],
)

# Library
# ========

cc_library_ydf(
    name = "smoothed_pav_calibrator",
    srcs = ["smoothed_pav_calibrator.cc"],
    hdrs = ["smoothed_pav_calibrator.h"],
    deps = [
        ":smoothed_pav_calibrator_cc_proto",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model/postprocessor:abstract_postprocessor",
        "//yggdrasil_decision_forests/model/postprocessor/smoothed_pav_calibrator",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:smoothed_pav_calibration_fit",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils:synchronization_primitives",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "smoothed_pav_calibrator_test",
    srcs = ["smoothed_pav_calibrator_test.cc"],
    deps = [
        ":smoothed_pav_calibrator",
        ":smoothed_pav_calibrator_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/model:model_testing",
        "//yggdrasil_decision_forests/model:prediction_cc_proto",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)
