load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library_ydf(
    name = "smoothed_pav_calibrator",
    srcs = [
        "smoothed_pav_calibrator.cc",
    ],
    hdrs = [
        "smoothed_pav_calibrator.h",
    ],
    deps = [
        ":smoothed_pav_calibrator_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/model:prediction_cc_proto",
        "//yggdrasil_decision_forests/model/postprocessor:abstract_postprocessor",
        "//yggdrasil_decision_forests/model/postprocessor:abstract_postprocessor_cc_proto",
        "//yggdrasil_decision_forests/utils:smoothed_pav_calibration_fit",
        "//yggdrasil_decision_forests/utils:smoothed_pav_calibration_inference",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

all_proto_library(
    name = "smoothed_pav_calibrator_proto",
    srcs = ["smoothed_pav_calibrator.proto"],
)

cc_test(
    name = "smoothed_pav_calibrator_test",
    srcs = ["smoothed_pav_calibrator_test.cc"],
    deps = [
        ":smoothed_pav_calibrator",
        ":smoothed_pav_calibrator_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/model:prediction_cc_proto",
        "//yggdrasil_decision_forests/model/postprocessor:abstract_postprocessor_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)
