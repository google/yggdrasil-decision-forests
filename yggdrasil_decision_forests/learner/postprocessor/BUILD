load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Proto
# ========

all_proto_library(
    name = "abstract_postprocessor_proto",
    srcs = ["abstract_postprocessor.proto"],
    deps = [
        "//yggdrasil_decision_forests/learner/postprocessor/smoothed_pav_calibrator:smoothed_pav_calibrator_proto",
    ],
)

# Library
# ========

cc_library_ydf(
    name = "postprocessor_library",
    srcs = ["postprocessor_library.cc"],
    hdrs = ["postprocessor_library.h"],
    deps = [
        ":abstract_postprocessor_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/postprocessor/smoothed_pav_calibrator",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model/postprocessor:abstract_postprocessor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "postprocessor_library_test",
    srcs = ["postprocessor_library_test.cc"],
    deps = [
        ":abstract_postprocessor_cc_proto",
        ":postprocessor_library",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/postprocessor/smoothed_pav_calibrator",
        "//yggdrasil_decision_forests/model:model_testing",
        "//yggdrasil_decision_forests/model:prediction_cc_proto",
        "//yggdrasil_decision_forests/model/postprocessor:abstract_postprocessor",
        "//yggdrasil_decision_forests/utils:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)
