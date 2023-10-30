load("@ydf_cc//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(["generate_wrapper.cc"])

# Bzl Libraries
# =============

bzl_library(
    name = "wrapper_generator_bzl",
    srcs = ["wrapper_generator.bzl"],
    deps = [
        "@ydf_cc//yggdrasil_decision_forests/utils:compile_bzl",
    ],
)

# Libraries
# =========

cc_library_ydf(
    name = "wrapper_generator",
    srcs = ["wrapper_generator.cc"],
    hdrs = ["wrapper_generator.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner:learner_library",
        "@ydf_cc//yggdrasil_decision_forests/model:hyperparameter_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:hyper_parameters",
        "@ydf_cc//yggdrasil_decision_forests/utils:status_macros",
    ],
)

# Tests
# =====

cc_test(
    name = "wrapper_test",
    srcs = ["wrapper_test.cc"],
    linkstatic = True,
    deps = [
        ":wrapper_generator",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@ydf_cc//yggdrasil_decision_forests/dataset:vertical_dataset",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/learner/random_forest",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model",
        "@ydf_cc//yggdrasil_decision_forests/model:hyperparameter_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
        "@ydf_cc//yggdrasil_decision_forests/utils:test",
        "@ydf_cc//yggdrasil_decision_forests/utils:testing_macros",
    ],
)
