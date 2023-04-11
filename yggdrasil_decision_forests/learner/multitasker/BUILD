load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Library
# =======

cc_library_ydf(
    name = "multitasker",
    srcs = ["multitasker.cc"],
    hdrs = ["multitasker.h"],
    deps = [
        ":multitasker_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model/multitasker",
        "//yggdrasil_decision_forests/serving:example_set",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:regex",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils:synchronization_primitives",
        "//yggdrasil_decision_forests/utils:usage",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

# Proto
# ========

all_proto_library(
    name = "multitasker_proto",
    srcs = ["multitasker.proto"],
    deps = [
        "//yggdrasil_decision_forests/learner:abstract_learner_proto",
    ],
)

# Test
# ========

cc_test(
    name = "multitasker_test",
    srcs = ["multitasker_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":multitasker",
        ":multitasker_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/learner/random_forest",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/model/multitasker",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:test_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
