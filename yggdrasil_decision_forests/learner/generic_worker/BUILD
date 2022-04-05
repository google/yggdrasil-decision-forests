load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Proto
# ========

all_proto_library(
    name = "generic_worker_proto",
    srcs = ["generic_worker.proto"],
    deps = [
        "//yggdrasil_decision_forests/dataset:data_spec_proto",
        "//yggdrasil_decision_forests/learner:abstract_learner_proto",
        "//yggdrasil_decision_forests/metric:metric_proto",
        "//yggdrasil_decision_forests/model:hyperparameter_proto",
        "//yggdrasil_decision_forests/utils/distribute:distribute_proto",
    ],
)

# Library
# =======

cc_library_ydf(
    name = "generic_worker",
    srcs = [
        "generic_worker.cc",
    ],
    hdrs = [
        "generic_worker.h",
    ],
    deps = [
        ":generic_worker_cc_proto",
        "//yggdrasil_decision_forests/cli:all_file_systems",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:weight",
        "//yggdrasil_decision_forests/learner:abstract_learner",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/metric:metric_cc_proto",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:protobuf",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils/distribute:core",
        "//yggdrasil_decision_forests/utils/distribute:distribute_cc_proto",
        "//yggdrasil_decision_forests/utils:uid",
    ] + select({
        "//conditions:default": ["//yggdrasil_decision_forests/learner:all_learners"],
    }),
    alwayslink = 1,
)

cc_test(
    name = "generic_worker_test",
    srcs = ["generic_worker_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":generic_worker",
        ":generic_worker_cc_proto",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/learner/random_forest",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/metric:report",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils/distribute",
        "//yggdrasil_decision_forests/utils/distribute:distribute_without_implementations",
        "//yggdrasil_decision_forests/utils/distribute/implementations/multi_thread:multi_thread_cc_proto",
    ],
)
