load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library_ydf(
    name = "all_optimizers",
    deps = [
        "//yggdrasil_decision_forests/learner/hyperparameters_optimizer/optimizers:random",
    ] + select({
        "//conditions:default": [
        ],
    }),
)

# Library
# =======

cc_library_ydf(
    name = "hyperparameters_optimizer",
    srcs = ["hyperparameters_optimizer.cc"],
    hdrs = ["hyperparameters_optimizer.h"],
    deps = [
        ":all_optimizers",
        ":hyperparameters_optimizer_cc_proto",
        ":optimizer_interface",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/learner:abstract_learner",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/learner/generic_worker:generic_worker_cc_proto",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/metric:metric_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:hyper_parameters",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils/distribute:core",
        "//yggdrasil_decision_forests/utils/distribute:distribute_without_implementations",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "optimizer_interface",
    hdrs = ["optimizer_interface.h"],
    deps = [
        ":hyperparameters_optimizer_cc_proto",
        "//yggdrasil_decision_forests/utils:registration",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

# Proto
# ========

all_proto_library(
    name = "hyperparameters_optimizer_proto",
    srcs = ["hyperparameters_optimizer.proto"],
    deps = [
        "//yggdrasil_decision_forests/learner:abstract_learner_proto",
        "//yggdrasil_decision_forests/metric:metric_proto",
        "//yggdrasil_decision_forests/model:hyperparameter_proto",
        "//yggdrasil_decision_forests/utils:fold_generator_proto",
    ],
)

# Test
# ========

cc_test(
    name = "hyperparameters_optimizer_test",
    size = "large",
    srcs = ["hyperparameters_optimizer_test.cc"],
    data = ["//yggdrasil_decision_forests/test_data"],
    shard_count = 8,
    tags = [
    ],
    deps = [
        ":hyperparameters_optimizer",
        "//yggdrasil_decision_forests/learner/generic_worker",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:test_utils",
        "//yggdrasil_decision_forests/utils/distribute/implementations/multi_thread",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
