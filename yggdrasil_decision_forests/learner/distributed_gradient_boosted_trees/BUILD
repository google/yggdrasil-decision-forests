load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Library
# =======

cc_library_ydf(
    name = "distributed_gradient_boosted_trees",
    srcs = ["distributed_gradient_boosted_trees.cc"],
    hdrs = ["distributed_gradient_boosted_trees.h"],
    deps = [
        ":common",
        ":distributed_gradient_boosted_trees_cc_proto",
        ":worker_cc_proto",
        "//yggdrasil_decision_forests/dataset:formats",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner:types",
        "//yggdrasil_decision_forests/learner/decision_tree:generic_parameters",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree:training",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_common",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/load_balancer",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:all_implementations",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:loss_library",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:snapshot",
        "//yggdrasil_decision_forests/utils:uid",
        "//yggdrasil_decision_forests/utils:usage",
        "//yggdrasil_decision_forests/utils/distribute:core",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "worker",
    srcs = ["worker.cc"],
    hdrs = ["worker.h"],
    deps = [
        ":common",
        ":distributed_gradient_boosted_trees_cc_proto",
        ":worker_cc_proto",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree:training",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:all_implementations",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:loss_library",
        "//yggdrasil_decision_forests/utils:bitmap",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils:uid",
        "//yggdrasil_decision_forests/utils/distribute:core",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:filesystem",
    ],
)

# Proto
# ========

all_proto_library(
    name = "distributed_gradient_boosted_trees_proto",
    srcs = ["distributed_gradient_boosted_trees.proto"],
    deps = [
        "//yggdrasil_decision_forests/learner:abstract_learner_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_proto",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_proto",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/load_balancer:load_balancer_proto",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_proto",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_proto",
    ],
)

all_proto_library(
    name = "worker_proto",
    srcs = ["worker.proto"],
    deps = [
        "//yggdrasil_decision_forests/dataset:data_spec_proto",
        "//yggdrasil_decision_forests/learner:abstract_learner_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_proto",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree:training_proto",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_proto",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/load_balancer:load_balancer_proto",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_proto",
    ],
)

# Test
# ========
cc_test(
    name = "distributed_gradient_boosted_trees_test",
    size = "large",
    srcs = ["distributed_gradient_boosted_trees_test.cc"],
    data = ["//yggdrasil_decision_forests/test_data"],
    shard_count = 10,
    deps = [
        ":distributed_gradient_boosted_trees",
        ":distributed_gradient_boosted_trees_cc_proto",
        ":worker",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_worker",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:test_utils",
        "//yggdrasil_decision_forests/utils/distribute/implementations/multi_thread",
    ],
)
