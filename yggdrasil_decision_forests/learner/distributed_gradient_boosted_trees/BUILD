load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Library
# =======

cc_library_ydf(
    name = "dgbt",
    srcs = ["distributed_gradient_boosted_trees.cc"],
    hdrs = ["distributed_gradient_boosted_trees.h"],
    deps = [
        ":common",
        ":dgbt_cc_proto",
        ":worker_cc_proto",
        "//yggdrasil_decision_forests/dataset:formats",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:generic_parameters",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree:training",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_common",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/load_balancer",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:all_implementations",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:loss_interface",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:loss_library",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:snapshot",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils:uid",
        "//yggdrasil_decision_forests/utils:usage",
        "//yggdrasil_decision_forests/utils/distribute:core",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "worker",
    srcs = ["worker.cc"],
    hdrs = ["worker.h"],
    deps = [
        ":common",
        ":dgbt_cc_proto",
        ":worker_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/dataset:weight",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree:training",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:all_implementations",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:loss_interface",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:loss_library",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/serving:fast_engine",
        "//yggdrasil_decision_forests/utils:bitmap",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:protobuf",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils:uid",
        "//yggdrasil_decision_forests/utils/distribute:core",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = [
        ":worker_cc_proto",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:column_cache",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:protobuf",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# Proto
# ========

all_proto_library(
    name = "dgbt_proto",
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
    tags = ["manual"],
    deps = [
        ":common",
        ":dgbt",
        ":worker",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_worker",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:csv",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:test_utils",
        "//yggdrasil_decision_forests/utils/distribute/implementations/multi_thread",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
    ],
)
