load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Library
# =======

cc_library_ydf(
    name = "gradient_boosted_trees",
    srcs = [
        "gradient_boosted_trees.cc",
        "gradient_boosted_trees_hparams_templates.cc",
    ],
    hdrs = [
        "gradient_boosted_trees.h",
    ],
    deps = [
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:all_implementations",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:loss_library",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:loss_interface",
        ":gradient_boosted_trees_cc_proto",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:formats",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/dataset:weight",
        "//yggdrasil_decision_forests/learner:abstract_learner",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:generic_parameters",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/decision_tree:utils",
        "//yggdrasil_decision_forests/learner:types",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/metric:ranking",
        "//yggdrasil_decision_forests/metric:ranking_utils",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/serving/decision_forest:register_engines",
        "//yggdrasil_decision_forests/utils:adaptive_work",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:csv",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:hyper_parameters",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:snapshot",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils:usage",
        "//yggdrasil_decision_forests/utils:feature_importance",
    ] + select({
        "//conditions:default": [
        ],
    }),
    alwayslink = 1,
)

# Proto
# ========

all_proto_library(
    name = "gradient_boosted_trees_proto",
    srcs = ["gradient_boosted_trees.proto"],
    deps = [
        "//yggdrasil_decision_forests/learner:abstract_learner_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_proto",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_proto",
    ],
)

# Test
# ========
cc_test(
    name = "gradient_boosted_trees_test",
    size = "large",
    srcs = ["gradient_boosted_trees_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    # By default, do not fail with tests with a "maximum training duration" last longer than
    # indicated. This can create flakyness with concurrent execution of tests.
    defines = ["SKIP_TIMING_TESTS"],
    shard_count = 12,
    deps = [
        ":gradient_boosted_trees",
        ":gradient_boosted_trees_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:loss_library",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/metric:report",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:csv",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:test_utils",
    ],
)
