load("//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

cc_library_ydf(
    name = "register_engines",
    srcs = [
        "register_engines.cc",
    ],
    hdrs = [
        "register_engines.h",
    ],
    deps = [
        ":decision_forest",
        ":quick_scorer_extended",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/serving:example_set_model_wrapper",
        "//yggdrasil_decision_forests/serving:fast_engine",
        "//yggdrasil_decision_forests/utils:compatibility",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "decision_forest",
    srcs = [
        "decision_forest.cc",
    ],
    hdrs = [
        "decision_forest.h",
    ],
    deps = [
        ":utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model/random_forest",
        "//yggdrasil_decision_forests/serving:example_set",
        "//yggdrasil_decision_forests/utils:bitmap",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils:usage",
    ],
)

cc_library_ydf(
    name = "quick_scorer_extended",
    srcs = [
        "quick_scorer_extended.cc",
    ],
    hdrs = [
        "quick_scorer_extended.h",
    ],
    deps = [
        ":utils",
        "@com_google_absl//absl/base:config",
        "@com_google_absl//absl/status",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/serving:example_set",
        "//yggdrasil_decision_forests/utils:bitmap",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:usage",
    ],
)

cc_library_ydf(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "utils.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model/random_forest",
    ],
)

# Proto
# =====

# Tests
# =====

cc_test(
    name = "decision_forest_test",
    size = "large",
    srcs = ["decision_forest_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    shard_count = 10,
    deps = [
        ":decision_forest",
        ":quick_scorer_extended",
        ":register_engines",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/model:prediction_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/utils:csv",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:test_utils",
    ],
)

cc_test(
    name = "quick_scorer_extended_test",
    srcs = ["quick_scorer_extended_test.cc"],
    deps = [
        ":quick_scorer_extended",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/utils:test",
    ],
)
