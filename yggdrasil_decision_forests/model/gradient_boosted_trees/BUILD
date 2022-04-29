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
    ],
    hdrs = [
        "gradient_boosted_trees.h",
    ],
    deps = [
        ":gradient_boosted_trees_cc_proto",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/metric:metric_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model:prediction_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_forest_interface",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils:usage",
    ],
    alwayslink = 1,
)

# Proto
# ========

all_proto_library(
    name = "gradient_boosted_trees_proto",
    srcs = ["gradient_boosted_trees.proto"],
)

# Test
# ========

cc_test(
    name = "gradient_boosted_trees_test",
    size = "medium",
    srcs = ["gradient_boosted_trees_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":gradient_boosted_trees",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:test",
    ],
)
