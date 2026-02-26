load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//yggdrasil_decision_forests/serving/embed:embed.bzl", "c_ydf_standalone_model")
load("//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library_ydf(
    name = "c_embed",
    srcs = ["c_embed.cc"],
    hdrs = ["c_embed.h"],
    deps = [
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_forest_interface",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model/random_forest",
        "//yggdrasil_decision_forests/serving/embed:common",
        "//yggdrasil_decision_forests/serving/embed:embed_cc_proto",
        "//yggdrasil_decision_forests/serving/embed:utils",
        "//yggdrasil_decision_forests/utils:bitmap",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

# Tests
# =====

cc_test(
    name = "c_embed_test",
    srcs = ["c_embed_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
        "//yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt_filegroup",
    ],
    deps = [
        ":c_embed",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:builder",
        "//yggdrasil_decision_forests/model/decision_tree:decision_forest_interface",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/serving/embed",
        "//yggdrasil_decision_forests/serving/embed:common",
        "//yggdrasil_decision_forests/serving/embed:embed_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "c_embed_pred_test",
    srcs = ["c_embed_pred_test.cc"],
    deps = [
        ":test_model_abalone_regression_gbdt_v2_routing",
        "@com_google_googletest//:gtest_main",
    ],
)

# Models to compile during the unit tests

c_ydf_standalone_model(
    name = "test_model_abalone_regression_gbdt_v2_routing",
    algorithm = "ROUTING",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/abalone_regression_gbdt_v2",
)
