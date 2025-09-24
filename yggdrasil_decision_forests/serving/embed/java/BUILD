load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

cc_library_ydf(
    name = "java_embed",
    srcs = ["java_embed.cc"],
    hdrs = ["java_embed.h"],
    deps = [
        ":model_data_bank",
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
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_ydf(
    name = "model_data_bank",
    srcs = ["model_data_bank.cc"],
    hdrs = ["model_data_bank.h"],
    deps = [
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model/random_forest",
        "//yggdrasil_decision_forests/serving/embed:common",
        "//yggdrasil_decision_forests/serving/embed:utils",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

# Tests
# =====

cc_test(
    name = "model_data_bank_test",
    srcs = ["model_data_bank_test.cc"],
    deps = [
        ":java_embed",
        ":model_data_bank",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/serving/embed:common",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "java_embed_test",
    srcs = ["java_embed_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
        "//yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt_filegroup",
    ],
    deps = [
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/model/decision_tree:decision_forest_interface",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/serving/embed",
        "//yggdrasil_decision_forests/serving/embed:embed_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_googletest//:gtest_main",
    ],
)
