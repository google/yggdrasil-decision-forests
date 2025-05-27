load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//yggdrasil_decision_forests/serving/embed:embed.bzl", "cc_ydf_embedded_model")
load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_binary_ydf", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

cc_library_ydf(
    name = "embed",
    srcs = ["embed.cc"],
    hdrs = ["embed.h"],
    deps = [
        ":embed_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

# Binaries
# ========

cc_binary_ydf(
    name = "write_embed",
    srcs = ["write_embed.cc"],
    deps = [
        ":embed",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
    ],
)

# Bzl Libraries
# =============

bzl_library(
    name = "embed_bzl",
    srcs = ["embed.bzl"],
)

# Proto
# ========

all_proto_library(
    name = "embed_proto",
    srcs = ["embed.proto"],
    deps = [],
)

# Tests
# =====

cc_test(
    name = "embed_test",
    srcs = ["embed_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
        "//yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt_filegroup",
    ],
    deps = [
        ":embed",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "embed_pred_test",
    srcs = ["embed_pred_test.cc"],
    deps = [
        ":test_model1",
        ":test_model2",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_ydf_embedded_model(
    name = "test_model1",
    data = "//yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt_filegroup",
)

cc_ydf_embedded_model(
    name = "test_model2",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt",
)
