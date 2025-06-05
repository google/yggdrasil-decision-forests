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
        ":utils",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_forest_interface",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model/random_forest",
        "//yggdrasil_decision_forests/utils:bitmap",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library_ydf(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        ":embed_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
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
        ":embed_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:protobuf",
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
        ":embed_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:builder",
        "//yggdrasil_decision_forests/model/decision_tree:decision_forest_interface",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/utils:test",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "embed_pred_test",
    srcs = ["embed_pred_test.cc"],
    deps = [
        ":test_model_abalone_regression_gbdt_v2",
        ":test_model_abalone_regression_rf_small",
        ":test_model_adult_binary_class_gbdt_filegroup_filegroup",
        ":test_model_adult_binary_class_gbdt_v2_class",
        ":test_model_adult_binary_class_gbdt_v2_proba",
        ":test_model_adult_binary_class_gbdt_v2_score",
        ":test_model_adult_binary_class_rf_nwta_small_class",
        ":test_model_adult_binary_class_rf_nwta_small_proba",
        ":test_model_adult_binary_class_rf_nwta_small_score",
        ":test_model_adult_binary_class_rf_wta_small_class",
        ":test_model_adult_binary_class_rf_wta_small_proba",
        ":test_model_adult_binary_class_rf_wta_small_score",
        ":test_model_iris_multi_class_gbdt_v2_class",
        ":test_model_iris_multi_class_gbdt_v2_proba",
        ":test_model_iris_multi_class_gbdt_v2_score",
        ":test_model_iris_multi_class_rf_nwta_small_class",
        ":test_model_iris_multi_class_rf_nwta_small_proba",
        ":test_model_iris_multi_class_rf_nwta_small_score",
        ":test_model_iris_multi_class_rf_wta_small_class",
        ":test_model_iris_multi_class_rf_wta_small_proba",
        ":test_model_iris_multi_class_rf_wta_small_score",
        "@com_google_googletest//:gtest_main",
    ],
)

# Models to compile during the unit tests

cc_ydf_embedded_model(
    name = "test_model_adult_binary_class_gbdt_filegroup_filegroup",
    data = "//yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt_filegroup",
)

# GBT binary class
cc_ydf_embedded_model(
    name = "test_model_adult_binary_class_gbdt_v2_class",
    classification_output = "CLASS",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt_v2",
)

cc_ydf_embedded_model(
    name = "test_model_adult_binary_class_gbdt_v2_score",
    classification_output = "SCORE",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt_v2",
)

cc_ydf_embedded_model(
    name = "test_model_adult_binary_class_gbdt_v2_proba",
    classification_output = "PROBABILITY",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt_v2",
)

# RF binary class

cc_ydf_embedded_model(
    name = "test_model_adult_binary_class_rf_wta_small_class",
    classification_output = "CLASS",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_rf_wta_small",
)

cc_ydf_embedded_model(
    name = "test_model_adult_binary_class_rf_wta_small_score",
    classification_output = "SCORE",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_rf_wta_small",
)

cc_ydf_embedded_model(
    name = "test_model_adult_binary_class_rf_wta_small_proba",
    classification_output = "PROBABILITY",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_rf_wta_small",
)

cc_ydf_embedded_model(
    name = "test_model_adult_binary_class_rf_nwta_small_class",
    classification_output = "CLASS",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_rf_nwta_small",
)

cc_ydf_embedded_model(
    name = "test_model_adult_binary_class_rf_nwta_small_score",
    classification_output = "SCORE",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_rf_nwta_small",
)

cc_ydf_embedded_model(
    name = "test_model_adult_binary_class_rf_nwta_small_proba",
    classification_output = "PROBABILITY",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_rf_nwta_small",
)

# Regression

cc_ydf_embedded_model(
    name = "test_model_abalone_regression_gbdt_v2",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/abalone_regression_gbdt_v2",
)

cc_ydf_embedded_model(
    name = "test_model_abalone_regression_rf_small",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/abalone_regression_rf_small",
)

# GBT multi-class
cc_ydf_embedded_model(
    name = "test_model_iris_multi_class_gbdt_v2_class",
    classification_output = "CLASS",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_gbdt_v2",
)

cc_ydf_embedded_model(
    name = "test_model_iris_multi_class_gbdt_v2_score",
    classification_output = "SCORE",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_gbdt_v2",
)

cc_ydf_embedded_model(
    name = "test_model_iris_multi_class_gbdt_v2_proba",
    classification_output = "PROBABILITY",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_gbdt_v2",
)

# RF multi-class
cc_ydf_embedded_model(
    name = "test_model_iris_multi_class_rf_nwta_small_class",
    classification_output = "CLASS",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_rf_nwta_small",
)

cc_ydf_embedded_model(
    name = "test_model_iris_multi_class_rf_nwta_small_score",
    classification_output = "SCORE",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_rf_nwta_small",
)

cc_ydf_embedded_model(
    name = "test_model_iris_multi_class_rf_nwta_small_proba",
    classification_output = "PROBABILITY",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_rf_nwta_small",
)

cc_ydf_embedded_model(
    name = "test_model_iris_multi_class_rf_wta_small_class",
    classification_output = "CLASS",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_rf_wta_small",
)

cc_ydf_embedded_model(
    name = "test_model_iris_multi_class_rf_wta_small_score",
    classification_output = "SCORE",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_rf_wta_small",
)

cc_ydf_embedded_model(
    name = "test_model_iris_multi_class_rf_wta_small_proba",
    classification_output = "PROBABILITY",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_rf_wta_small",
)
