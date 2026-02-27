load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//yggdrasil_decision_forests/serving/embed:embed.bzl", "c_ydf_standalone_model")
load("//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library_ydf(
    name = "c_emitter",
    srcs = ["c_emitter.cc"],
    hdrs = ["c_emitter.h"],
    deps = [
        ":c_ir",
        "//yggdrasil_decision_forests/serving/embed:embed_cc_proto",
        "//yggdrasil_decision_forests/serving/embed/ir:model_ir",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_ydf(
    name = "c_target_lowering",
    srcs = ["c_target_lowering.cc"],
    hdrs = ["c_target_lowering.h"],
    deps = [
        ":c_ir",
        "//yggdrasil_decision_forests/serving/embed:common",
        "//yggdrasil_decision_forests/serving/embed:embed_cc_proto",
        "//yggdrasil_decision_forests/serving/embed:utils",
        "//yggdrasil_decision_forests/serving/embed/ir:model_ir",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_ydf(
    name = "c_ir",
    hdrs = ["c_ir.h"],
    deps = [
        "//yggdrasil_decision_forests/serving/embed:common",
        "//yggdrasil_decision_forests/serving/embed/ir:model_ir",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library_ydf(
    name = "c_embed",
    srcs = ["c_embed.cc"],
    hdrs = ["c_embed.h"],
    deps = [
        ":c_emitter",
        ":c_ir",
        ":c_target_lowering",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model/random_forest",
        "//yggdrasil_decision_forests/serving/embed:common",
        "//yggdrasil_decision_forests/serving/embed:utils",
        "//yggdrasil_decision_forests/serving/embed/ir:builder",
        "//yggdrasil_decision_forests/serving/embed/ir:model_ir",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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

cc_test(
    name = "c_embed_pred_test",
    srcs = ["c_embed_pred_test.cc"],
    deps = [
        ":test_model_abalone_regression_gbdt_v2_routing",
        ":test_model_abalone_regression_rf_small_routing",
        ":test_model_adult_binary_class_gbdt_oblique_proba_routing",
        ":test_model_adult_binary_class_gbdt_v2_class_routing",
        ":test_model_adult_binary_class_gbdt_v2_proba_routing",
        ":test_model_adult_binary_class_gbdt_v2_score_routing",
        ":test_model_adult_binary_class_rf_nwta_small_class_routing",
        ":test_model_adult_binary_class_rf_nwta_small_proba_routing",
        ":test_model_adult_binary_class_rf_nwta_small_score_routing",
        ":test_model_adult_binary_class_rf_wta_small_class_routing",
        ":test_model_adult_binary_class_rf_wta_small_proba_routing",
        ":test_model_adult_binary_class_rf_wta_small_score_routing",
        ":test_model_iris_multi_class_gbdt_v2_class_routing",
        ":test_model_iris_multi_class_gbdt_v2_proba_routing",
        ":test_model_iris_multi_class_gbdt_v2_score_routing",
        ":test_model_iris_multi_class_rf_nwta_small_class_routing",
        ":test_model_iris_multi_class_rf_nwta_small_proba_routing",
        ":test_model_iris_multi_class_rf_nwta_small_score_routing",
        ":test_model_iris_multi_class_rf_wta_small_class_routing",
        ":test_model_iris_multi_class_rf_wta_small_proba_routing",
        ":test_model_iris_multi_class_rf_wta_small_score_routing",
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

c_ydf_standalone_model(
    name = "test_model_abalone_regression_rf_small_routing",
    algorithm = "ROUTING",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/abalone_regression_rf_small",
)

c_ydf_standalone_model(
    name = "test_model_adult_binary_class_gbdt_v2_score_routing",
    algorithm = "ROUTING",
    classification_output = "SCORE",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt_v2",
)

c_ydf_standalone_model(
    name = "test_model_adult_binary_class_gbdt_v2_class_routing",
    algorithm = "ROUTING",
    classification_output = "CLASS",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt_v2",
)

c_ydf_standalone_model(
    name = "test_model_adult_binary_class_gbdt_v2_proba_routing",
    algorithm = "ROUTING",
    classification_output = "PROBABILITY",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt_v2",
)

c_ydf_standalone_model(
    name = "test_model_adult_binary_class_gbdt_oblique_proba_routing",
    algorithm = "ROUTING",
    classification_output = "PROBABILITY",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt_oblique",
)

c_ydf_standalone_model(
    name = "test_model_adult_binary_class_rf_nwta_small_class_routing",
    algorithm = "ROUTING",
    classification_output = "CLASS",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_rf_nwta_small",
)

c_ydf_standalone_model(
    name = "test_model_adult_binary_class_rf_nwta_small_score_routing",
    algorithm = "ROUTING",
    classification_output = "SCORE",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_rf_nwta_small",
)

c_ydf_standalone_model(
    name = "test_model_adult_binary_class_rf_nwta_small_proba_routing",
    algorithm = "ROUTING",
    classification_output = "PROBABILITY",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_rf_nwta_small",
)

c_ydf_standalone_model(
    name = "test_model_adult_binary_class_rf_wta_small_class_routing",
    algorithm = "ROUTING",
    classification_output = "CLASS",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_rf_wta_small",
)

c_ydf_standalone_model(
    name = "test_model_adult_binary_class_rf_wta_small_score_routing",
    algorithm = "ROUTING",
    classification_output = "SCORE",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_rf_wta_small",
)

c_ydf_standalone_model(
    name = "test_model_adult_binary_class_rf_wta_small_proba_routing",
    algorithm = "ROUTING",
    classification_output = "PROBABILITY",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/adult_binary_class_rf_wta_small",
)

c_ydf_standalone_model(
    name = "test_model_iris_multi_class_gbdt_v2_class_routing",
    algorithm = "ROUTING",
    classification_output = "CLASS",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_gbdt_v2",
)

c_ydf_standalone_model(
    name = "test_model_iris_multi_class_gbdt_v2_score_routing",
    algorithm = "ROUTING",
    classification_output = "SCORE",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_gbdt_v2",
)

c_ydf_standalone_model(
    name = "test_model_iris_multi_class_gbdt_v2_proba_routing",
    algorithm = "ROUTING",
    classification_output = "PROBABILITY",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_gbdt_v2",
)

c_ydf_standalone_model(
    name = "test_model_iris_multi_class_rf_nwta_small_class_routing",
    algorithm = "ROUTING",
    classification_output = "CLASS",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_rf_nwta_small",
)

c_ydf_standalone_model(
    name = "test_model_iris_multi_class_rf_nwta_small_score_routing",
    algorithm = "ROUTING",
    classification_output = "SCORE",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_rf_nwta_small",
)

c_ydf_standalone_model(
    name = "test_model_iris_multi_class_rf_nwta_small_proba_routing",
    algorithm = "ROUTING",
    classification_output = "PROBABILITY",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_rf_nwta_small",
)

c_ydf_standalone_model(
    name = "test_model_iris_multi_class_rf_wta_small_class_routing",
    algorithm = "ROUTING",
    classification_output = "CLASS",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_rf_wta_small",
)

c_ydf_standalone_model(
    name = "test_model_iris_multi_class_rf_wta_small_score_routing",
    algorithm = "ROUTING",
    classification_output = "SCORE",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_rf_wta_small",
)

c_ydf_standalone_model(
    name = "test_model_iris_multi_class_rf_wta_small_proba_routing",
    algorithm = "ROUTING",
    classification_output = "PROBABILITY",
    data = "//yggdrasil_decision_forests/test_data",
    path = "yggdrasil_decision_forests/test_data/model/iris_multi_class_rf_wta_small",
)
