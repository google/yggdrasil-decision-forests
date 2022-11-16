load("//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Library
# =======

cc_library_ydf(
    name = "loss_interface",
    srcs = ["loss_interface.cc"],
    hdrs = ["loss_interface.h"],
    deps = [
        ":loss_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/metric:ranking",
        "//yggdrasil_decision_forests/metric:ranking_utils",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:registration",
    ],
)

cc_library_ydf(
    name = "loss_library",
    srcs = ["loss_library.cc"],
    hdrs = ["loss_library.h"],
    deps = [
        ":loss_interface",
        "@com_google_absl//absl/status:statusor",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
    ],
)

cc_library_ydf(
    name = "loss_utils",
    srcs = ["loss_utils.cc"],
    hdrs = ["loss_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:types",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:utils",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
)

cc_library_ydf(
    name = "all_implementations",
    deps = [
        ":loss_imp_binary_focal",
        ":loss_imp_binomial",
        ":loss_imp_cross_entropy_ndcg",
        ":loss_imp_mean_square_error",
        ":loss_imp_multinomial",
        ":loss_imp_ndcg",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "loss_imp_binary_focal",
    srcs = ["loss_imp_binary_focal.cc"],
    hdrs = ["loss_imp_binary_focal.h"],
    deps = [
        ":loss_imp_binomial",
        ":loss_interface",
        ":loss_utils",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/decision_tree:utils",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:random",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "loss_imp_binomial",
    srcs = ["loss_imp_binomial.cc"],
    hdrs = ["loss_imp_binomial.h"],
    deps = [
        ":loss_interface",
        ":loss_utils",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/decision_tree:utils",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "loss_imp_cross_entropy_ndcg",
    srcs = ["loss_imp_cross_entropy_ndcg.cc"],
    hdrs = ["loss_imp_cross_entropy_ndcg.h"],
    deps = [
        ":loss_imp_ndcg",
        ":loss_interface",
        ":loss_utils",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/decision_tree:utils",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:random",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "loss_imp_mean_square_error",
    srcs = ["loss_imp_mean_square_error.cc"],
    hdrs = ["loss_imp_mean_square_error.h"],
    deps = [
        ":loss_interface",
        ":loss_utils",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/decision_tree:utils",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:random",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "loss_imp_multinomial",
    srcs = ["loss_imp_multinomial.cc"],
    hdrs = ["loss_imp_multinomial.h"],
    deps = [
        ":loss_interface",
        ":loss_utils",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/decision_tree:utils",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:random",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "loss_imp_ndcg",
    srcs = ["loss_imp_ndcg.cc"],
    hdrs = ["loss_imp_ndcg.h"],
    deps = [
        ":loss_interface",
        ":loss_utils",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/decision_tree:utils",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/metric:ranking",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:random",
    ],
    alwayslink = 1,
)

# Test
# ========
cc_test(
    name = "loss_library_test",
    srcs = ["loss_library_test.cc"],
    deps = [
        ":all_implementations",
        ":loss_library",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/utils:test",
    ],
)

# Test
# ========

cc_test(
    name = "loss_imp_binomial_test",
    srcs = ["loss_imp_binomial_test.cc"],
    deps = [
        ":loss_imp_binomial",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
    ],
)

cc_test(
    name = "loss_imp_multinomial_test",
    srcs = ["loss_imp_multinomial_test.cc"],
    deps = [
        ":loss_imp_multinomial",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status:statusor",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
    ],
)

cc_test(
    name = "loss_imp_mean_square_error_test",
    srcs = ["loss_imp_mean_square_error_test.cc"],
    deps = [
        ":loss_imp_mean_square_error",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
    ],
)

cc_test(
    name = "loss_imp_binary_focal_test",
    srcs = ["loss_imp_binary_focal_test.cc"],
    deps = [
        ":loss_imp_binary_focal",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
    ],
)

cc_test(
    name = "loss_imp_ndcg_test",
    srcs = ["loss_imp_ndcg_test.cc"],
    deps = [
        ":loss_imp_cross_entropy_ndcg",
        ":loss_imp_ndcg",
        ":loss_interface",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
    ],
)
