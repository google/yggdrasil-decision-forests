load("//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Library
# =======

cc_library_ydf(
    name = "loss_interface",
    srcs = [
        "loss_interface.cc",
        "loss_utils.cc",
    ],
    hdrs = [
        "loss_interface.h",
        "loss_utils.h",
    ],
    deps = [
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/decision_tree:utils",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/metric:ranking",
        "//yggdrasil_decision_forests/metric:ranking_utils",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:registration",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library_ydf(
    name = "loss_library",
    srcs = ["loss_library.cc"],
    hdrs = ["loss_library.h"],
    deps = [
        ":loss_interface",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library_ydf(
    name = "all_implementations",
    deps = [
        ":loss_imp_binary_focal",
        ":loss_imp_binomial",
        ":loss_imp_cross_entropy_ndcg",
        ":loss_imp_mean_average_error",
        ":loss_imp_mean_square_error",
        ":loss_imp_multinomial",
        ":loss_imp_ndcg",
        ":loss_imp_poisson",
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
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "loss_imp_binomial",
    srcs = ["loss_imp_binomial.cc"],
    hdrs = ["loss_imp_binomial.h"],
    deps = [
        ":loss_interface",
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
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "loss_imp_mean_square_error",
    srcs = ["loss_imp_mean_square_error.cc"],
    hdrs = ["loss_imp_mean_square_error.h"],
    deps = [
        ":loss_interface",
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
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:random",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "loss_imp_mean_average_error",
    srcs = ["loss_imp_mean_average_error.cc"],
    hdrs = ["loss_imp_mean_average_error.h"],
    deps = [
        ":loss_interface",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:math",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "loss_imp_multinomial",
    srcs = ["loss_imp_multinomial.cc"],
    hdrs = ["loss_imp_multinomial.h"],
    deps = [
        ":loss_interface",
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
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "loss_imp_ndcg",
    srcs = ["loss_imp_ndcg.cc"],
    hdrs = ["loss_imp_ndcg.h"],
    deps = [
        ":loss_interface",
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
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "loss_imp_poisson",
    srcs = ["loss_imp_poisson.cc"],
    hdrs = ["loss_imp_poisson.h"],
    deps = [
        ":loss_interface",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/decision_tree:utils",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/utils:test",
        "@com_google_googletest//:gtest_main",
    ],
)

# Test
# ========

cc_test(
    name = "loss_imp_binomial_test",
    srcs = ["loss_imp_binomial_test.cc"],
    deps = [
        ":loss_imp_binomial",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "loss_imp_multinomial_test",
    srcs = ["loss_imp_multinomial_test.cc"],
    deps = [
        ":loss_imp_multinomial",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "loss_imp_mean_square_error_test",
    srcs = ["loss_imp_mean_square_error_test.cc"],
    deps = [
        ":loss_imp_mean_square_error",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "loss_imp_mean_average_error_test",
    srcs = ["loss_imp_mean_average_error_test.cc"],
    deps = [
        ":loss_imp_mean_average_error",
        ":loss_interface",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "loss_imp_binary_focal_test",
    srcs = ["loss_imp_binary_focal_test.cc"],
    deps = [
        ":loss_imp_binary_focal",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "loss_imp_ndcg_test",
    srcs = ["loss_imp_ndcg_test.cc"],
    deps = [
        ":loss_imp_cross_entropy_ndcg",
        ":loss_imp_ndcg",
        ":loss_interface",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "loss_imp_poisson_test",
    srcs = ["loss_imp_poisson_test.cc"],
    deps = [
        ":loss_imp_poisson",
        ":loss_interface",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "loss_utils_test",
    srcs = ["loss_utils_test.cc"],
    deps = [
        ":loss_interface",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_googletest//:gtest_main",
    ],
)
