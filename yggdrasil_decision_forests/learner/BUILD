load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_binary_ydf", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Binaries
# ========

cc_binary_ydf(
    name = "export_doc_main",
    srcs = ["export_doc_main.cc"],
    deps = [
        ":all_learners",
        ":export_doc",
        ":learner_library",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:logging",
    ],
)

# Learner groups
# ==============

# Add new learners here.
cc_library_ydf(
    name = "all_learners_except_hparam_optimizer",
    deps = [
        "//yggdrasil_decision_forests/learner/cart",
        "//yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/learner/random_forest",
    ],
)

cc_library_ydf(
    name = "all_learners",
    deps = [
        ":all_learners_except_hparam_optimizer",
    ],
)

# Library
# =======

cc_library_ydf(
    name = "types",
    srcs = ["types.cc"],
    hdrs = ["types.h"],
    defines = select({
        "//yggdrasil_decision_forests:example_idx_32bits": ["YGGDRASIL_EXAMPLE_IDX_32_BITS"],
        "//yggdrasil_decision_forests:example_idx_64bits": ["YGGDRASIL_EXAMPLE_IDX_64_BITS"],
        "//conditions:default": ["YGGDRASIL_EXAMPLE_IDX_32_BITS"],
    }),
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_ydf(
    name = "abstract_learner",
    srcs = [
        "abstract_learner.cc",
    ],
    hdrs = [
        "abstract_learner.h",
    ],
    deps = [
        ":abstract_learner_cc_proto",
        ":types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/dataset:weight",
        "//yggdrasil_decision_forests/dataset:weight_cc_proto",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/metric:metric_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:fold_generator",
        "//yggdrasil_decision_forests/utils:fold_generator_cc_proto",
        "//yggdrasil_decision_forests/utils:hyper_parameters",
        "//yggdrasil_decision_forests/utils:synchronization_primitives",
        "//yggdrasil_decision_forests/utils:uid",
    ],
)

# Note: The hyper parameter optimizer and the model library are in the same cc_library_ydf because they
# co-depend on each others.
cc_library_ydf(
    name = "learner_library",
    srcs = [
        "learner_library.cc",
    ],
    hdrs = [
        "learner_library.h",
    ],
    deps = [
        ":abstract_learner",
        ":abstract_learner_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ] + select({
        "//conditions:default": [
        ],
    }),
)

cc_library_ydf(
    name = "export_doc",
    srcs = [
        "export_doc.cc",
    ],
    hdrs = [
        "export_doc.h",
    ],
    deps = [
        ":abstract_learner",
        ":abstract_learner_cc_proto",
        ":learner_library",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
)

# Proto
# ========

all_proto_library(
    name = "abstract_learner_proto",
    srcs = ["abstract_learner.proto"],
    exports = ["//yggdrasil_decision_forests/model:hyperparameter_proto"],
    deps = [
        "//yggdrasil_decision_forests/dataset:data_spec_proto",
        "//yggdrasil_decision_forests/dataset:weight_proto",
        "//yggdrasil_decision_forests/model:abstract_model_proto",
        "//yggdrasil_decision_forests/model:hyperparameter_proto",
        "//yggdrasil_decision_forests/utils/distribute:distribute_proto",
    ],
)

# Tests
# =====

cc_test(
    name = "abstract_learner_test",
    srcs = ["abstract_learner_test.cc"],
    deps = [
        ":abstract_learner",
        ":abstract_learner_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:weight_cc_proto",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/metric:metric_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:model_testing",
        "//yggdrasil_decision_forests/model:prediction_cc_proto",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
    ],
)

cc_test(
    name = "learner_library_test",
    srcs = ["learner_library_test.cc"],
    deps = [
        ":abstract_learner",
        ":abstract_learner_cc_proto",
        ":learner_library",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/utils:test",
    ],
)

cc_test(
    name = "export_doc_test",
    srcs = ["export_doc_test.cc"],
    deps = [
        ":abstract_learner",
        ":abstract_learner_cc_proto",
        ":export_doc",
        "@com_google_googletest//:gtest_main",
    ],
)
