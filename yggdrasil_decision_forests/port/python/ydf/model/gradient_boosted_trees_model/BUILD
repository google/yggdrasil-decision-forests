# pytype test and library
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

pybind_library(
    name = "gradient_boosted_trees_wrapper",
    srcs = ["gradient_boosted_trees_wrapper.cc"],
    hdrs = ["gradient_boosted_trees_wrapper.h"],
    deps = [
        "//ydf/model/decision_forest_model:decision_forest_wrapper",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model",
        "@ydf_cc//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
    ],
)

py_library(
    name = "gradient_boosted_trees_model",
    srcs = ["gradient_boosted_trees_model.py"],
    deps = [
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_py_proto",
        "//ydf/cc:ydf",
        "//ydf/learner:custom_loss_py",
        "//ydf/metric",
        "//ydf/model:generic_model",
        "//ydf/model/decision_forest_model",
    ],
)

# Tests
# =====

py_test(
    name = "gradient_boosted_trees_model_test",
    srcs = ["gradient_boosted_trees_model_test.py"],
    data = [
        "//test_data",
        "@ydf_cc//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":gradient_boosted_trees_model",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # pandas dep,
        "//ydf/dataset:dataspec",
        "//ydf/learner:custom_loss_py",
        "//ydf/learner:specialized_learners",
        "//ydf/model:generic_model",
        "//ydf/model:model_lib",
        "//ydf/model/tree",
        "//ydf/model/tree:condition",
        "//ydf/model/tree:node",
        "//ydf/model/tree:value",
        "//ydf/utils:test_utils",
    ],
)
