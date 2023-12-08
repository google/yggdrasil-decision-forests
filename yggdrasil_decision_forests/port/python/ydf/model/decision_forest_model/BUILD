# pytype test and library
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

pybind_library(
    name = "decision_forest_wrapper",
    srcs = ["decision_forest_wrapper.cc"],
    hdrs = ["decision_forest_wrapper.h"],
    deps = [
        "//ydf/model:model_wrapper",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@ydf_cc//yggdrasil_decision_forests/dataset:vertical_dataset",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model",
        "@ydf_cc//yggdrasil_decision_forests/model/decision_tree:decision_forest_interface",
        "@ydf_cc//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
        "@ydf_cc//yggdrasil_decision_forests/utils:protobuf",
        "@ydf_cc//yggdrasil_decision_forests/utils:status_macros",
    ],
)

py_library(
    name = "decision_forest_model",
    srcs = ["decision_forest_model.py"],
    deps = [
        # numpy dep,
        "//ydf/cc:ydf",
        "//ydf/dataset",
        "//ydf/model:generic_model",
        "//ydf/model/tree",
    ],
)

# Tests
# =====

py_test(
    name = "decision_forest_model_test",
    srcs = ["decision_forest_model_test.py"],
    data = [
        "//test_data",
        "@ydf_cc//yggdrasil_decision_forests/test_data",
    ],
    python_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # pandas dep,
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest:random_forest_py_proto",
        "//ydf/model:generic_model",
        "//ydf/model:model_lib",
        "//ydf/utils:test_utils",
    ],
)
