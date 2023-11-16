# pytype test and library
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

pybind_library(
    name = "random_forest_wrapper",
    srcs = ["random_forest_wrapper.cc"],
    hdrs = ["random_forest_wrapper.h"],
    deps = [
        "//ydf/model/decision_forest_model:decision_forest_wrapper",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model",
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
    ],
)

py_library(
    name = "random_forest_model",
    srcs = ["random_forest_model.py"],
    deps = [
        "//ydf/cc:ydf",
        "//ydf/model/decision_forest_model",
    ],
)

# Tests
# =====

py_test(
    name = "random_forest_model_test",
    srcs = ["random_forest_model_test.py"],
    data = [
        "//test_data",
        "@ydf_cc//yggdrasil_decision_forests/test_data",
    ],
    python_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        # numpy dep,
        # pandas dep,
        "//ydf/model:model_lib",
        "//ydf/utils:test_utils",
    ],
)
