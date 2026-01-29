load("@pybind11_bazel//:build_defs.bzl", "pybind_library")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

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
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest:random_forest_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
    ],
)

py_library(
    name = "random_forest_model",
    srcs = ["random_forest_model.py"],
    deps = [
        "//ydf/cc:ydf",
        "//ydf/metric",
        "//ydf/model:generic_model",
        "//ydf/model/decision_forest_model",
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest:random_forest_py_proto",
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
    deps = [
        # absl/testing:absltest dep,
        # numpy dep,
        # pandas dep,
        "//ydf/model:model_lib",
        "//ydf/utils:test_utils",
    ],
)
