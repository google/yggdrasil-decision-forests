# Model bindings for PYDF
# pytype test and library
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")
load("@ydf_cc//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

# Add new models here.
cc_library_ydf(
    name = "pydf_models",
    deps = [
        "@ydf_cc//yggdrasil_decision_forests/model/decision_tree:decision_forest_interface",
        "@ydf_cc//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest",
    ],
)

pybind_library(
    name = "model_wrapper",
    srcs = ["model_wrapper.cc"],
    hdrs = ["model_wrapper.h"],
    deps = [
        ":pydf_models",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:vertical_dataset",
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model",
        "@ydf_cc//yggdrasil_decision_forests/model:model_library",
        "@ydf_cc//yggdrasil_decision_forests/model/decision_tree:decision_forest_interface",
        "@ydf_cc//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest",
        "@ydf_cc//yggdrasil_decision_forests/serving:example_set",
        "@ydf_cc//yggdrasil_decision_forests/serving:fast_engine",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
        "@ydf_cc//yggdrasil_decision_forests/utils:random",
        "@ydf_cc//yggdrasil_decision_forests/utils:status_macros",
    ],
)

pybind_library(
    name = "model_cc",
    srcs = ["model.cc"],
    hdrs = ["model.h"],
    deps = [
        ":model_wrapper",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_pybind11_abseil//pybind11_abseil:status_casters",
        "@com_google_pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model",
        "@ydf_cc//yggdrasil_decision_forests/model:model_library",
        "@ydf_cc//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest",
    ],
)

py_library(
    name = "generic_model",
    srcs = ["generic_model.py"],
    deps = [
        # absl/logging dep,
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_py_proto",
        "//ydf/cc:ydf",
        "//ydf/dataset",
        "//ydf/metric",
    ],
)

py_library(
    name = "decision_forest_model",
    srcs = ["decision_forest_model.py"],
    deps = [
        ":generic_model",
        "//ydf/cc:ydf",
    ],
)

py_library(
    name = "random_forest_model",
    srcs = ["random_forest_model.py"],
    deps = [
        ":decision_forest_model",
        "//ydf/cc:ydf",
    ],
)

py_library(
    name = "gradient_boosted_trees_model",
    srcs = ["gradient_boosted_trees_model.py"],
    deps = [
        ":decision_forest_model",
        "//ydf/cc:ydf",
    ],
)

py_library(
    name = "model_lib",
    srcs = ["model_lib.py"],
    deps = [
        ":generic_model",
        ":gradient_boosted_trees_model",
        ":random_forest_model",
        # absl/logging dep,
        "//ydf/cc:ydf",
    ],
)
# Tests
# =====

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    data = [
        "@ydf_cc//yggdrasil_decision_forests/test_data",
        # absl/flags dep,
        # absl/testing:absltest dep,
        # numpy dep,
        # pandas dep,
    ],
    python_version = "PY3",
    deps = [
        ":generic_model",
        ":model_lib",
        ":random_forest_model",
    ],
)
