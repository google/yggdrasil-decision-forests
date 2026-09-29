load("@pybind11_bazel//:build_defs.bzl", "pybind_library")
load("@rules_python//python:py_library.bzl", "py_library")

# Dataset bindings for PYDF
load("@rules_python//python:py_test.bzl", "py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

pybind_library(
    name = "dataset_cc",
    srcs = ["dataset.cc"],
    hdrs = ["dataset.h"],
    deps = [
        "//ydf/utils:numpy_data",
        "//ydf/utils:status_casters",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@ydf_cc//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_inference",
        "@ydf_cc//yggdrasil_decision_forests/dataset:formats",
        "@ydf_cc//yggdrasil_decision_forests/dataset:formats_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:types",
        "@ydf_cc//yggdrasil_decision_forests/dataset:vertical_dataset",
        "@ydf_cc//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
        "@ydf_cc//yggdrasil_decision_forests/utils:status_macros",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        ":dataspec",
        "//ydf/cc:ydf",
        "//ydf/dataset/io:dataset_io",
        "//ydf/dataset/io:dataset_io_types",
        "//ydf/dataset/io:generator",
        "//ydf/utils:log",
        "//ydf/utils:paths",
        "@pypi_ydf//numpy",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
    ],
)

py_library(
    name = "dataspec",
    srcs = ["dataspec.py"],
    deps = [
        "//ydf/dataset/io:dataset_io_types",
        "//ydf/utils:log",
        "@pypi_ydf//numpy",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
    ],
)
# Tests
# =====

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    data = [
        "//test_data",
        "@ydf_cc//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":dataset",
        ":dataspec",
        "//ydf/dataset/io:dataset_io",
        "//ydf/utils:test_utils",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//fastavro",
        "@pypi_ydf//numpy",
        "@pypi_ydf//pandas",
        "@pypi_ydf//pyarrow",
        "@pypi_ydf//pyarrow:pkg",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
    ],
)

py_test(
    name = "dataset_with_tf_test",
    srcs = ["dataset_with_tf_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    tags = ["tf_dep"],
    deps = [
        ":dataset",
        "@abseil-py//absl/testing:absltest",
        "@pypi_ydf//numpy",
        "@pypi_ydf//tensorflow",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
    ],
)

py_test(
    name = "dataspec_test",
    srcs = ["dataspec_test.py"],
    deps = [
        ":dataspec",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//numpy",
        "@pypi_ydf//pandas",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
    ],
)
