# Dataset bindings for PYDF
# pytype test and library
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@ydf_cc//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_inference",
        "@ydf_cc//yggdrasil_decision_forests/dataset:formats",
        "@ydf_cc//yggdrasil_decision_forests/dataset:formats_cc_proto",
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
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "//ydf/cc:ydf",
        "//ydf/dataset/io:dataset_io",
        "//ydf/dataset/io:dataset_io_types",
        "//ydf/utils:log",
        "//ydf/utils:paths",
    ],
)

py_library(
    name = "dataspec",
    srcs = ["dataspec.py"],
    deps = [
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
    ],
)
# Tests
# =====

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    deps = [
        ":dataset",
        ":dataspec",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # pandas dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "//ydf/utils:test_utils",
    ],
)

py_test(
    name = "dataset_with_tf_test",
    srcs = ["dataset_with_tf_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    deps = [
        ":dataset",
        # absl/testing:absltest dep,
        # numpy dep,
        # tensorflow dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
    ],
)

py_test(
    name = "dataspec_test",
    srcs = ["dataspec_test.py"],
    python_version = "PY3",
    deps = [
        ":dataspec",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
    ],
)
