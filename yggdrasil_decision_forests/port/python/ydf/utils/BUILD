load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "string_lib",
    srcs = ["string_lib.py"],
)

py_library(
    name = "html",
    srcs = ["html.py"],
)

py_library(
    name = "paths",
    srcs = ["paths.py"],
)

py_library(
    name = "documentation",
    srcs = ["documentation.py"],
)

py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    deps = [
        "//ydf/dataset",
        "//ydf/dataset:dataspec",
        "@abseil-py//absl/flags",
        "@abseil-py//absl/testing:absltest",
        "@pypi_ydf//jax",
        "@pypi_ydf//numpy",
        "@pypi_ydf//pandas",
        "@pypi_ydf//scikit_learn",
    ],
)

py_library(
    name = "log",
    srcs = ["log.py"],
    deps = [
        "//ydf/cc:ydf",
        "@abseil-py//absl/logging",
    ],
)

py_library(
    name = "func_helpers",
    srcs = ["func_helpers.py"],
    deps = [],
)

pybind_library(
    name = "custom_casters",
    hdrs = ["custom_casters.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

pybind_library(
    name = "log_cc",
    srcs = ["log.cc"],
    hdrs = ["log.h"],
    deps = ["@ydf_cc//yggdrasil_decision_forests/utils:logging"],
)

pybind_library(
    name = "numpy_data",
    srcs = ["numpy_data.cc"],
    hdrs = ["numpy_data.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
    ],
)

pybind_library(
    name = "status_casters",
    hdrs = ["status_casters.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

pybind_library(
    name = "pybind",
    hdrs = ["pybind.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

py_library(
    name = "concurrency",
    srcs = ["concurrency.py"],
    deps = [":log"],
)

py_library(
    name = "filesystem",
    srcs = ["filesystem.py"],
    deps = [
        ":filesystem_default",
    ],
)

py_library(
    name = "filesystem_default",
    srcs = ["filesystem_default.py"],
    deps = [],
)

# Tests
# =====

py_test(
    name = "filesystem_test",
    srcs = ["filesystem_test.py"],
    deps = [
        ":filesystem",
        "@abseil-py//absl/testing:parameterized",
    ],
)

py_test(
    name = "func_helpers_test",
    srcs = ["func_helpers_test.py"],
    deps = [
        ":func_helpers",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_test(
    name = "html_test",
    srcs = ["html_test.py"],
    deps = [
        ":html",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_test(
    name = "paths_test",
    srcs = ["paths_test.py"],
    deps = [
        ":paths",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_test(
    name = "string_lib_test",
    srcs = ["string_lib_test.py"],
    deps = [
        ":string_lib",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.py"],
    deps = [
        ":test_utils",
        "@abseil-py//absl/testing:absltest",
        "@pypi_ydf//numpy",
    ],
)

pybind_extension(
    name = "pybind_test_helper",
    testonly = True,
    srcs = ["pybind_test_helper.cc"],
    deps = [
        ":pybind",
        ":status_casters",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

py_test(
    name = "pybind_test",
    srcs = ["pybind_test.py"],
    deps = [
        ":pybind_test_helper",
        "@abseil-py//absl/testing:absltest",
    ],
)
