# pytype test and library
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

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
        # absl/flags dep,
        # absl/testing:absltest dep,
        # jax dep,
        # numpy dep,
        # pandas dep,
        # sklearn dep,
        "//ydf/dataset",
        "//ydf/dataset:dataspec",
    ],
)

py_library(
    name = "log",
    srcs = ["log.py"],
    deps = [
        # absl/logging dep,
        # tqdm dep,
        "//ydf/cc:ydf",
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
        # absl/testing:parameterized dep,
    ],
)

py_test(
    name = "func_helpers_test",
    srcs = ["func_helpers_test.py"],
    deps = [
        ":func_helpers",
        # absl/testing:absltest dep,
    ],
)

py_test(
    name = "html_test",
    srcs = ["html_test.py"],
    deps = [
        ":html",
        # absl/testing:absltest dep,
    ],
)

py_test(
    name = "paths_test",
    srcs = ["paths_test.py"],
    deps = [
        ":paths",
        # absl/testing:absltest dep,
    ],
)

py_test(
    name = "string_lib_test",
    srcs = ["string_lib_test.py"],
    deps = [
        ":string_lib",
        # absl/testing:absltest dep,
    ],
)

py_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.py"],
    deps = [
        ":test_utils",
        # absl/testing:absltest dep,
        # numpy dep,
    ],
)
