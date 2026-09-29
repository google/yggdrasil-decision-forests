load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "tf_example",
    srcs = ["tf_example.py"],
    deps = [
        ":dataset_io",
        ":tf_example_impl",
        "//ydf/utils:log",
    ],
)

# Note: This build rule does not depends on TF. To use its functionalities, TF needs to be
# imported manually by the call i.e. //third_party/py/tensorflow.
py_library(
    name = "tf_example_impl",
    srcs = ["tf_example_impl.py"],
    deps = [
        ":dataset_io",
        "//ydf/utils:log",
        "@pypi_ydf//numpy",
    ],
)

py_library(
    name = "dataset_io",
    srcs = ["dataset_io.py"],
    deps = [
        "@pypi_ydf//numpy",
    ],
)

py_library(
    name = "help",
    srcs = ["help.py"],
    deps = [],
)

py_library(
    name = "vertex_ai",
    srcs = ["vertex_ai.py"],
    deps = [],
)

py_library(
    name = "log_book",
    srcs = ["log_book.py"],
    deps = [
        # "@pypi_ydf//pandas",
    ],
)

# Tests
# =====

py_test(
    name = "tf_example_test",
    srcs = ["tf_example_test.py"],
    data = [
        "//test_data",
        "@ydf_cc//yggdrasil_decision_forests/test_data",
    ],
    tags = ["tf_dep"],
    deps = [
        ":tf_example",
        "//ydf/utils:test_utils",
        "@abseil-py//absl/logging",
        "@abseil-py//absl/testing:absltest",
        "@pypi_ydf//numpy",
        "@pypi_ydf//tensorflow",
    ],
)

py_test(
    name = "vertex_ai_test",
    srcs = ["vertex_ai_test.py"],
    deps = [
        ":vertex_ai",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_test(
    name = "log_book_test",
    srcs = ["log_book_test.py"],
    deps = [
        ":log_book",
        "@abseil-py//absl/logging",
        "@abseil-py//absl/testing:absltest",
        "@pypi_ydf//pandas",
    ],
)
