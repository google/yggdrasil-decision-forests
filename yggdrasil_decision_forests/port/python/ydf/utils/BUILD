# pytype test and library

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
    name = "documentation",
    srcs = ["documentation.py"],
)

py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    data = [
        # absl/flags dep,
        # absl/testing:absltest dep,
        "@com_google_pybind11_abseil//pybind11_abseil:status.so",
    ],
)

# Tests
# =====

py_test(
    name = "string_lib_test",
    srcs = ["string_lib_test.py"],
    python_version = "PY3",
    deps = [
        ":string_lib",
        # absl/testing:absltest dep,
    ],
)

py_test(
    name = "html_test",
    srcs = ["html_test.py"],
    python_version = "PY3",
    deps = [
        ":html",
        # absl/testing:absltest dep,
    ],
)
