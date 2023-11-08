# I/O bindings
# pytype test and library

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library(
    name = "dataset_io_types",
    srcs = ["dataset_io_types.py"],
    deps = [
        # numpy dep,
        # pandas dep,
    ],
)

py_library(
    name = "dataset_io",
    srcs = ["dataset_io.py"],
    deps = [
        ":dataset_io_types",
        ":pandas_io",
        ":tensorflow_io",
        # numpy dep,
    ],
)

py_library(
    name = "pandas_io",
    srcs = ["pandas_io.py"],
    deps = [
        ":dataset_io_types",
        # absl/logging dep,
        # pandas dep,
    ],
)

py_test(
    name = "pandas_io_test",
    srcs = ["pandas_io_test.py"],
    deps = [
        ":pandas_io",
        # absl/testing:absltest dep,
        # pandas dep,
    ],
)

py_library(
    name = "tensorflow_io",
    srcs = ["tensorflow_io.py"],
    deps = [":dataset_io_types"],
)
