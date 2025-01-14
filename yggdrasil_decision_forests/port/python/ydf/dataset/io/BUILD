# I/O bindings
# pytype test and library

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "dataset_io_types",
    srcs = ["dataset_io_types.py"],
    deps = [
        # numpy dep,
        # pandas dep,
        # # polars dep,   # TODO: Re-enable.
        # xarray dep,
    ],
)

py_library(
    name = "dataset_io",
    srcs = ["dataset_io.py"],
    deps = [
        ":dataset_io_types",
        ":numpy_io",
        ":pandas_io",
        ":polars_io",
        ":pygrain_io",
        ":tensorflow_io",
        ":xarray_io",
        # numpy dep,
    ],
)

py_library(
    name = "pandas_io",
    srcs = ["pandas_io.py"],
    deps = [
        ":dataset_io_types",
        ":generator",
        # absl/logging dep,
        # numpy dep,
    ],
)

py_library(
    name = "polars_io",
    srcs = ["polars_io.py"],
    deps = [
        ":dataset_io_types",
        # absl/logging dep,
    ],
)

py_library(
    name = "xarray_io",
    srcs = ["xarray_io.py"],
    deps = [
        ":dataset_io_types",
        # absl/logging dep,
        # xarray dep,
    ],
)

py_library(
    name = "tensorflow_io",
    srcs = ["tensorflow_io.py"],
    deps = [":dataset_io_types"],
)

py_library(
    name = "pygrain_io",
    srcs = ["pygrain_io.py"],
    deps = [
        ":dataset_io_types",
        # absl/logging dep,
        # numpy dep,
    ],
)

py_library(
    name = "generator",
    srcs = ["generator.py"],
    deps = [
        # numpy dep,
    ],
)

py_library(
    name = "numpy_io",
    srcs = ["numpy_io.py"],
    deps = [
        ":generator",
        # numpy dep,
    ],
)

# Tests
# =====

py_test(
    name = "pandas_io_test",
    srcs = ["pandas_io_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    deps = [
        ":pandas_io",
        # absl/testing:absltest dep,
        # pandas dep,
        # # polars dep,  # TODO: Re-enable.
        "//ydf/utils:test_utils",
    ],
)

py_test(
    name = "polars_io_test",
    srcs = ["polars_io_test.py"],
    deps = [
        ":polars_io",
        # absl/testing:absltest dep,
        # pandas dep,
        # # polars dep,  # TODO: Re-enable.
    ],
)

py_test(
    name = "xarray_io_test",
    srcs = ["xarray_io_test.py"],
    deps = [
        ":dataset_io_types",
        ":xarray_io",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # pandas dep,
        # xarray dep,
    ],
)

py_test(
    name = "pygrain_io_test",
    srcs = ["pygrain_io_test.py"],
    # TODO: Figure out what to do with Pygrain support, since it does not work on MacOS.
    tags = ["manual"],  # Grain is not supported on MacOS
    deps = [
        ":dataset_io_types",
        ":pygrain_io",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # grain:python dep,
        # jax dep,  # buildcleaner: keep
        # numpy dep,
    ],
)

py_test(
    name = "dataset_io_test",
    srcs = ["dataset_io_test.py"],
    deps = [
        ":dataset_io",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
    ],
)

py_test(
    name = "numpy_io_test",
    srcs = ["numpy_io_test.py"],
    deps = [
        ":numpy_io",
        # absl/testing:absltest dep,
        # numpy dep,
        "//ydf/utils:test_utils",
    ],
)
