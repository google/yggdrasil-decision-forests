load("@rules_python//python:py_library.bzl", "py_library")

# I/O bindings
load("@rules_python//python:py_test.bzl", "py_test")

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
        "@pypi_ydf//numpy",
        "@pypi_ydf//pandas",
        # "@pypi_ydf//polars",   # TODO: Re-enable.
        "@pypi_ydf//xarray",
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
        ":pyarrow_io",
        ":pygrain_io",
        ":tensorflow_io",
        ":xarray_io",
        "@pypi_ydf//numpy",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
    ],
)

py_library(
    name = "pandas_io",
    srcs = ["pandas_io.py"],
    deps = [
        ":dataset_io_types",
        ":generator",
        "@abseil-py//absl/logging",
        "@pypi_ydf//numpy",
    ],
)

py_library(
    name = "polars_io",
    srcs = ["polars_io.py"],
    deps = [
        ":dataset_io_types",
        "@abseil-py//absl/logging",
    ],
)

py_library(
    name = "pyarrow_io",
    srcs = ["pyarrow_io.py"],
    deps = [
        ":dataset_io_types",
        "@abseil-py//absl/logging",
        "@pypi_ydf//numpy",
    ],
)

py_library(
    name = "xarray_io",
    srcs = ["xarray_io.py"],
    deps = [
        ":dataset_io_types",
        "@abseil-py//absl/logging",
        "@pypi_ydf//xarray",
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
        "@abseil-py//absl/logging",
        "@pypi_ydf//numpy",
    ],
)

py_library(
    name = "generator",
    srcs = ["generator.py"],
    deps = [
        "@pypi_ydf//numpy",
    ],
)

py_library(
    name = "numpy_io",
    srcs = ["numpy_io.py"],
    deps = [
        ":generator",
        "@pypi_ydf//numpy",
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
        "@abseil-py//absl/testing:absltest",
        "@pypi_ydf//pandas",
        # "@pypi_ydf//polars",  # TODO: Re-enable.
        "//ydf/utils:test_utils",
    ],
)

py_test(
    name = "polars_io_test",
    srcs = ["polars_io_test.py"],
    deps = [
        ":polars_io",
        "@abseil-py//absl/testing:absltest",
        "@pypi_ydf//pandas",
        # "@pypi_ydf//polars",  # TODO: Re-enable.
    ],
)

py_test(
    name = "pyarrow_io_test",
    srcs = ["pyarrow_io_test.py"],
    deps = [
        ":pyarrow_io",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//numpy",
        "@pypi_ydf//pandas",
        "@pypi_ydf//pyarrow",
    ],
)

py_test(
    name = "xarray_io_test",
    srcs = ["xarray_io_test.py"],
    deps = [
        ":dataset_io_types",
        ":xarray_io",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//numpy",
        "@pypi_ydf//pandas",
        "@pypi_ydf//xarray",
    ],
)

py_test(
    name = "pygrain_io_test",
    srcs = ["pygrain_io_test.py"],
    # TODO: Figure out what to do with Pygrain support, since it does not work on MacOS.
    tags = [
        "grain_dep",
        "manual",
    ],  # Grain is not supported on MacOS
    deps = [
        ":dataset_io_types",
        ":pygrain_io",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//grain",
        "@pypi_ydf//jax",  # buildcleaner: keep
        "@pypi_ydf//numpy",
    ],
)

py_test(
    name = "dataset_io_test",
    srcs = ["dataset_io_test.py"],
    deps = [
        ":dataset_io",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//numpy",
    ],
)

py_test(
    name = "numpy_io_test",
    srcs = ["numpy_io_test.py"],
    deps = [
        ":numpy_io",
        "//ydf/utils:test_utils",
        "@abseil-py//absl/testing:absltest",
        "@pypi_ydf//numpy",
    ],
)
