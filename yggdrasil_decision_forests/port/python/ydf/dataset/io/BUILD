# I/O bindings
load("@ydf_cc//yggdrasil_decision_forests/utils:compile.bzl", "py_library_ydf", "py_test_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library_ydf(
    name = "init",
    srcs = ["__init__.py"],
)

# Libraries
# =========

# Note: Pandas and Xarray are optional dependencies of YDF. This build rule only imports them for
# type checking, and the connectors (e.g. ":pandas_io", ":xarray_io") import them lazily. Users who
# pass a Pandas DataFrame or a Xarray Dataset to YDF already depend on those libraries. Do not add
# //third_party/py/pandas or //third_party/py/xarray here (build_cleaner may suggest it): they
# would be linked into every YDF binary.
py_library_ydf(
    name = "dataset_io_types",
    srcs = ["dataset_io_types.py"],
    deps = [
        "@pypi_ydf//numpy",
        # "@pypi_ydf//polars",   # TODO: Re-enable.
    ],
)

py_library_ydf(
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

py_library_ydf(
    name = "pandas_io",
    srcs = ["pandas_io.py"],
    deps = [
        ":dataset_io_types",
        ":generator",
        "@abseil-py//absl/logging",
        "@pypi_ydf//numpy",
    ],
)

py_library_ydf(
    name = "polars_io",
    srcs = ["polars_io.py"],
    deps = [
        ":dataset_io_types",
        "@abseil-py//absl/logging",
    ],
)

py_library_ydf(
    name = "pyarrow_io",
    srcs = ["pyarrow_io.py"],
    deps = [
        ":dataset_io_types",
        "@abseil-py//absl/logging",
        "@pypi_ydf//numpy",
    ],
)

py_library_ydf(
    name = "xarray_io",
    srcs = ["xarray_io.py"],
    deps = [
        ":dataset_io_types",
        "@abseil-py//absl/logging",
    ],
)

py_library_ydf(
    name = "tensorflow_io",
    srcs = ["tensorflow_io.py"],
    deps = [":dataset_io_types"],
)

py_library_ydf(
    name = "pygrain_io",
    srcs = ["pygrain_io.py"],
    deps = [
        ":dataset_io_types",
        "@abseil-py//absl/logging",
        "@pypi_ydf//numpy",
    ],
)

py_library_ydf(
    name = "generator",
    srcs = ["generator.py"],
    deps = [
        "@pypi_ydf//numpy",
    ],
)

py_library_ydf(
    name = "numpy_io",
    srcs = ["numpy_io.py"],
    deps = [
        ":generator",
        "@pypi_ydf//numpy",
    ],
)

# Tests
# =====

py_test_ydf(
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

py_test_ydf(
    name = "polars_io_test",
    srcs = ["polars_io_test.py"],
    deps = [
        ":polars_io",
        "@abseil-py//absl/testing:absltest",
        "@pypi_ydf//pandas",
        # "@pypi_ydf//polars",  # TODO: Re-enable.
    ],
)

py_test_ydf(
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

py_test_ydf(
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

py_test_ydf(
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

py_test_ydf(
    name = "dataset_io_test",
    srcs = ["dataset_io_test.py"],
    deps = [
        ":dataset_io",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//numpy",
    ],
)

py_test_ydf(
    name = "numpy_io_test",
    srcs = ["numpy_io_test.py"],
    deps = [
        ":numpy_io",
        "//ydf/utils:test_utils",
        "@abseil-py//absl/testing:absltest",
        "@pypi_ydf//numpy",
    ],
)
