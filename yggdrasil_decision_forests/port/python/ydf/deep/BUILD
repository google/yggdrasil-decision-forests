# Dataset bindings for PYDF
# pytype test and library

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        # jax dep,
        "//ydf/dataset/io:generator",
    ],
)

py_library(
    name = "metric",
    srcs = ["metric.py"],
    deps = [
        # jax dep,
        # optax dep,
    ],
)

py_library(
    name = "preprocessor",
    srcs = ["preprocessor.py"],
    deps = [
        ":dataset",
        ":layer",
        # jax dep,
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
    ],
)

py_library(
    name = "layer",
    srcs = ["layer.py"],
    deps = [
        # flax dep,
        # jax dep,
    ],
)

py_library(
    name = "hyperparameter",
    srcs = ["hyperparameter.py"],
    deps = [],
)

# Tests
# =====

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    python_version = "PY3",
    deps = [
        ":dataset",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
    ],
)

py_test(
    name = "preprocessor_test",
    srcs = ["preprocessor_test.py"],
    python_version = "PY3",
    deps = [
        ":dataset",
        ":layer",
        ":preprocessor",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "//ydf/utils:test_utils",
    ],
)

py_test(
    name = "hyperparameter_test",
    srcs = ["hyperparameter_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    deps = [
        ":hyperparameter",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
    ],
)

py_test(
    name = "layer_test",
    srcs = ["layer_test.py"],
    python_version = "PY3",
    deps = [
        ":layer",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
    ],
)

py_test(
    name = "metric_test",
    srcs = ["metric_test.py"],
    python_version = "PY3",
    deps = [
        ":metric",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
    ],
)
