# Dataset bindings for PYDF
# pytype test and library

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "metric",
    srcs = ["metric.py"],
    deps = [
        # jax dep,
        # optax dep,
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
