# pytype binary, test, library

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Binaries
# ========

py_binary(
    name = "benchmark_train_speed_main",
    srcs = ["benchmark_train_speed_main.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    deps = [
        ":benchmark_train_speed",
        # absl:app dep,
        # absl/flags dep,
    ],
)

py_binary(
    name = "benchmark_io_speed_main",
    srcs = ["benchmark_io_speed_main.py"],
    deps = [
        ":benchmark_io_speed",
        # absl:app dep,
        # absl/flags dep,
        "//ydf/utils:log",
    ],
)

# Libraries
# =========

py_library(
    name = "benchmark_io_speed",
    srcs = ["benchmark_io_speed.py"],
    deps = [
        # numpy dep,
        "//ydf/dataset",
        "//ydf/util:tf_example",
        "//ydf/utils:log",
    ],
)

py_library(
    name = "benchmark_train_speed",
    srcs = ["benchmark_train_speed.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # pandas dep,
        # tensorflow_decision_forests dep,
        # ydf dep,
    ],
)

py_library(
    name = "benchmark_inference_speed",
    srcs = ["benchmark_inference_speed.py"],
    deps = [
        # jax dep,
        # numpy dep,
        # tensorflow:tensorflow_no_contrib dep,
        # tensorflow_decision_forests dep,
        # ydf dep,
        "//ydf/dataset",
        "//ydf/learner:specialized_learners",
        "//ydf/model:export_jax",  # buildcleaner: keep
        "//ydf/model:export_tf",  # buildcleaner: keep
        "//ydf/model:generic_model",
        "//ydf/utils:string_lib",
    ],
)

# Tests
# =====

py_test(
    name = "benchmark_inference_speed_test",
    srcs = ["benchmark_inference_speed_test.py"],
    python_version = "PY3",
    deps = [
        ":benchmark_inference_speed",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
    ],
)
