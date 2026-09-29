load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

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
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)

py_binary(
    name = "benchmark_io_speed_main",
    srcs = ["benchmark_io_speed_main.py"],
    deps = [
        ":benchmark_io_speed",
        "//ydf/utils:log",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)

# Libraries
# =========

py_library(
    name = "benchmark_io_speed",
    srcs = ["benchmark_io_speed.py"],
    deps = [
        "//ydf/dataset",
        "//ydf/util:tf_example",
        "//ydf/utils:log",
        "@pypi_ydf//numpy",
    ],
)

py_library(
    name = "benchmark_train_speed",
    srcs = ["benchmark_train_speed.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    deps = [
        "//ydf:api",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
        "@pypi_ydf//pandas",
    ],
)

py_library(
    name = "benchmark_inference_speed",
    srcs = ["benchmark_inference_speed.py"],
    tags = ["tf_dep"],
    deps = [
        "@pypi_ydf//jax",
        "@pypi_ydf//numpy",
        "@pypi_ydf//tensorflow:pkg",
        # tensorflow_decision_forests dep,
        "//ydf:api",
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
    tags = ["tf_dep"],
    deps = [
        ":benchmark_inference_speed",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//numpy",
    ],
)
