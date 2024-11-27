# pytype test and library
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "evaluate",
    srcs = ["evaluate.py"],
    deps = [
        ":metric",
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_py_proto",
        "//ydf/cc:ydf",
        "//ydf/model:generic_model",
        "//ydf/utils:concurrency",
    ],
)

py_library(
    name = "metric",
    srcs = [
        "display_metric.py",
        "metric.py",
    ],
    deps = [
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_py_proto",
        "//ydf/cc:ydf",
        "//ydf/dataset:dataspec",
        "//ydf/utils:documentation",
        "//ydf/utils:html",
        "//ydf/utils:string_lib",
    ],
)

pybind_library(
    name = "metric_cc",
    srcs = ["metric.cc"],
    hdrs = ["metric.h"],
    deps = [
        "//ydf/utils:status_casters",
        "@com_google_absl//absl/status:statusor",
        "@com_google_pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/metric:report",
        "@ydf_cc//yggdrasil_decision_forests/utils:status_macros",
    ],
)

pybind_library(
    name = "evaluate_cc",
    srcs = ["evaluate.cc"],
    hdrs = ["evaluate.h"],
    deps = [
        "//ydf/utils:numpy_data",
        "//ydf/utils:status_casters",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec",
        "@ydf_cc//yggdrasil_decision_forests/metric",
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:distribution",
        "@ydf_cc//yggdrasil_decision_forests/utils:random",
        "@ydf_cc//yggdrasil_decision_forests/utils:status_macros",
    ],
)

# Tests
# =====

py_test(
    name = "metric_test",
    srcs = ["metric_test.py"],
    data = ["//test_data"],
    python_version = "PY3",
    deps = [
        ":metric",
        # absl/testing:absltest dep,
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:distribution_py_proto",
    ],
)

py_test(
    name = "evaluate_test",
    srcs = ["evaluate_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    deps = [
        ":evaluate",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # pandas dep,
        # sklearn dep,
        "//ydf/model:generic_model",
        "//ydf/model:model_lib",
        "//ydf/utils:test_utils",
    ],
)
