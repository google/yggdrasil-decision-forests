# pytype test and library
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

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
