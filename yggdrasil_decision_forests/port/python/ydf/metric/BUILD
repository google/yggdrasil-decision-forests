# pytype test and library

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
        # matplotlib dep,
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_py_proto",
        "//ydf/dataset:dataspec",
        "//ydf/utils:documentation",
        "//ydf/utils:html",
        "//ydf/utils:string_lib",
    ],
)

# Tests
# =====

py_test(
    name = "metric_test",
    srcs = ["metric_test.py"],
    data = ["//ydf/test_data"],
    python_version = "PY3",
    deps = [
        ":metric",
        # absl/testing:absltest dep,
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_py_proto",
        "//ydf/utils:test_utils",
        "@ydf_cc//yggdrasil_decision_forests/utils:distribution_py_proto",
    ],
)
