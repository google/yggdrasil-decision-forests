# pytype binary, test, library

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library(
    name = "benchmark",
    srcs = ["benchmark.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # pandas dep,
        # tensorflow_decision_forests dep,
        # ydf dep,
    ],
)

py_binary(
    name = "benchmark_main",
    srcs = ["benchmark_main.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    deps = [
        ":benchmark",
        # absl:app dep,
        # absl/flags dep,
    ],
)
