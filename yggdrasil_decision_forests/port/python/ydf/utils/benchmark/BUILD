# pytype binary only

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_binary(
    name = "benchmark",
    srcs = ["benchmark.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # pandas dep,
        # tensorflow_decision_forests dep,
        "//ydf/dataset",
        "//ydf/learner:generic_learner",
        "//ydf/learner:specialized_learners",
    ],
)
