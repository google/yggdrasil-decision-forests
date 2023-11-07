# TODO: describe this package.

# pytype test and library

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# User facing API of pydf.
# TODO: Create alias for the library to be imported with "import ydf".
py_library(
    name = "api",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":version",
        "//ydf/dataset",
        "//ydf/dataset:dataspec",
        "//ydf/learner:generic_learner",
        "//ydf/learner:specialized_learners",
        "//ydf/learner:tuner",
        "//ydf/learner:worker",
        "//ydf/model:generic_model",
        "//ydf/model:gradient_boosted_trees_model",
        "//ydf/model:model_lib",
        "//ydf/model:random_forest_model",
        "//ydf/utils:log",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
)

# Tests
# =====

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":api",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # pandas dep,
        "//ydf/utils:test_utils",
    ],
)
