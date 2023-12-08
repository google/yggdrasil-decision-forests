# Representation of a decision tree as a python object

# pytype test and library

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "all",  # TODO: Replace "tree"
    srcs = ["__init__.py"],
    deps = [
        ":condition",
        ":node",
        ":tree",
        ":value",
    ],
)

py_library(
    name = "tree",
    srcs = ["tree.py"],
    deps = [
        ":condition",
        ":node",
        ":value",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
    ],
)

py_library(
    name = "value",
    srcs = ["value.py"],
    deps = [
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
    ],
)

py_library(
    name = "condition",
    srcs = ["condition.py"],
    deps = [
        "//ydf/dataset:dataspec",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
    ],
)

py_library(
    name = "node",
    srcs = ["node.py"],
    deps = [
        ":condition",
        ":value",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
    ],
)

# Tests
# =====

py_test(
    name = "tree_test",
    srcs = ["tree_test.py"],
    data = [
        "//test_data",
    ],
    python_version = "PY3",
    deps = [
        ":condition",
        ":node",
        ":tree",
        ":value",
        # absl/testing:absltest dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
        "//ydf/utils:test_utils",
    ],
)

py_test(
    name = "value_test",
    srcs = ["value_test.py"],
    python_version = "PY3",
    deps = [
        ":value",
        # absl/testing:absltest dep,
        "@ydf_cc//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
        "//ydf/utils:test_utils",
        "@ydf_cc//yggdrasil_decision_forests/utils:distribution_py_proto",
    ],
)

py_test(
    name = "node_test",
    srcs = ["node_test.py"],
    python_version = "PY3",
    deps = [
        ":condition",
        ":node",
        ":value",
        # absl/testing:absltest dep,
    ],
)

py_test(
    name = "condition_test",
    srcs = ["condition_test.py"],
    python_version = "PY3",
    deps = [
        ":condition",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
        "//ydf/utils:test_utils",
    ],
)
