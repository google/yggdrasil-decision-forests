# pytype test and library

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Users of the library
# TODO: Make public when fully available.
package_group(
    name = "users",
    packages = ["//ydf/..."],
)

# Internal to the library
package_group(
    name = "internal",
    packages = ["//ydf/..."],
)

# User facing API of pydf.
# TODO: Create alias for the library to be imported with "import ydf".
py_library(
    name = "api",
    srcs = ["__init__.py"],
    visibility = [":users"],
    deps = [
        "//ydf/dataset",
        "//ydf/learner:generic_learner",
        "//ydf/learner:specialized_learners",
        "//ydf/model:generic_model",
        "//ydf/model:model_lib",
    ],
)

# Tests
# =====

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    deps = [
        ":api",
        # absl/flags dep,
        # absl/logging dep,
        # absl/testing:absltest dep,
        # pandas dep,
    ],
)
