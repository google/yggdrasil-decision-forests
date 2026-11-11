load("@rules_python//python:py_library.bzl", "py_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library(
    name = "init",
    srcs = ["__init__.py"],
)

# Libraries
# =========

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//ydf/util:log_book",
        "//ydf/util:tf_example",
        "//ydf/util:vertex_ai",
    ],
)

py_library(
    name = "internal",
    srcs = ["internal.py"],
    deps = ["//ydf/monitoring:benchmark_io_speed"],
)

# Note: This build rule intentionally does not depend on the JAX based deep models
# (//ydf/deep:{mlp,model_lib,tabular_transformer})
# so that `:api` does not link JAX. Those are added by the opt-in target
# //ydf:experimental_deep (google3 users:
# //third_party/py/ydf:experimental_deep). Without it, `ydf.experimental.*` raise a helpful error.
py_library(
    name = "experimental",
    srcs = ["experimental.py"],
    deps = [
        "//ydf/deep:jax_fallback",
    ],
)

py_library(
    name = "help",
    srcs = ["help.py"],
    deps = [
        "//ydf/util:help",
    ],
)
