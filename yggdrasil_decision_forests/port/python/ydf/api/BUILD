load("@ydf_cc//yggdrasil_decision_forests/utils:compile.bzl", "py_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library_ydf(
    name = "init",
    srcs = ["__init__.py"],
)

# Libraries
# =========

py_library_ydf(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//ydf/util:log_book",
        "//ydf/util:tf_example",
        "//ydf/util:vertex_ai",
    ],
)

py_library_ydf(
    name = "internal",
    srcs = ["internal.py"],
    deps = ["//ydf/monitoring:benchmark_io_speed"],
)

# Note: This build rule intentionally does not depend on the JAX based deep models
# (//ydf/deep:{mlp,model_lib,tabular_transformer})
# so that `:api` does not link JAX. Those are added by the opt-in target
# //ydf:experimental_deep (google3 users:
# //third_party/py/ydf:experimental_deep). Without it, `ydf.experimental.*` raise a helpful error.
py_library_ydf(
    name = "experimental",
    srcs = ["experimental.py"],
    deps = [
        "//ydf/deep:jax_fallback",
    ],
)

py_library_ydf(
    name = "help",
    srcs = ["help.py"],
    deps = [
        "//ydf/util:help",
    ],
)
