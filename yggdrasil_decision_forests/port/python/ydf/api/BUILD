# pytype library only

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//ydf/util:tf_example",
    ],
)

py_library(
    name = "internal",
    srcs = ["internal.py"],
    deps = ["//ydf/monitoring:benchmark_io_speed"],
)

py_library(
    name = "experimental",
    srcs = ["experimental.py"],
    deps = [
        "//ydf/deep:jax_fallback",
        "//ydf/deep:mlp",
        "//ydf/deep:tabular_transformer",
    ],
)

py_library(
    name = "help",
    srcs = ["help.py"],
    deps = [
        "//ydf/util:help",
    ],
)
