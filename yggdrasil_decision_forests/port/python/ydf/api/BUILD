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
