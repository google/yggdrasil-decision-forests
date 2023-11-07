load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Pybind
# ======

pybind_extension(
    name = "ydf",
    srcs = ["ydf.cc"],
    deps = [
        "//ydf/dataset:dataset_cc",
        "//ydf/learner:learner_cc",
        "//ydf/learner:worker_cc",
        "//ydf/metric:metric_cc",
        "//ydf/model:model_cc",
        "//ydf/utils:log_cc",
        "@com_google_pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

# Libraries
# =========

py_library(
    name = "ydf",
    data = ["ydf.so"],
)
