load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Pybind
# ======

pybind_extension(
    name = "ydf",
    srcs = ["ydf.cc"],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": ["-Wl,-exported_symbol,_PyInit_ydf"],
        "//conditions:default": [],
    }),
    deps = [
        ":maybe_gcs_filesystem",
        "//ydf/dataset:dataset_cc",
        "//ydf/learner:learner_cc",
        "//ydf/learner:worker_cc",
        "//ydf/metric:evaluate_cc",
        "//ydf/metric:metric_cc",
        "//ydf/model:model_cc",
        "//ydf/utils:log_cc",
        "@com_google_pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

pybind_library(
    name = "maybe_gcs_filesystem",
    deps = [
        # GCS support is enabled by default in the external build.
        "@ydf_cc//yggdrasil_decision_forests/utils:filesystem_default_gcs",
    ],
)

# Libraries
# =========

py_library(
    name = "ydf",
    data = ["ydf.so"],
)
