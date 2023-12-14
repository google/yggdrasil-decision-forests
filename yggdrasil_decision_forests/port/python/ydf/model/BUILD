# Model bindings for PYDF
# pytype binary, test, library
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")
load("@ydf_cc//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

# Add new models here.
cc_library_ydf(
    name = "pydf_models",
    deps = [
        "@ydf_cc//yggdrasil_decision_forests/model/decision_tree:decision_forest_interface",
        "@ydf_cc//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest",
    ],
)

pybind_library(
    name = "model_wrapper",
    srcs = ["model_wrapper.cc"],
    hdrs = ["model_wrapper.h"],
    deps = [
        ":pydf_models",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:types",
        "@ydf_cc//yggdrasil_decision_forests/dataset:vertical_dataset",
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:describe",
        "@ydf_cc//yggdrasil_decision_forests/model:model_library",
        "@ydf_cc//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest",
        "@ydf_cc//yggdrasil_decision_forests/serving:example_set",
        "@ydf_cc//yggdrasil_decision_forests/serving:fast_engine",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
        "@ydf_cc//yggdrasil_decision_forests/utils:model_analysis",
        "@ydf_cc//yggdrasil_decision_forests/utils:model_analysis_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:random",
        "@ydf_cc//yggdrasil_decision_forests/utils:status_macros",
        "@ydf_cc//yggdrasil_decision_forests/utils:synchronization_primitives",
        "@ydf_cc//yggdrasil_decision_forests/utils:uid",
        "@ydf_cc//yggdrasil_decision_forests/utils/benchmark:inference",
    ],
)

pybind_library(
    name = "model_cc",
    srcs = ["model.cc"],
    hdrs = ["model.h"],
    deps = [
        ":model_wrapper",
        "//ydf/model/decision_forest_model:decision_forest_wrapper",
        "//ydf/model/gradient_boosted_trees_model:gradient_boosted_trees_wrapper",
        "//ydf/model/random_forest_model:random_forest_wrapper",
        "//ydf/utils:custom_casters",
        "//ydf/utils:status_casters",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model",
        "@ydf_cc//yggdrasil_decision_forests/model:model_library",
        "@ydf_cc//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
        "@ydf_cc//yggdrasil_decision_forests/utils:model_analysis",
        "@ydf_cc//yggdrasil_decision_forests/utils:model_analysis_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils/benchmark:inference",
    ],
)

py_library(
    name = "generic_model",
    srcs = ["generic_model.py"],
    deps = [
        ":analysis",
        ":model_metadata",
        ":optimizer_logs",
        ":template_cpp_export",
        # absl/logging dep,
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "//ydf/cc:ydf",
        "//ydf/dataset",
        "//ydf/dataset:dataspec",
        "//ydf/metric",
        "//ydf/utils:html",
        "//ydf/utils:log",
        "@ydf_cc//yggdrasil_decision_forests/utils:model_analysis_py_proto",
    ],
)

py_library(
    name = "model_lib",
    srcs = ["model_lib.py"],
    deps = [
        ":generic_model",
        # absl/logging dep,
        "//ydf/cc:ydf",
        "//ydf/model/gradient_boosted_trees_model",
        "//ydf/model/random_forest_model",
    ],
)

py_library(
    name = "analysis",
    srcs = ["analysis.py"],
    deps = [
        "//ydf/cc:ydf",
        "@ydf_cc//yggdrasil_decision_forests/utils:model_analysis_py_proto",
    ],
)

py_library(
    name = "model_metadata",
    srcs = ["model_metadata.py"],
    deps = [
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
    ],
)

py_library(
    name = "template_cpp_export",
    srcs = ["template_cpp_export.py"],
    deps = [
        "//ydf:version",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
    ],
)

py_library(
    name = "optimizer_logs",
    srcs = ["optimizer_logs.py"],
    deps = [
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:hyperparameter_py_proto",
    ],
)

# Tests
# =====

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    data = [
        "//test_data",
        "@ydf_cc//yggdrasil_decision_forests/test_data",
    ],
    python_version = "PY3",
    deps = [
        ":generic_model",
        ":model_lib",
        ":model_metadata",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # pandas dep,
        "//ydf/dataset",
        "//ydf/model/gradient_boosted_trees_model",
        "//ydf/model/random_forest_model",
        "//ydf/utils:log",
        "//ydf/utils:test_utils",
    ],
)

py_test(
    name = "tf_model_test",
    srcs = ["tf_model_test.py"],
    data = [
        "//test_data",
        "@ydf_cc//yggdrasil_decision_forests/test_data",
    ],
    python_version = "PY3",
    # TODO: This test fails in OSS due to a package name clash on yggdrasil_decision_forests.
    tags = ["manual"],
    deps = [
        ":model_lib",
        # absl/testing:absltest dep,
        # numpy dep,
        # pandas dep,
        # tensorflow dep,
        # tensorflow_decision_forests dep,
        "//ydf/utils:test_utils",
    ],
)

cc_test(
    name = "export_cc_run_test",
    srcs = [
        "export_cc_generated_lib.h",
        "export_cc_run_test.cc",
    ],
    data = [
        ":export_cc_generate_lib",
        "@ydf_cc//yggdrasil_decision_forests/test_data",
    ],
    tags = ["manual"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@ydf_cc//yggdrasil_decision_forests/api:serving",
        "@ydf_cc//yggdrasil_decision_forests/utils:filesystem",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
        "@ydf_cc//yggdrasil_decision_forests/utils:test",
    ],
)

genrule(
    name = "export_cc_generate_lib",
    srcs = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    outs = ["export_cc_generated_lib.h"],
    cmd = "$(location export_cc_generator) --input_model=third_party/yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt --output_code=$@",
    tags = ["manual"],
    tools = [":export_cc_generator"],
)

py_binary(
    name = "export_cc_generator",
    srcs = ["export_cc_generator.py"],
    python_version = "PY3",
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
        "//ydf:api",
    ],
)

py_test(
    name = "optimizer_logs_test",
    srcs = ["optimizer_logs_test.py"],
    python_version = "PY3",
    deps = [
        ":optimizer_logs",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:hyperparameter_py_proto",
    ],
)
