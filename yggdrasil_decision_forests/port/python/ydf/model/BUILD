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
        "@ydf_cc//yggdrasil_decision_forests/model/isolation_forest",
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
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:types",
        "@ydf_cc//yggdrasil_decision_forests/dataset:vertical_dataset",
        "@ydf_cc//yggdrasil_decision_forests/dataset:weight",
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
        "//ydf/model/isolation_forest_model:isolation_forest_wrapper",
        "//ydf/model/random_forest_model:random_forest_wrapper",
        "//ydf/utils:custom_casters",
        "//ydf/utils:status_casters",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model",
        "@ydf_cc//yggdrasil_decision_forests/model:model_library",
        "@ydf_cc//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/model/isolation_forest",
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
        "@ydf_cc//yggdrasil_decision_forests/utils:model_analysis",
        "@ydf_cc//yggdrasil_decision_forests/utils:model_analysis_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:status_macros",
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
        "//ydf/dataset/io:dataset_io",
        "//ydf/metric",
        "//ydf/utils:html",
        "//ydf/utils:log",
        "@ydf_cc//yggdrasil_decision_forests/utils:model_analysis_py_proto",
    ],
)

# Note: This build rule does not depends on TF. To use its functionalities, TF and TF-DF needs to be
# imported manually by the call i.e. //third_party/py/tensorflow_decision_forests.
py_library(
    name = "export_tf",
    srcs = ["export_tf.py"],
    deps = [
        ":generic_model",
        "//ydf/dataset:dataspec",
        "//ydf/dataset/io:dataset_io",
        "//ydf/utils:log",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
    ],
)

# Note: This build rule does not depends on JAX. To use its functionalities, JAX needs to be
# imported manually by the call i.e. //third_party/py/jax.
py_library(
    name = "export_jax",
    srcs = ["export_jax.py"],
    deps = [
        ":generic_model",
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "//ydf/dataset:dataspec",
        "//ydf/learner:custom_loss_py",
        "//ydf/model/decision_forest_model",
        "//ydf/model/gradient_boosted_trees_model",
        "//ydf/model/tree:all",
    ],
)

# Note: This build rule does not depends on SKLearn. To use its functionalities, SKLearn needs to be
# imported manually by the call i.e. //third_party/py/sklearn.
py_library(
    name = "export_sklearn",
    srcs = ["export_sklearn.py"],
    deps = [
        ":generic_model",
        # numpy dep,
        # sklearn dep,
        "//ydf/learner:generic_learner",
        "//ydf/learner:specialized_learners",
        "//ydf/model/gradient_boosted_trees_model",
        "//ydf/model/isolation_forest_model",
        "//ydf/model/random_forest_model",
        "//ydf/model/tree:all",
    ],
)

py_library(
    name = "model_lib",
    srcs = ["model_lib.py"],
    deps = [
        ":generic_model",
        # absl/logging dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "//ydf/cc:ydf",
        "//ydf/dataset:dataspec",
        "//ydf/model/gradient_boosted_trees_model",
        "//ydf/model/isolation_forest_model",
        "//ydf/model/random_forest_model",
        "//ydf/utils:log",
    ],
)

py_library(
    name = "analysis",
    srcs = ["analysis.py"],
    deps = [
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "//ydf/cc:ydf",
        "//ydf/dataset:dataspec",
        "@ydf_cc//yggdrasil_decision_forests/utils:model_analysis_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:partial_dependence_plot_py_proto",
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

py_library(
    name = "export_docker",
    srcs = ["export_docker.py"],
    deps = [
        ":generic_model",
        "//ydf/dataset:dataspec",
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
        ":analysis",
        ":generic_model",
        ":model_lib",
        ":model_metadata",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # pandas dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "//ydf/dataset",
        "//ydf/model/gradient_boosted_trees_model",
        "//ydf/model/random_forest_model",
        "//ydf/utils:test_utils",
    ],
)

py_test(
    name = "tf_model_test",
    srcs = ["tf_model_test.py"],
    data = [
        # tensorflow_decision_forests dep,  # buildcleaner: keep
        "//test_data",
        "@ydf_cc//yggdrasil_decision_forests/test_data",
    ],
    python_version = "PY3",
    shard_count = 5,
    # TODO: This test fails in OSS due to a package name clash on yggdrasil_decision_forests.
    tags = ["manual"],
    deps = [
        ":export_tf",
        ":generic_model",
        ":model_lib",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # pandas dep,
        # tensorflow dep,
        # tensorflow_decision_forests dep,
        "//ydf/dataset:dataspec",
        "//ydf/learner:generic_learner",
        "//ydf/learner:specialized_learners",
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
        "@com_google_absl//absl/log",
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

py_test(
    name = "jax_model_test",
    timeout = "long",
    srcs = ["jax_model_test.py"],
    data = [
        "//test_data",
        "@ydf_cc//yggdrasil_decision_forests/test_data",
    ],
    python_version = "PY3",
    deps = [
        ":export_jax",
        ":generic_model",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # jax/experimental/jax2tf dep,
        # matplotlib dep,
        # numpy dep,
        # optax dep,
        # tensorflow dep,
        # tensorflow_datasets dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "//ydf/dataset:dataspec",
        "//ydf/learner:generic_learner",
        "//ydf/learner:specialized_learners",
        "//ydf/model/gradient_boosted_trees_model",
        "//ydf/model/tree:all",
    ],
)

py_test(
    name = "sklearn_model_test",
    srcs = ["sklearn_model_test.py"],
    python_version = "PY3",
    deps = [
        ":export_sklearn",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # sklearn dep,
        "//ydf/model/decision_forest_model",
    ],
)

py_test(
    name = "export_docker_test",
    srcs = ["export_docker_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    deps = [
        ":export_docker",
        ":model_lib",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # fastapi dep,  # buildcleaner: keep
        # pydantic dep,  # buildcleaner: keep
        # requests dep,  # buildcleaner: keep
        "//ydf:api",
        "//ydf/utils:test_utils",
    ],
)
