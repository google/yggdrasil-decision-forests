# Dataset bindings for PYDF
# pytype test and library
load("@ydf_cc//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "jax_fallback",
    srcs = ["jax_fallback.py"],
    deps = [],
)

py_library(
    name = "safetensors",
    srcs = ["safetensors.py"],
    deps = [
        # jaxtyping dep,
        # numpy dep,
    ],
)

py_library(
    name = "model_lib",
    srcs = ["model_lib.py"],
    deps = [
        ":deep_model_py_proto",
        ":generic_jax",
        ":mlp",
        ":preprocessor",
        ":safetensors",
        ":tabular_transformer",
        # safetensors dep,
        # safetensors:flax dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "//ydf/model:generic_model",
        "//ydf/utils:filesystem",
    ],
)

py_library(
    name = "generic_jax",
    srcs = ["generic_jax.py"],
    deps = [
        ":analysis",
        ":dataset",
        ":deep_model_py_proto",
        ":hyperparameter",
        ":metric",
        ":preprocessor",
        ":safetensors",
        # jax dep,
        # jaxtyping dep,
        # numpy dep,
        # optax dep,
        # safetensors dep,
        # safetensors:flax dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "//ydf/cc:ydf",
        "//ydf/dataset",
        "//ydf/dataset:dataspec",
        "//ydf/dataset/io:dataset_io",
        "//ydf/dataset/io:generator",
        "//ydf/learner:abstract_feature_selector",
        "//ydf/learner:generic_learner",
        "//ydf/learner:hyperparameters",
        "//ydf/learner:tuner",
        "//ydf/metric",
        "//ydf/metric:evaluate",
        "//ydf/model:analysis",
        "//ydf/model:feature_selector_logs",
        "//ydf/model:generic_model",
        "//ydf/model:model_metadata",
        "//ydf/model:optimizer_logs",
        "//ydf/utils:concurrency",
        "//ydf/utils:filesystem",
        "//ydf/utils:html",
        "//ydf/utils:log",
        "@ydf_cc//yggdrasil_decision_forests/utils:model_analysis_py_proto",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        # jax dep,
        "//ydf/dataset/io:generator",
    ],
)

py_library(
    name = "metric",
    srcs = ["metric.py"],
    deps = [
        # jax dep,
        # optax dep,
    ],
)

py_library(
    name = "preprocessor",
    srcs = ["preprocessor.py"],
    deps = [
        ":dataset",
        ":deep_model_py_proto",
        ":layer",
        # jax dep,
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "//ydf/dataset:dataspec",
    ],
)

py_library(
    name = "layer",
    srcs = ["layer.py"],
    deps = [
        # flax:core dep,
        # jax dep,
    ],
)

py_library(
    name = "hyperparameter",
    srcs = ["hyperparameter.py"],
    deps = [],
)

py_library(
    name = "mlp",
    srcs = ["mlp.py"],
    deps = [
        ":deep_model_py_proto",
        ":generic_jax",
        ":hyperparameter",
        ":layer",
        # flax:core dep,
        # jax dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "//ydf/dataset:dataspec",
        "//ydf/learner:abstract_feature_selector",
        "//ydf/learner:generic_learner",
        "//ydf/learner:tuner",
        "//ydf/model:generic_model",
        "//ydf/utils:func_helpers",
    ],
)

py_library(
    name = "tabular_transformer",
    srcs = ["tabular_transformer.py"],
    deps = [
        ":deep_model_py_proto",
        ":generic_jax",
        ":hyperparameter",
        ":layer",
        # flax:core dep,
        # jax dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "//ydf/dataset:dataspec",
        "//ydf/learner:abstract_feature_selector",
        "//ydf/learner:generic_learner",
        "//ydf/learner:tuner",
        "//ydf/model:generic_model",
        "//ydf/utils:func_helpers",
        "//ydf/utils:log",
    ],
)

py_library(
    name = "analysis",
    srcs = ["analysis.py"],
    deps = [
        # absl/logging dep,
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:example_py_proto",
        "//ydf/dataset",
        "//ydf/dataset:dataspec",
        "//ydf/dataset/io:dataset_io",
        "//ydf/dataset/io:generator",
        "//ydf/model:analysis",
        "//ydf/model:generic_model",
        "//ydf/utils:log",
        "@ydf_cc//yggdrasil_decision_forests/utils:distribution_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:model_analysis_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:partial_dependence_plot_py_proto",
    ],
)

all_proto_library(
    name = "deep_model_proto",
    srcs = ["deep_model.proto"],
)

# Tests
# =====

py_test(
    name = "mlp_test",
    srcs = ["mlp_test.py"],
    data = [
        "//test_data",
        "@ydf_cc//yggdrasil_decision_forests/test_data",
    ],
    shard_count = 4,
    deps = [
        ":mlp",
        ":model_lib",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # pandas dep,
        "//ydf/dataset:dataspec",
        "//ydf/dataset/io:dataset_io",
        "//ydf/learner:specialized_learners",
        "//ydf/model:generic_model",
        "//ydf/utils:test_utils",
    ],
)

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    data = [
        "//test_data",
        "@ydf_cc//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":dataset",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
    ],
)

py_test(
    name = "preprocessor_test",
    srcs = ["preprocessor_test.py"],
    deps = [
        ":dataset",
        ":layer",
        ":preprocessor",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "//ydf/utils:test_utils",
    ],
)

py_test(
    name = "hyperparameter_test",
    srcs = ["hyperparameter_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    deps = [
        ":hyperparameter",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
    ],
)

py_test(
    name = "layer_test",
    srcs = ["layer_test.py"],
    deps = [
        ":layer",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
    ],
)

py_test(
    name = "safetensors_test",
    srcs = ["safetensors_test.py"],
    deps = [
        ":safetensors",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
    ],
)

py_test(
    name = "metric_test",
    srcs = ["metric_test.py"],
    deps = [
        ":metric",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
    ],
)

py_test(
    name = "analysis_test",
    srcs = ["analysis_test.py"],
    deps = [
        ":analysis",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:example_py_proto",
        "//ydf/dataset:dataspec",
        "//ydf/model:generic_model",
        "//ydf/utils:test_utils",
        "@ydf_cc//yggdrasil_decision_forests/utils:distribution_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:model_analysis_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:partial_dependence_plot_py_proto",
    ],
)

py_test(
    name = "tabular_transformer_test",
    size = "large",
    srcs = ["tabular_transformer_test.py"],
    data = [
        "//test_data",
        "@ydf_cc//yggdrasil_decision_forests/test_data",
    ],
    shard_count = 4,
    deps = [
        ":layer",
        ":model_lib",
        ":tabular_transformer",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        # pandas dep,
        "//ydf/model:generic_model",
        "//ydf/utils:test_utils",
    ],
)
