load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

# Learner bindings for PYDF
load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("@ydf_cc//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")
load("//ydf/learner/wrapper:wrapper_generator.bzl", "py_wrap_yggdrasil_learners")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Binary
# =========

py_binary(
    name = "worker_main",
    srcs = ["worker_main.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//ydf:api",
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
    ],
)

# Libraries
# =========

# Available learning algorithms.
cc_library_ydf(
    name = "canonical_learners",
    deps = [
        "@ydf_cc//yggdrasil_decision_forests/learner/cart",
        "@ydf_cc//yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees:dgbt",
        "@ydf_cc//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/learner/isolation_forest",
        "@ydf_cc//yggdrasil_decision_forests/learner/random_forest",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "canonical_workers",
    deps = [
        "@ydf_cc//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_worker",
        "@ydf_cc//yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees:worker",
        "@ydf_cc//yggdrasil_decision_forests/learner/generic_worker",
    ],
    alwayslink = 1,
)

py_wrap_yggdrasil_learners(
    name = "specialized_learners",
    learner_deps = [":canonical_learners"],
)

pybind_library(
    name = "learner_cc",
    srcs = ["learner.cc"],
    hdrs = ["learner.h"],
    deps = [
        ":canonical_learners",
        ":custom_loss",
        ":custom_metric",
        "//ydf/model:model_cc",
        "//ydf/model:model_wrapper",
        "//ydf/utils:status_casters",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_inference",
        "@ydf_cc//yggdrasil_decision_forests/dataset:formats",
        "@ydf_cc//yggdrasil_decision_forests/dataset:types",
        "@ydf_cc//yggdrasil_decision_forests/dataset:vertical_dataset",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner:learner_library",
        "@ydf_cc//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:loss_imp_custom_regression",
        "@ydf_cc//yggdrasil_decision_forests/learner/hyperparameters_optimizer",
        "@ydf_cc//yggdrasil_decision_forests/learner/random_forest",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model",
        "@ydf_cc//yggdrasil_decision_forests/model:hyperparameter_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
        "@ydf_cc//yggdrasil_decision_forests/utils:status_macros",
    ],
)

pybind_library(
    name = "custom_loss",
    srcs = ["custom_loss.cc"],
    hdrs = ["custom_loss.h"],
    deps = [
        "//ydf/utils:numpy_data",
        "//ydf/utils:pybind",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner",
        "@ydf_cc//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:loss_imp_custom_binary_classification",
        "@ydf_cc//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:loss_imp_custom_multi_classification",
        "@ydf_cc//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:loss_imp_custom_regression",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
        "@ydf_cc//yggdrasil_decision_forests/utils:status_macros",
    ],
)

py_library(
    name = "generic_learner",
    srcs = ["generic_learner.py"],
    deps = [
        ":abstract_feature_selector",
        ":custom_loss_py",
        ":custom_metric_py",
        ":hyperparameters",
        ":tuner",
        "//ydf/cc:ydf",
        "//ydf/dataset",
        "//ydf/dataset:dataspec",
        "//ydf/metric",
        "//ydf/model:generic_model",
        "//ydf/model:model_lib",
        "//ydf/utils:concurrency",
        "//ydf/utils:log",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:weight_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:fold_generator_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils/distribute/implementations/grpc:grpc_py_proto",
    ],
)

py_library(
    name = "abstract_feature_selector",
    srcs = ["abstract_feature_selector.py"],
    deps = [
        "//ydf/dataset",
        "//ydf/metric",
        "//ydf/model:generic_model",
    ],
)

py_library(
    name = "feature_selector",
    srcs = ["feature_selector.py"],
    deps = [
        ":abstract_feature_selector",
        ":generic_learner",
        "//ydf/dataset",
        "//ydf/dataset:dataspec",
        "//ydf/model:feature_selector_logs",
        "//ydf/model:generic_model",
        "//ydf/utils:log",
    ],
)

py_library(
    name = "hyperparameters",
    srcs = ["hyperparameters.py"],
    deps = [
        ":custom_loss_py",
        ":custom_metric_py",
        "//ydf/cc:ydf",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:hyperparameter_py_proto",
    ],
)

py_library(
    name = "custom_loss_py",
    srcs = ["custom_loss.py"],
    deps = [
        "//ydf/cc:ydf",
        "@pypi_ydf//numpy",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
    ],
)

py_library(
    name = "custom_metric_py",
    srcs = ["custom_metric.py"],
    deps = [
        "//ydf/cc:ydf",
        "@pypi_ydf//numpy",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
    ],
)

# This target is not referenced is included for documentation purposes only.
# See `:specialized_learners` for further information.
py_library(
    name = "specialized_learners_pre_generated",
    srcs = ["specialized_learners_pre_generated.py"],
    deps = [
        ":abstract_feature_selector",
        ":custom_loss_py",
        ":custom_metric_py",
        ":generic_learner",
        ":hyperparameters",
        ":tuner",
        "//ydf/dataset",
        "//ydf/dataset:dataspec",
        "//ydf/model/gradient_boosted_trees_model",
        "//ydf/model/isolation_forest_model",
        "//ydf/model/random_forest_model",
        "//ydf/utils:func_helpers",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
    ],
)

py_library(
    name = "tuner",
    srcs = ["tuner.py"],
    deps = [
        "//ydf/model:generic_model",
        "//ydf/utils:log",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/hyperparameters_optimizer:hyperparameters_optimizer_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/hyperparameters_optimizer/optimizers:random_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:hyperparameter_py_proto",
    ],
)

py_library(
    name = "worker",
    srcs = ["worker.py"],
    deps = [
        "//ydf/cc:ydf",
    ],
)

pybind_library(
    name = "worker_cc",
    srcs = ["worker.cc"],
    hdrs = ["worker.h"],
    deps = [
        ":canonical_workers",
        "//ydf/utils:status_casters",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@ydf_cc//yggdrasil_decision_forests/utils:concurrency",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
        "@ydf_cc//yggdrasil_decision_forests/utils:synchronization_primitives",
        "@ydf_cc//yggdrasil_decision_forests/utils:uid",
        "@ydf_cc//yggdrasil_decision_forests/utils/distribute/implementations/grpc",
        "@ydf_cc//yggdrasil_decision_forests/utils/distribute/implementations/grpc:grpc_worker",
    ],
)

pybind_library(
    name = "custom_metric",
    srcs = ["custom_metric.cc"],
    hdrs = ["custom_metric.h"],
    deps = [
        "//ydf/utils:numpy_data",
        "//ydf/utils:pybind",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner",
        "@ydf_cc//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/learner/gradient_boosted_trees/loss:loss_interface",
        "@ydf_cc//yggdrasil_decision_forests/utils:status_macros",
    ],
)

# Tests
# =====

py_library(
    name = "learner_test_utils",
    testonly = 1,
    srcs = ["learner_test_utils.py"],
    deps = [
        ":generic_learner",
        "//ydf/dataset:dataspec",
        "//ydf/metric",
        "//ydf/model:generic_model",
        "//ydf/model:model_lib",
        "//ydf/utils:test_utils",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//numpy",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_py_proto",
    ],
)

py_test(
    name = "generic_learner_test",
    srcs = ["generic_learner_test.py"],
    deps = [
        ":generic_learner",
        ":specialized_learners",
        "@abseil-py//absl/logging",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//fastavro",
        "@pypi_ydf//numpy",
        # "@pypi_ydf//polars",  # TODO: Re-enable.
        "@pypi_ydf//pandas",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "//ydf/dataset:dataspec",
        "//ydf/model:generic_model",
        "//ydf/utils:log",
    ],
)

py_test(
    name = "random_forest_learner_test",
    size = "large",
    srcs = ["random_forest_learner_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    shard_count = 10,
    deps = [
        ":generic_learner",
        ":learner_test_utils",
        ":specialized_learners",
        "//ydf/dataset:dataspec",
        "//ydf/model/decision_forest_model",
        "//ydf/model/tree:condition",
        "//ydf/model/tree:node",
        "//ydf/utils:test_utils",
        "@abseil-py//absl/logging",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//numpy",
        "@pypi_ydf//pandas",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_py_proto",
    ],
)

py_test(
    name = "gradient_boosted_trees_learner_test",
    size = "large",
    srcs = ["gradient_boosted_trees_learner_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    shard_count = 10,
    tags = [
    ],
    deps = [
        ":generic_learner",
        ":learner_test_utils",
        ":specialized_learners",
        ":tuner",
        "//ydf/dataset:dataspec",
        "//ydf/model/decision_forest_model",
        "//ydf/utils:test_utils",
        "@abseil-py//absl/logging",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//numpy",
        "@pypi_ydf//pandas",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
    ],
)

py_test(
    name = "cart_learner_test",
    srcs = ["cart_learner_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    deps = [
        ":generic_learner",
        ":learner_test_utils",
        ":specialized_learners",
        ":tuner",
        "//ydf/dataset:dataspec",
        "//ydf/utils:test_utils",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//numpy",
        "@pypi_ydf//pandas",
    ],
)

py_test(
    name = "isolation_forest_learner_test",
    size = "large",
    srcs = ["isolation_forest_learner_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    shard_count = 10,
    deps = [
        ":generic_learner",
        ":learner_test_utils",
        ":specialized_learners",
        "//ydf/model/tree:condition",
        "//ydf/model/tree:node",
        "@abseil-py//absl/logging",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//numpy",
        "@pypi_ydf//scikit_learn",
    ],
)

# Splits the test with tensorflow so that non-tensorflow tests can be run in debug mode.
# (TensorFlow does not compile in debug mode)
py_test(
    name = "custom_loss_test",
    timeout = "long",
    srcs = ["custom_loss_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    deps = [
        ":custom_loss_py",
        ":generic_learner",
        ":specialized_learners",
        ":tuner",
        "//ydf/dataset:dataspec",
        "//ydf/model:generic_model",
        "//ydf/model/gradient_boosted_trees_model",
        "//ydf/utils:test_utils",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//jax",
        "@pypi_ydf//numpy",
        "@pypi_ydf//pandas",
    ],
)

py_test(
    name = "distributed_learner_test",
    srcs = ["distributed_learner_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    shard_count = 10,
    deps = [
        ":generic_learner",
        ":specialized_learners",
        ":tuner",
        ":worker",
        "//ydf/utils:test_utils",
        "@abseil-py//absl/logging",
        "@abseil-py//absl/testing:absltest",
        "@pypi_ydf//pandas",
        "@pypi_ydf//portpicker",
    ],
)

# Splits the test with tensorflow so that non-tensorflow tests can be run in debug mode.
# (TensorFlow does not compile in debug mode)
py_test(
    name = "learner_with_tf_test",
    srcs = ["learner_with_tf_test.py"],
    tags = ["tf_dep"],
    deps = [
        ":generic_learner",
        ":specialized_learners",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//pandas",
        "@pypi_ydf//tensorflow:pkg",
    ],
)

py_test(
    name = "hyperparameters_test",
    srcs = ["hyperparameters_test.py"],
    deps = [
        ":hyperparameters",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
    ],
)

py_test(
    name = "learner_with_xarray_test",
    srcs = ["learner_with_xarray_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    deps = [
        ":specialized_learners",
        "//ydf/utils:test_utils",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//xarray",
    ],
)

py_test(
    name = "tuner_test",
    srcs = ["tuner_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    tags = [
    ],
    deps = [
        ":specialized_learners",
        ":tuner",
        "//ydf/dataset:dataspec",
        "//ydf/model:generic_model",
        "//ydf/utils:test_utils",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/hyperparameters_optimizer:hyperparameters_optimizer_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/hyperparameters_optimizer/optimizers:random_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:hyperparameter_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:fold_generator_py_proto",
    ],
)

py_test(
    name = "feature_selector_test",
    srcs = ["feature_selector_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    shard_count = 5,
    deps = [
        ":feature_selector",
        ":specialized_learners",
        "//ydf/utils:test_utils",
        "@abseil-py//absl/logging",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//numpy",
    ],
)

py_test(
    name = "custom_metric_test",
    srcs = ["custom_metric_test.py"],
    deps = [
        ":custom_metric_py",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi_ydf//numpy",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
    ],
)
