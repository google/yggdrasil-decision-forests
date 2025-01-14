# Learner bindings for PYDF
# pytype binary, test, library
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")
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
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # ydf dep,
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
        "//ydf/model:model_cc",
        "//ydf/model:model_wrapper",
        "//ydf/utils:status_casters",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_inference",
        "@ydf_cc//yggdrasil_decision_forests/dataset:formats",
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
        ":hyperparameters",
        ":tuner",
        # absl/logging dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:weight_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "//ydf/cc:ydf",
        "//ydf/dataset",
        "//ydf/dataset:dataspec",
        "//ydf/metric",
        "//ydf/model:generic_model",
        "//ydf/model:model_lib",
        "//ydf/utils:concurrency",
        "//ydf/utils:log",
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
        "//ydf/cc:ydf",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:hyperparameter_py_proto",
    ],
)

py_library(
    name = "custom_loss_py",
    srcs = ["custom_loss.py"],
    deps = [
        # numpy dep,
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "//ydf/cc:ydf",
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
    srcs_version = "PY3",
    deps = [
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/hyperparameters_optimizer:hyperparameters_optimizer_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/hyperparameters_optimizer/optimizers:random_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:hyperparameter_py_proto",
    ],
)

py_library(
    name = "worker",
    srcs = ["worker.py"],
    srcs_version = "PY3",
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
        "@ydf_cc//yggdrasil_decision_forests/utils:uid",
        "@ydf_cc//yggdrasil_decision_forests/utils/distribute/implementations/grpc",
        "@ydf_cc//yggdrasil_decision_forests/utils/distribute/implementations/grpc:grpc_worker",
    ],
)

# Tests
# =====

py_test(
    name = "learner_test",
    size = "large",
    srcs = ["learner_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    shard_count = 10,
    tags = [
    ],
    deps = [
        ":generic_learner",
        ":specialized_learners",
        ":tuner",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # fastavro dep,
        # numpy dep,
        # pandas dep,
        # # polars dep,  # TODO: Re-enable.
        # sklearn dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "//ydf/dataset:dataspec",
        "//ydf/metric",
        "//ydf/model:generic_model",
        "//ydf/model:model_lib",
        "//ydf/model/decision_forest_model",
        "//ydf/model/tree:condition",
        "//ydf/model/tree:node",
        "//ydf/utils:log",
        "//ydf/utils:test_utils",
    ],
)

# Splits the test with tensorflow so that non-tensorflow tests can be run in debug mode.
# (TensorFlow does not compile in debug mode)
py_test(
    name = "custom_loss_test",
    timeout = "long",
    srcs = ["custom_loss_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    deps = [
        ":custom_loss_py",
        ":generic_learner",
        ":specialized_learners",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # jax dep,
        # numpy dep,
        # pandas dep,
        "//ydf/dataset:dataspec",
        "//ydf/model:generic_model",
        "//ydf/model/gradient_boosted_trees_model",
        "//ydf/utils:test_utils",
    ],
)

py_test(
    name = "distributed_learner_test",
    srcs = ["distributed_learner_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    shard_count = 10,
    deps = [
        ":generic_learner",
        ":specialized_learners",
        ":tuner",
        ":worker",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # pandas dep,
        # portpicker dep,
        "//ydf/utils:test_utils",
    ],
)

# Splits the test with tensorflow so that non-tensorflow tests can be run in debug mode.
# (TensorFlow does not compile in debug mode)
py_test(
    name = "learner_with_tf_test",
    srcs = ["learner_with_tf_test.py"],
    python_version = "PY3",
    deps = [
        ":generic_learner",
        ":specialized_learners",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # pandas dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "hyperparameters_test",
    srcs = ["hyperparameters_test.py"],
    python_version = "PY3",
    deps = [
        ":hyperparameters",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
    ],
)

py_test(
    name = "learner_with_xarray_test",
    srcs = ["learner_with_xarray_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    deps = [
        ":specialized_learners",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # xarray dep,
        "//ydf/utils:test_utils",
    ],
)

py_test(
    name = "tuner_test",
    srcs = ["tuner_test.py"],
    python_version = "PY3",
    tags = [
    ],
    deps = [
        ":tuner",
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/hyperparameters_optimizer:hyperparameters_optimizer_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/hyperparameters_optimizer/optimizers:random_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:hyperparameter_py_proto",
        "//ydf/utils:test_utils",
    ],
)

py_test(
    name = "feature_selector_test",
    srcs = ["feature_selector_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":feature_selector",
        ":specialized_learners",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        "//ydf/utils:test_utils",
    ],
)
