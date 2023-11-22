# Learner bindings for PYDF
# pytype binary, test, library
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")
load("//ydf/learner/wrapper:wrapper_generator.bzl", "py_wrap_yggdrasil_learners")
load("@ydf_cc//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")

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
    deps = [
        # absl:app dep,
        # absl/flags dep,
        # absl/logging dep,
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
        "@ydf_cc//yggdrasil_decision_forests/learner/hyperparameters_optimizer",
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
        "//ydf/model:model_cc",
        "//ydf/model:model_wrapper",
        "//ydf/utils:status_casters",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
        "@com_google_pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_inference",
        "@ydf_cc//yggdrasil_decision_forests/dataset:formats",
        "@ydf_cc//yggdrasil_decision_forests/dataset:vertical_dataset",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner:learner_library",
        "@ydf_cc//yggdrasil_decision_forests/learner/random_forest",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model",
        "@ydf_cc//yggdrasil_decision_forests/model:hyperparameter_cc_proto",
        "@ydf_cc//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest",
        "@ydf_cc//yggdrasil_decision_forests/utils:logging",
    ],
)

py_library(
    name = "generic_learner",
    srcs = ["generic_learner.py"],
    deps = [
        ":hyperparameters",
        ":tuner",
        # absl/logging dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:weight_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_py_proto",
        "//ydf/cc:ydf",
        "//ydf/dataset",
        "//ydf/dataset:dataspec",
        "//ydf/metric",
        "//ydf/model:generic_model",
        "//ydf/model:model_lib",
        "//ydf/utils:log",
        "@ydf_cc//yggdrasil_decision_forests/utils:fold_generator_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils/distribute/implementations/grpc:grpc_py_proto",
    ],
)

py_library(
    name = "hyperparameters",
    srcs = ["hyperparameters.py"],
    deps = ["@ydf_cc//yggdrasil_decision_forests/model:hyperparameter_py_proto"],
)

# This target is not referenced is included for documentation purposes only.
# See `:specialized_learners` for further information.
py_library(
    name = "specialized_learners_pre_generated",
    srcs = ["specialized_learners_pre_generated.py"],
    deps = [
        ":generic_learner",
        ":hyperparameters",
        ":tuner",
        "//ydf/dataset:dataspec",
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
        "@com_google_absl//absl/memory",
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
    deps = [
        ":generic_learner",
        ":specialized_learners",
        ":tuner",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # pandas dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "//ydf/dataset",
        "//ydf/dataset:dataspec",
        "//ydf/metric",
        "//ydf/model:generic_model",
        "//ydf/model/decision_forest_model",
        "//ydf/utils:log",
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
        # absl/testing:parameterized dep,
        # numpy dep,
        # pandas dep,
        # portpicker dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "//ydf/dataset",
        "//ydf/dataset:dataspec",
        "//ydf/metric",
        "//ydf/model:generic_model",
        "//ydf/model/decision_forest_model",
        "//ydf/utils:log",
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
        # absl/logging dep,
        # absl/testing:absltest dep,
        # numpy dep,
        # pandas dep,
        # tensorflow:tensorflow_no_contrib dep,
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "//ydf/dataset",
        "//ydf/utils:test_utils",
    ],
)

py_test(
    name = "tuner_test",
    srcs = ["tuner_test.py"],
    python_version = "PY3",
    deps = [
        ":tuner",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # absl/testing:parameterized dep,
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/hyperparameters_optimizer:hyperparameters_optimizer_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/hyperparameters_optimizer/optimizers:random_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:hyperparameter_py_proto",
        "//ydf/utils:test_utils",
    ],
)
