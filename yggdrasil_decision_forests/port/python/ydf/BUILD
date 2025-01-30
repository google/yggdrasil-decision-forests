# pytype test and library

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# User facing API of pydf.
# TODO: Create alias for the library to be imported with "import ydf".
py_library(
    name = "api",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":version",
        ":ydf_protos",
        "//ydf/api:experimental",
        "//ydf/api:internal",
        "//ydf/api:util",
        "//ydf/dataset",
        "//ydf/dataset:dataspec",
        "//ydf/learner:custom_loss_py",
        "//ydf/learner:feature_selector",
        "//ydf/learner:generic_learner",
        "//ydf/learner:specialized_learners",
        "//ydf/learner:tuner",
        "//ydf/learner:worker",
        "//ydf/metric:evaluate",
        "//ydf/model:export_docker",  # buildcleaner: keep
        "//ydf/model:export_jax",  # buildcleaner: keep
        "//ydf/model:export_sklearn",  # buildcleaner: keep
        "//ydf/model:export_tf",  # buildcleaner: keep
        "//ydf/model:feature_selector_logs",
        "//ydf/model:generic_model",
        "//ydf/model:model_lib",
        "//ydf/model:model_metadata",
        "//ydf/model:optimizer_logs",
        "//ydf/model/gradient_boosted_trees_model",
        "//ydf/model/isolation_forest_model",
        "//ydf/model/random_forest_model",
        "//ydf/model/tree:all",
        "//ydf/utils:log",
    ],
)

# All the YDF protos, including the ones possibly not used by PYDF.
py_library(
    name = "ydf_protos",
    deps = [
        "@ydf_cc//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:example_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:formats_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:synthetic_dataset_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/dataset:weight_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner:abstract_learner_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/cart:cart_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/decision_tree:decision_tree_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/distributed_decision_tree:training_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/distributed_decision_tree/load_balancer:load_balancer_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees:dgbt_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees:worker_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/generic_worker:generic_worker_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/gradient_boosted_trees/early_stopping:early_stopping_snapshot_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/hyperparameters_optimizer:hyperparameters_optimizer_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/hyperparameters_optimizer/optimizers:random_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/isolation_forest:isolation_forest_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/multitasker:multitasker_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/learner/random_forest:random_forest_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/metric:metric_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:abstract_model_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:hyperparameter_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model:prediction_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model/decision_tree:decision_tree_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model/gradient_boosted_trees:gradient_boosted_trees_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model/isolation_forest:isolation_forest_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model/multitasker:multitasker_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/model/random_forest:random_forest_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/serving:serving_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:bitmap_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:distribution_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:fold_generator_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:model_analysis_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils:partial_dependence_plot_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils/distribute:distribute_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils/distribute/implementations/grpc:grpc_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils/distribute/implementations/multi_thread:multi_thread_py_proto",
        "@ydf_cc//yggdrasil_decision_forests/utils/distribute_cli:distribute_cli_py_proto",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
    deps = [
    ],
)

# Tests
# =====

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    data = ["@ydf_cc//yggdrasil_decision_forests/test_data"],
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":api",
        # absl/logging dep,
        # absl/testing:absltest dep,
        # fastavro dep,
        # jax dep,  # buildcleaner: keep
        # numpy dep,
        # pandas dep,
        # sklearn dep,  # buildcleaner: keep
        # tensorflow dep,  # buildcleaner: keep
        # tensorflow_decision_forests dep,  # buildcleaner: keep
        "//ydf/utils:test_utils",
    ],
)
