load("//yggdrasil_decision_forests/utils:compile.bzl", "cc_binary_ydf", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Collections
# ===========

cc_library_ydf(
    name = "all_file_systems",
    deps = [],
    alwayslink = 1,
)

# Binaries
# ========

cc_binary_ydf(
    name = "benchmark_inference",
    srcs = ["benchmark_inference.cc"],
    deps = [
        ":all_file_systems",
        "@com_google_absl//absl/flags:flag",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:all_models",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:logging",
    ],
)

cc_binary_ydf(
    name = "infer_dataspec",
    srcs = ["infer_dataspec.cc"],
    deps = [
        ":all_file_systems",
        "@com_google_absl//absl/flags:flag",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
    ],
)

cc_binary_ydf(
    name = "show_dataspec",
    srcs = ["show_dataspec.cc"],
    deps = [
        ":all_file_systems",
        "@com_google_absl//absl/flags:flag",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
    ],
)

cc_binary_ydf(
    name = "train",
    srcs = ["train.cc"],
    deps = [
        ":all_file_systems",
        "@com_google_absl//absl/flags:flag",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/learner:all_learners",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils/distribute/implementations/grpc",
    ],
)

cc_binary_ydf(
    name = "show_model",
    srcs = ["show_model.cc"],
    deps = [
        ":all_file_systems",
        "@com_google_absl//absl/flags:flag",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:all_models",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:logging",
    ],
)

cc_binary_ydf(
    name = "convert_dataset",
    srcs = ["convert_dataset.cc"],
    deps = [
        ":all_file_systems",
        "@com_google_absl//absl/flags:flag",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_reader",
        "//yggdrasil_decision_forests/dataset:example_writer",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
)

cc_binary_ydf(
    name = "predict",
    srcs = ["predict.cc"],
    deps = [
        ":all_file_systems",
        "@com_google_absl//absl/flags:flag",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:all_models",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/model:prediction_cc_proto",
        "//yggdrasil_decision_forests/serving/decision_forest:register_engines",
        "//yggdrasil_decision_forests/utils:evaluation",
        "//yggdrasil_decision_forests/utils:logging",
    ],
)

cc_binary_ydf(
    name = "evaluate",
    srcs = ["evaluate.cc"],
    deps = [
        ":all_file_systems",
        "@com_google_absl//absl/flags:flag",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/dataset:weight",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/metric:metric_cc_proto",
        "//yggdrasil_decision_forests/metric:report",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:all_models",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/model:prediction_cc_proto",
        "//yggdrasil_decision_forests/serving/decision_forest:register_engines",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:protobuf",
        "//yggdrasil_decision_forests/utils:random",
    ],
)

# Tests
# =====

sh_test(
    name = "cli_test",
    size = "large",
    srcs = ["cli_test.sh"],
    data = [
        ":benchmark_inference",
        ":convert_dataset",
        ":evaluate",
        ":infer_dataspec",
        ":predict",
        ":show_dataspec",
        ":show_model",
        ":train",
        "//yggdrasil_decision_forests/test_data",
    ],
)
