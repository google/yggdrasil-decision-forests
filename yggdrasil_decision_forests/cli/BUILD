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
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:all_models",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils/benchmark:inference",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_binary_ydf(
    name = "infer_dataspec",
    srcs = ["infer_dataspec.cc"],
    deps = [
        ":all_file_systems",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_binary_ydf(
    name = "show_dataspec",
    srcs = ["show_dataspec.cc"],
    deps = [
        ":all_file_systems",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_binary_ydf(
    name = "train",
    srcs = ["train.cc"],
    deps = [
        ":all_file_systems",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/learner:all_learners",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils/distribute/implementations/grpc",
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_binary_ydf(
    name = "show_model",
    srcs = ["show_model.cc"],
    deps = [
        ":all_file_systems",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:all_models",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:logging",
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_binary_ydf(
    name = "convert_dataset",
    srcs = ["convert_dataset.cc"],
    deps = [
        ":all_file_systems",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_reader",
        "//yggdrasil_decision_forests/dataset:example_writer",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_binary_ydf(
    name = "predict",
    srcs = ["predict.cc"],
    deps = [
        ":all_file_systems",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec",
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
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_binary_ydf(
    name = "evaluate",
    srcs = ["evaluate.cc"],
    deps = [
        ":all_file_systems",
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
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_binary_ydf(
    name = "edit_model",
    srcs = ["edit_model.cc"],
    deps = [
        ":all_file_systems",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:all_models",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary_ydf(
    name = "compute_variable_importances",
    srcs = ["compute_variable_importances.cc"],
    deps = [
        ":all_file_systems",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/model:all_models",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:feature_importance",
        "//yggdrasil_decision_forests/utils:logging",
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_binary_ydf(
    name = "analyze_model_and_dataset",
    srcs = ["analyze_model_and_dataset.cc"],
    deps = [
        ":all_file_systems",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/model:all_models",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:model_analysis",
        "//yggdrasil_decision_forests/utils:protobuf",
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_binary_ydf(
    name = "compile_model",
    srcs = ["compile_model.cc"],
    deps = [
        ":all_file_systems",
        "//yggdrasil_decision_forests/serving/decision_forest:model_compiler",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
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
        ":compute_variable_importances",
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

genrule(
    name = "compile_model_for_test",
    testonly = 1,
    srcs = [
        ":compile_model.cc",
        "//yggdrasil_decision_forests/test_data",
    ],
    outs = ["generated_model.h"],
    cmd = """
    TESTDATA_BASEDIR=$$(dirname $$(echo $(locations //yggdrasil_decision_forests/test_data) | cut -d ' ' -f1))/model/synthetic_ranking_gbdt_numerical/
    $(location :compile_model) --model $$TESTDATA_BASEDIR --namespace test_model > $@
    """,
    tools = [":compile_model"],
)

cc_library_ydf(
    name = "compiled_model_for_test",
    testonly = 1,
    hdrs = ["generated_model.h"],
    deps = [
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/serving/decision_forest:decision_forest_serving",
        "//yggdrasil_decision_forests/utils:protobuf",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "compile_model_test",
    srcs = ["compile_model_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":compiled_model_for_test",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/serving/decision_forest",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_googletest//:gtest_main",
    ],
)
