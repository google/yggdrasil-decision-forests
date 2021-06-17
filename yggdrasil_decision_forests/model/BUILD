load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Model groups
# ==============

# Add new models here.
cc_library_ydf(
    name = "all_models",
    deps = [
        "//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "//yggdrasil_decision_forests/model/random_forest",
        "//yggdrasil_decision_forests/serving/decision_forest:register_engines",
    ],
)

# Libraries
# =========

cc_library_ydf(
    name = "abstract_model",
    srcs = [
        "abstract_model.cc",
        "fast_engine_factory.cc",
    ],
    hdrs = [
        "abstract_model.h",
        "fast_engine_factory.h",
    ],
    deps = [
        ":abstract_model_cc_proto",
        ":prediction_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:weight",
        "//yggdrasil_decision_forests/dataset:weight_cc_proto",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/metric:metric_cc_proto",
        "//yggdrasil_decision_forests/serving:example_set",
        "//yggdrasil_decision_forests/serving:fast_engine",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:distribution",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:registration",
    ],
)

# Note: The hyper parameter optimizer and the model library are in the same cc_library_ydf because they
# co-depend on each others.
cc_library_ydf(
    name = "model_library",
    srcs = [
        "model_library.cc",
    ],
    hdrs = [
        "model_library.h",
    ],
    deps = [
        ":abstract_model",
        ":abstract_model_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
)

cc_library_ydf(
    name = "model_engine_wrapper",
    srcs = [
        "model_engine_wrapper.cc",
    ],
    hdrs = [
        "model_engine_wrapper.h",
    ],
    deps = [
        ":abstract_model",
        ":prediction_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/serving:example_set",
        "//yggdrasil_decision_forests/serving:fast_engine",
        "//yggdrasil_decision_forests/utils:logging",
    ],
)

# Proto
# ========

all_proto_library(
    name = "abstract_model_proto",
    srcs = ["abstract_model.proto"],
    deps = [
        "//yggdrasil_decision_forests/dataset:data_spec_proto",
        "//yggdrasil_decision_forests/dataset:weight_proto",
    ],
)

all_proto_library(
    name = "prediction_proto",
    srcs = ["prediction.proto"],
    deps = ["//yggdrasil_decision_forests/utils:distribution_proto"],
)

# Tests
# =====

cc_test(
    name = "abstract_model_test",
    srcs = ["abstract_model_test.cc"],
    deps = [
        ":abstract_model",
        ":abstract_model_cc_proto",
        ":prediction_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/serving:example_set",
        "//yggdrasil_decision_forests/serving:fast_engine",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:protobuf",
        "//yggdrasil_decision_forests/utils:test",
    ],
)

cc_test(
    name = "model_library_test",
    srcs = ["model_library_test.cc"],
    deps = [
        ":abstract_model",
        ":model_library",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
    ],
)
