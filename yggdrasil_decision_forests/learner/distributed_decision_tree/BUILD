load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Proto
# ========

all_proto_library(
    name = "training_proto",
    srcs = ["training.proto"],
    deps = [
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_proto",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_proto",
    ],
)

# Libraries
# =========================================================

cc_library_ydf(
    name = "training",
    srcs = [
        "training.cc",
    ],
    hdrs = [
        "label_accessor.h",
        "splitter.h",
        "training.h",
    ],
    deps = [
        ":training_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:label",
        "//yggdrasil_decision_forests/learner/decision_tree:splitter",
        "//yggdrasil_decision_forests/learner/decision_tree:utils",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_common",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_reader",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:bitmap",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:distribution",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils:synchronization_primitives",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library_ydf(
    name = "histogram_bnb",
    srcs = ["histogram_bnb.cc"],
    hdrs = [
        "histogram_bnb.h",
        "label_accessor.h",
    ],
    deps = [
        ":training",
        ":training_cc_proto",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:splitter",
        "//yggdrasil_decision_forests/learner/decision_tree:utils",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:distribution",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library_ydf(
    name = "quantile_sketch",
    srcs = ["quantile_sketch.cc"],
    hdrs = ["quantile_sketch.h"],
    deps = [
        ":training_cc_proto",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

# Test
# =========================================================

cc_test(
    name = "histogram_bnb_test",
    srcs = ["histogram_bnb_test.cc"],
    deps = [
        ":histogram_bnb",
        ":quantile_sketch",
        ":training",
        ":training_cc_proto",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/learner/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:utils",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "quantile_sketch_test",
    srcs = ["quantile_sketch_test.cc"],
    deps = [
        ":quantile_sketch",
        ":training_cc_proto",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "training_test",
    srcs = ["training_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":training",
        "//yggdrasil_decision_forests/dataset:csv_example_reader",
        "//yggdrasil_decision_forests/dataset:csv_example_writer",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/learner:abstract_learner",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/decision_tree:training",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_worker",
        "//yggdrasil_decision_forests/utils:bitmap",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils/distribute/implementations/multi_thread",
        "//yggdrasil_decision_forests/utils/distribute/implementations/multi_thread:multi_thread_cc_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
