load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Proto
# ========

all_proto_library(
    name = "dataset_cache_proto",
    srcs = ["dataset_cache.proto"],
    deps = ["//yggdrasil_decision_forests/dataset:data_spec_proto"],
)

# Libraries
# =========================================================

cc_library_ydf(
    name = "column_cache",
    srcs = ["column_cache.cc"],
    hdrs = ["column_cache.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils/distribute:core",
    ],
)

cc_library_ydf(
    name = "dataset_cache",
    srcs = ["dataset_cache.cc"],
    hdrs = ["dataset_cache.h"],
    deps = [
        ":column_cache",
        ":dataset_cache_cc_proto",
        ":dataset_cache_common",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:formats",
        "//yggdrasil_decision_forests/learner:types",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:math",
        "//yggdrasil_decision_forests/utils:sharded_io",
        "//yggdrasil_decision_forests/utils/distribute:core",
        "//yggdrasil_decision_forests/utils/distribute:distribute_cc_proto",
        "//yggdrasil_decision_forests/utils/distribute:distribute_without_implementations",
    ],
)

cc_library_ydf(
    name = "dataset_cache_reader",
    srcs = ["dataset_cache_reader.cc"],
    hdrs = ["dataset_cache_reader.h"],
    deps = [
        ":column_cache",
        ":dataset_cache",
        ":dataset_cache_cc_proto",
        ":dataset_cache_common",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:formats",
        "//yggdrasil_decision_forests/learner:types",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:sharded_io",
        "//yggdrasil_decision_forests/utils/distribute:core",
        "//yggdrasil_decision_forests/utils/distribute:distribute_cc_proto",
        "//yggdrasil_decision_forests/utils/distribute:distribute_without_implementations",
    ],
)

cc_library_ydf(
    name = "dataset_cache_worker",
    srcs = ["dataset_cache_worker.cc"],
    hdrs = ["dataset_cache_worker.h"],
    deps = [
        ":column_cache",
        ":dataset_cache_cc_proto",
        ":dataset_cache_common",
        "@com_google_absl//absl/status",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/learner:types",
        "//yggdrasil_decision_forests/learner/decision_tree:utils",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils:uid",
        "//yggdrasil_decision_forests/utils/distribute:core",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "dataset_cache_common",
    srcs = ["dataset_cache_common.cc"],
    hdrs = ["dataset_cache_common.h"],
    deps = [
        ":column_cache",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/learner:types",
        "//yggdrasil_decision_forests/learner/decision_tree:utils",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:uid",
    ],
)

# Test
# =========================================================

cc_test(
    name = "column_cache_test",
    srcs = ["column_cache_test.cc"],
    deps = [
        ":column_cache",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
    ],
)

cc_test(
    name = "dataset_cache_test",
    srcs = ["dataset_cache_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":column_cache",
        ":dataset_cache",
        ":dataset_cache_common",
        ":dataset_cache_reader",
        ":dataset_cache_worker",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "//yggdrasil_decision_forests/dataset:csv_example_reader",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:test_utils",
        "//yggdrasil_decision_forests/utils/distribute/implementations/multi_thread",
        "//yggdrasil_decision_forests/utils/distribute/implementations/multi_thread:multi_thread_cc_proto",
    ],
)

cc_test(
    name = "dataset_cache_reader_test",
    srcs = ["dataset_cache_reader_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":column_cache",
        ":dataset_cache",
        ":dataset_cache_common",
        ":dataset_cache_reader",
        ":dataset_cache_worker",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/dataset:csv_example_reader",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:test_utils",
        "//yggdrasil_decision_forests/utils/distribute/implementations/multi_thread",
        "//yggdrasil_decision_forests/utils/distribute/implementations/multi_thread:multi_thread_cc_proto",
    ],
)

cc_test(
    name = "dataset_cache_common_test",
    srcs = ["dataset_cache_common_test.cc"],
    deps = [
        ":dataset_cache_common",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
    ],
)
