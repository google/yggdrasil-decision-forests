load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========================================================

cc_library_ydf(
    name = "decision_tree",
    srcs = [
        "decision_tree.cc",
        "decision_tree_io.cc",
        "structure_analysis.cc",
    ],
    hdrs = [
        "decision_tree.h",
        "decision_tree_io.h",
        "structure_analysis.h",
    ],
    deps = [
        ":decision_tree_cc_proto",
        ":decision_tree_io_blob_sequence",
        ":decision_tree_io_interface",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:bitmap",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:histogram",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:sharded_io",
        "//yggdrasil_decision_forests/utils:status_macros",
    ] + select({
        "//conditions:default": [
        ],
    }),
)

cc_library_ydf(
    name = "decision_tree_io_interface",
    hdrs = [
        "decision_tree_io_interface.h",
    ],
    deps = [
        ":decision_tree_cc_proto",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:registration",
        "//yggdrasil_decision_forests/utils:sharded_io",
    ],
)

# Storage of nodes in TFRecord.
cc_library_ydf(
    name = "decision_tree_io_tfrecord",
    srcs = [
        "decision_tree_io_tfrecord.cc",
    ],
    deps = [
        ":decision_tree_cc_proto",
        ":decision_tree_io_interface",
        "@com_google_absl//absl/memory",
        "//yggdrasil_decision_forests/utils:sharded_io",
        "//yggdrasil_decision_forests/utils:sharded_io_tfrecord",
    ],
    alwayslink = 1,
)

# Storage of nodes in BlobSequence.
cc_library_ydf(
    name = "decision_tree_io_blob_sequence",
    srcs = ["decision_tree_io_blob_sequence.cc"],
    deps = [
        ":decision_tree_cc_proto",
        ":decision_tree_io_interface",
        "//yggdrasil_decision_forests/utils:sharded_io",
        "//yggdrasil_decision_forests/utils:sharded_io_blob_sequence",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "decision_forest_interface",
    hdrs = ["decision_forest_interface.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
    ],
)

# Proto
# ========

all_proto_library(
    name = "decision_tree_proto",
    srcs = ["decision_tree.proto"],
    deps = ["//yggdrasil_decision_forests/utils:distribution_proto"],
)

# Unit Tests
# =========================================================

cc_test(
    name = "decision_tree_test",
    srcs = ["decision_tree_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":decision_tree",
        ":decision_tree_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:csv_example_reader",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
    ],
)
