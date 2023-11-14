load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# ========

cc_library_ydf(
    name = "tf_record",
    srcs = ["tf_record.cc"],
    hdrs = ["tf_record.h"],
    deps = [
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:protobuf",
        "//yggdrasil_decision_forests/utils:sharded_io",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/crc:crc32c",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_ydf(
    name = "tf_record_tf_example",
    srcs = ["tf_record_tf_example.cc"],
    hdrs = ["tf_record_tf_example.h"],
    deps = [
        ":tf_example_cc_proto",
        ":tf_record",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:example_reader_interface",
        "//yggdrasil_decision_forests/dataset:example_writer_interface",
        "//yggdrasil_decision_forests/dataset/tensorflow:tf_example_io_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "tf_example",
    srcs = ["tf_example.cc"],
    hdrs = ["tf_example.h"],
    defines = select({
        "//yggdrasil_decision_forests:use_ydf_tensorflow_proto": ["YGG_USE_YDF_TENSORFLOW_PROTO"],
        "//conditions:default": [],
    }),
    features = ["-layering_check"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ] + select({
        "//yggdrasil_decision_forests:use_ydf_tensorflow_proto": [":tf_example_cc_proto"],
        "//conditions:default": ["//yggdrasil_decision_forests/dataset/tensorflow"],
    }),
)

# Proto
# ========

all_proto_library(
    name = "tf_example_proto",
    srcs = [
        "example.proto",
        "feature.proto",
    ],
)

# Tests
# =====

cc_test(
    name = "tf_record_test",
    srcs = ["tf_record_test.cc"],
    data = ["//yggdrasil_decision_forests/test_data"],
    deps = [
        ":tf_example",
        ":tf_record",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tf_record_tf_example_test",
    srcs = ["tf_record_tf_example_test.cc"],
    data = ["//yggdrasil_decision_forests/test_data"],
    deps = [
        ":tf_example",
        ":tf_record_tf_example",
        "//yggdrasil_decision_forests/dataset:example_reader",
        "//yggdrasil_decision_forests/dataset:example_reader_interface",
        "//yggdrasil_decision_forests/dataset:example_writer",
        "//yggdrasil_decision_forests/dataset:example_writer_interface",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
