load("//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Library
# =======

cc_library_ydf(
    name = "tf_example_io_interface",
    srcs = [
        "tf_example_io_interface.cc",
    ],
    hdrs = [
        "tf_example_io_interface.h",
    ],
    deps = [
        ":tf_example",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_reader_interface",
        "//yggdrasil_decision_forests/dataset:example_writer_interface",
        "//yggdrasil_decision_forests/dataset:formats",
        "//yggdrasil_decision_forests/dataset:formats_cc_proto",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:registration",
        "//yggdrasil_decision_forests/utils:sharded_io",
        "//yggdrasil_decision_forests/utils:status_macros",
    ] + select({
        "//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ],
        "//conditions:default": ["@org_tensorflow//tensorflow/core:protos_all_cc"],
    }),
)

cc_library_ydf(
    name = "tf_example",
    srcs = ["tf_example.cc"],
    hdrs = ["tf_example.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/serving:example_set",
        "//yggdrasil_decision_forests/serving:tf_example",
    ] + select({
        "//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ],
        "//conditions:default": ["@org_tensorflow//tensorflow/core:protos_all_cc"],
    }),
)

cc_library_ydf(
    name = "tf_example_io_tfrecord",
    srcs = [
        "tf_example_io_tfrecord.cc",
    ],
    hdrs = [
        "tf_example_io_tfrecord.h",
    ],
    deps = [
        ":tf_example",
        ":tf_example_io_interface",
        "@com_google_absl//absl/types:optional",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:example_reader_interface",
        "//yggdrasil_decision_forests/dataset:example_writer_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:sharded_io_tfrecord",
    ] + select({
        "//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ],
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:lib",
            "@org_tensorflow//tensorflow/core:protos_all_cc",
        ],
    }),
    alwayslink = 1,
)

# Tests
# =====

cc_test(
    name = "tf_example_io_interface_test",
    srcs = ["tf_example_io_interface_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":tf_example_io_interface",
        ":tf_example_io_tfrecord",
        "@com_google_absl//absl/status:statusor",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_writer",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/dataset:example_writer_interface",
        "//yggdrasil_decision_forests/utils:compatibility",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:sharded_io",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/serving:tf_example",
        "//yggdrasil_decision_forests/serving:example_set",
    ] + select({
        "//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ],
        "//conditions:default": ["@org_tensorflow//tensorflow/core:protos_all_cc"],
    }),
)

cc_test(
    name = "tf_example_test",
    srcs = ["tf_example_test.cc"],
    deps = [
        ":tf_example",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/core/example:example_protos_cc",
        "@org_tensorflow//tensorflow/core/example:feature_util",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/utils:test",
    ],
)
