load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library_ydf(
    name = "all_dataset_formats",
    deps = [
        ":csv_example_reader",
        ":csv_example_writer",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_reader",
    ] + select({
        "//yggdrasil_decision_forests:use_tensorflow_io": [":tf_example_io_tfrecord"],
        "//conditions:default": [
        ],
    }) + select({
        "//conditions:default": [
        ],
    }),
    alwayslink = 1,
)

# Libraries
# ========

cc_library_ydf(
    name = "formats",
    srcs = [
        "formats.cc",
    ],
    hdrs = [
        "formats.h",
    ],
    deps = [
        ":formats_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:logging",
    ],
)

cc_library_ydf(
    name = "data_spec",
    srcs = [
        "data_spec.cc",
    ],
    hdrs = [
        "data_spec.h",
    ],
    deps = [
        ":data_spec_cc_proto",
        ":example_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:regex",
        "//yggdrasil_decision_forests/utils:hash",
    ] + select({
        "//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ],
        "//conditions:default": ["@org_tensorflow//tensorflow/core:protos_all_cc"],
    }),
)

cc_library_ydf(
    name = "example_reader",
    srcs = [
        "example_reader.cc",
    ],
    hdrs = [
        "example_reader.h",
    ],
    deps = [
        ":data_spec_cc_proto",
        ":example_reader_interface",
        ":formats",
        ":formats_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
)

cc_library_ydf(
    name = "example_reader_interface",
    hdrs = ["example_reader_interface.h"],
    deps = [
        ":data_spec_cc_proto",
        ":example_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:registration",
    ],
)

cc_library_ydf(
    name = "csv_example_reader",
    srcs = [
        "csv_example_reader.cc",
    ],
    hdrs = [
        "csv_example_reader.h",
    ],
    deps = [
        ":data_spec",
        ":data_spec_cc_proto",
        ":data_spec_inference",
        ":example_cc_proto",
        ":example_reader_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:csv",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:sharded_io",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "vertical_dataset",
    srcs = [
        "vertical_dataset.cc",
    ],
    hdrs = [
        "vertical_dataset.h",
    ],
    deps = [
        ":data_spec",
        ":data_spec_cc_proto",
        ":example_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
)

cc_library_ydf(
    name = "vertical_dataset_html",
    srcs = [
        "vertical_dataset_html.cc",
    ],
    hdrs = [
        "vertical_dataset_html.h",
    ],
    deps = [
        ":data_spec_cc_proto",
        ":vertical_dataset",
        "//yggdrasil_decision_forests/utils:html",
    ],
)

cc_library_ydf(
    name = "vertical_dataset_io",
    srcs = ["vertical_dataset_io.cc"],
    hdrs = ["vertical_dataset_io.h"],
    deps = [
        ":data_spec_cc_proto",
        ":example_cc_proto",
        ":example_reader",
        ":example_reader_interface",
        ":example_writer",
        ":example_writer_interface",
        ":formats",
        ":formats_cc_proto",
        ":vertical_dataset",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:sharded_io",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
)

cc_library_ydf(
    name = "example_writer",
    srcs = [
        "example_writer.cc",
    ],
    hdrs = [
        "example_writer.h",
    ],
    deps = [
        ":data_spec",
        ":data_spec_cc_proto",
        ":example_cc_proto",
        ":example_writer_interface",
        ":formats",
        ":formats_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:filesystem",
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
    name = "example_writer_interface",
    hdrs = ["example_writer_interface.h"],
    deps = [
        ":data_spec_cc_proto",
        ":example_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:registration",
    ],
)

cc_library_ydf(
    name = "csv_example_writer",
    srcs = [
        "csv_example_writer.cc",
    ],
    hdrs = [
        "csv_example_writer.h",
    ],
    deps = [
        ":data_spec",
        ":data_spec_cc_proto",
        ":example_cc_proto",
        ":example_writer_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:csv",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:sharded_io",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "data_spec_inference",
    srcs = [
        "data_spec_inference.cc",
    ],
    hdrs = [
        "data_spec_inference.h",
    ],
    deps = [
        ":data_spec",
        ":data_spec_cc_proto",
        ":formats",
        ":formats_cc_proto",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:accurate_sum",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:registration",
        "//yggdrasil_decision_forests/utils:sharded_io",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
)

cc_library_ydf(
    name = "tf_example_io_interface",
    srcs = [
        "tf_example_io_interface.cc",
    ],
    hdrs = [
        "tf_example_io_interface.h",
    ],
    deps = [
        ":data_spec",
        ":data_spec_cc_proto",
        ":data_spec_inference",
        ":example_cc_proto",
        ":example_reader_interface",
        ":example_writer_interface",
        ":formats",
        ":formats_cc_proto",
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
    name = "tf_example_io_tfrecord",
    srcs = [
        "tf_example_io_tfrecord.cc",
    ],
    hdrs = [
        "tf_example_io_tfrecord.h",
    ],
    deps = [
        ":data_spec_cc_proto",
        ":data_spec_inference",
        ":example_reader_interface",
        ":example_writer_interface",
        ":tf_example_io_interface",
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

cc_library_ydf(
    name = "weight",
    srcs = [
        "weight.cc",
    ],
    hdrs = [
        "weight.h",
    ],
    deps = [
        ":data_spec",
        ":data_spec_cc_proto",
        ":example_cc_proto",
        ":vertical_dataset",
        ":weight_cc_proto",
        "@com_google_absl//absl/status",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
)

cc_library_ydf(
    name = "synthetic_dataset",
    srcs = [
        "synthetic_dataset.cc",
    ],
    hdrs = [
        "synthetic_dataset.h",
    ],
    deps = [
        ":formats",
        ":formats_cc_proto",
        ":synthetic_dataset_cc_proto",
        ":tf_example_io_interface",
        ":vertical_dataset",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:hash",
        "//yggdrasil_decision_forests/utils:csv",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:random",
    ] + select({
        "//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ],
        "//conditions:default": ["@org_tensorflow//tensorflow/core:protos_all_cc"],
    }),
)

# Proto
# ========

all_proto_library(
    name = "formats_proto",
    srcs = ["formats.proto"],
)

all_proto_library(
    name = "data_spec_proto",
    srcs = ["data_spec.proto"],
)

all_proto_library(
    name = "example_proto",
    srcs = ["example.proto"],
)

all_proto_library(
    name = "weight_proto",
    srcs = ["weight.proto"],
)

all_proto_library(
    name = "synthetic_dataset_proto",
    srcs = ["synthetic_dataset.proto"],
)

# Tests
# =====

cc_test(
    name = "formats_test",
    srcs = ["formats_test.cc"],
    deps = [
        ":formats",
        ":formats_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "data_spec_test",
    size = "small",
    srcs = ["data_spec_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":all_dataset_formats",
        ":data_spec",
        ":data_spec_cc_proto",
        ":data_spec_inference",
        ":example_cc_proto",
        #"@com_google_googletest//:gtest_main", # When fixed
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
    ] + select({
        "//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ],
        "//conditions:default": ["@org_tensorflow//tensorflow/core:protos_all_cc"],
    }),
)

cc_test(
    name = "example_reader_test",
    srcs = ["example_reader_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":all_dataset_formats",
        ":data_spec",
        ":data_spec_cc_proto",
        ":data_spec_inference",
        ":example_cc_proto",
        ":example_reader",
        ":example_reader_interface",
        #"@com_google_googletest//:gtest_main", # When fixed
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/utils:compatibility",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
    ],
)

cc_test(
    name = "vertical_dataset_test",
    srcs = ["vertical_dataset_test.cc"],
    data = ["//yggdrasil_decision_forests/test_data"],
    deps = [
        ":all_dataset_formats",
        ":data_spec",
        ":data_spec_cc_proto",
        ":data_spec_inference",
        ":example_cc_proto",
        ":vertical_dataset",
        ":vertical_dataset_io",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "//yggdrasil_decision_forests/utils:compatibility",
        "@com_google_absl//absl/strings",
        #"@com_google_googletest//:gtest_main", # When fixed
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
    ],
)

cc_test(
    name = "vertical_dataset_io_test",
    srcs = ["vertical_dataset_io_test.cc"],
    data = ["//yggdrasil_decision_forests/test_data"],
    deps = [
        ":all_dataset_formats",
        ":data_spec_cc_proto",
        ":data_spec_inference",
        ":example_cc_proto",
        ":vertical_dataset",
        ":vertical_dataset_io",
        #"@com_google_googletest//:gtest_main", # When fixed
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
    ],
)

cc_test(
    name = "example_writer_test",
    srcs = ["example_writer_test.cc"],
    deps = [
        ":all_dataset_formats",
        ":data_spec_cc_proto",
        ":example_cc_proto",
        ":example_writer",
        ":example_writer_interface",
        ":tf_example_io_interface",
        #"@com_google_googletest//:gtest_main", # When fixed
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//yggdrasil_decision_forests/utils:compatibility",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:sharded_io",
        "//yggdrasil_decision_forests/utils:test",
    ] + select({
        "//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ],
        "//conditions:default": ["@org_tensorflow//tensorflow/core:protos_all_cc"],
    }),
)

cc_test(
    name = "data_spec_inference_test",
    srcs = ["data_spec_inference_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":all_dataset_formats",
        ":data_spec",
        ":data_spec_cc_proto",
        ":data_spec_inference",
        ":formats",
        ":formats_cc_proto",
        ":synthetic_dataset",
        ":synthetic_dataset_cc_proto",
        #"@com_google_googletest//:gtest_main", # When fixed
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//yggdrasil_decision_forests/utils:compatibility",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:sharded_io",
        "//yggdrasil_decision_forests/utils:test",
    ],
)

cc_test(
    name = "tf_example_io_interface_test",
    srcs = ["tf_example_io_interface_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":tf_example_io_interface",
        ":tf_example_io_tfrecord",
        #"@com_google_googletest//:gtest_main", # When fixed
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/utils:compatibility",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:sharded_io",
        "//yggdrasil_decision_forests/utils:test",
    ] + select({
        "//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ],
        "//conditions:default": ["@org_tensorflow//tensorflow/core:protos_all_cc"],
    }),
)

cc_test(
    name = "vertical_dataset_html_test",
    srcs = ["vertical_dataset_html_test.cc"],
    data = ["//yggdrasil_decision_forests/test_data"],
    deps = [
        ":data_spec_cc_proto",
        ":data_spec_inference",
        ":vertical_dataset",
        ":vertical_dataset_io",
        #"@com_google_googletest//:gtest_main", # When fixed
        "@com_google_googletest//:gtest_main",
        ":vertical_dataset_html",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:html",
        ":all_dataset_formats",
        "//yggdrasil_decision_forests/utils:test",
    ],
)

cc_test(
    name = "weight_test",
    srcs = ["weight_test.cc"],
    deps = [
        ":data_spec_cc_proto",
        ":example_cc_proto",
        ":vertical_dataset",
        ":weight",
        ":weight_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
    ],
)

cc_test(
    name = "synthetic_dataset_test",
    srcs = ["synthetic_dataset_test.cc"],
    deps = [
        ":all_dataset_formats",
        ":data_spec",
        ":data_spec_cc_proto",
        ":data_spec_inference",
        ":synthetic_dataset",
        ":synthetic_dataset_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:test",
    ],
)
