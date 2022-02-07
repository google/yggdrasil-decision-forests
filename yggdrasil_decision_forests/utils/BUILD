load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Library
# =======

alias(
    name = "logging",
    actual = ":logging_default",
)

# Default logging.
cc_library_ydf(
    name = "logging_default",
    srcs = [
        "logging_default.cc",
    ],
    hdrs = [
        "logging.h",
        "logging_default.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:usage",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

alias(
    name = "status_macros",
    actual = ":status_macros_default",
)

# Default macros.
cc_library_ydf(
    name = "status_macros_default",
    hdrs = [
        "status_macros.h",
        "status_macros_default.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":logging",
    ],
)

alias(
    name = "filesystem",
    actual = select({
        "//yggdrasil_decision_forests:use_tensorflow_io": ":filesystem_tensorflow",
        "//conditions:default": ":filesystem_default",
    }),
)

# Default filesystem.
cc_library_ydf(
    name = "filesystem_default",
    srcs = [
        "filesystem.cc",
        "filesystem_default.cc",
    ],
    hdrs = [
        "filesystem.h",
        "filesystem_default.h",
    ],
    copts = ["-fexceptions"],  # The code does not rely on exceptions.
    defines = ["YGG_FILESYSTEM_USES_DEFAULT"],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    visibility = ["//visibility:private"],
    linkopts = select({
        "@platforms//os:linux": ["-lstdc++fs"],  # Required for gcc<=8.
        "//conditions:default": [],
    }),
    deps = [
        ":bytestream",
        ":compatibility",
        ":logging_default",
        ":protobuf",
        ":status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

# TensorFlow::io based filesystem.
cc_library_ydf(
    name = "filesystem_tensorflow",
    srcs = [
        "filesystem.cc",
        "filesystem_tensorflow.cc",
    ],
    hdrs = [
        "filesystem.h",
        "filesystem_tensorflow.h",
    ],
    copts = ["-fexceptions"],
    defines = [
        "YGG_FILESYSTEM_USES_DEFAULT",
        "YGG_FILESYSTEM_USES_TENSORFLOW",
    ],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    visibility = ["//visibility:private"],
    deps = [
        ":compatibility",
        ":bytestream",
        ":status_macros",
        ":tensorflow",
        "@com_google_protobuf//:protobuf",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ] + select({
        "//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ],
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core/platform:path",
            "@org_tensorflow//tensorflow/core:framework",
            "@org_tensorflow//tensorflow/core:lib",
        ],
    }),
)

cc_library_ydf(
    name = "test",
    testonly = 1,
    srcs = [
        "test.cc",
    ],
    hdrs = [
        "test.h",
    ],
    deps = [
        ":filesystem",
        ":logging",
        ":uid",
        "@com_google_protobuf//:protobuf",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_ydf(
    name = "sharded_io",
    srcs = [
        "sharded_io.cc",
    ],
    hdrs = [
        "sharded_io.h",
    ],
    deps = [
        ":compatibility",
        ":filesystem",
        ":logging",
        ":status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_ydf(
    name = "sharded_io_tfrecord",
    hdrs = [
        "sharded_io_tfrecord.h",
    ],
    deps = [
        ":sharded_io",
        ":status_macros",
        ":tensorflow",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ] + select({
        "//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ],
        "//conditions:default": ["@org_tensorflow//tensorflow/core:lib"],
    }),
)

cc_library_ydf(
    name = "regex",
    srcs = [
        "regex.cc",
    ],
    hdrs = [
        "regex.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library_ydf(
    name = "protobuf",
    srcs = [
        "protobuf.cc",
    ],
    hdrs = [
        "protobuf.h",
    ],
    deps = [
        ":compatibility",
        ":logging",
        "@com_google_protobuf//:protobuf",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_ydf(
    name = "registration",
    srcs = [
        "registration.cc",
    ],
    hdrs = [
        "registration.h",
    ],
    deps = [
        ":compatibility",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:synchronization_primitives",
    ],
)

cc_library_ydf(
    name = "bytestream",
    srcs = [
        "bytestream.cc",
    ],
    hdrs = [
        "bytestream.h",
    ],
    deps = [
        ":compatibility",
        ":status_macros",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library_ydf(
    name = "csv",
    srcs = [
        "csv.cc",
    ],
    hdrs = [
        "csv.h",
    ],
    deps = [
        ":bytestream",
        ":logging",
        ":status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library_ydf(
    name = "accurate_sum",
    hdrs = [
        "accurate_sum.h",
    ],
    deps = [
    ],
)

alias(
    name = "concurrency",
    actual = ":concurrency_default",
)

# Default logging.
cc_library_ydf(
    name = "concurrency_default",
    srcs = [
        "concurrency_default.cc",
    ],
    hdrs = [
        "concurrency.h",
        "concurrency_channel.h",
        "concurrency_default.h",
        "concurrency_streamprocessor.h",
    ],
    defines = ["YGG_CONCURRENCY_USES_DEFAULT"],
    visibility = ["//visibility:private"],
    deps = [
        ":logging",
        ":synchronization_primitives",
        "@com_google_absl//absl/status",
    ],
)

cc_library_ydf(
    name = "synchronization_primitives",
    hdrs = [
        "synchronization_primitives.h",
    ],
    defines = select({
        "//yggdrasil_decision_forests:std_synchronization_primitives": [
            "YGG_STD_MUTEX",
            "YGG_STD_BLOCKING_COUNTER_AND_NOTIFICATION",
            # When YGG_STD_MUTEX and YGG_STD_BLOCKING_COUNTER_AND_NOTIFICATION are set,
            # this build rule relies on c++17 and c++20 primitives (see the comments in
            # synchronization_primitives.h). However, this rule can effectively be compiled with
            # c++14 using a modern enought compiler with availabilities disabled (done by defining
            # _LIBCPP_DISABLE_AVAILABILITY).
            #
            # See:
            # https://releases.llvm.org/3.1/tools/clang/docs/LanguageExtensions.html#availability
            "_LIBCPP_DISABLE_AVAILABILITY",
        ],
        "//conditions:default": [],
    }),
    deps = select({
        "//yggdrasil_decision_forests:std_synchronization_primitives": [],
        "//conditions:default": ["@com_google_absl//absl/synchronization"],
    }),
)

cc_library_ydf(
    name = "html",
    srcs = [
        "html.cc",
    ],
    hdrs = [
        "html.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library_ydf(
    name = "distribution",
    srcs = [
        "distribution.cc",
    ],
    hdrs = [
        "distribution.h",
    ],
    deps = [
        ":distribution_cc_proto",
        ":html",
        ":logging",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
    ],
)

cc_library_ydf(
    name = "fold_generator",
    srcs = [
        "fold_generator.cc",
    ],
    hdrs = [
        "fold_generator.h",
    ],
    deps = [
        ":csv",
        ":filesystem",
        ":fold_generator_cc_proto",
        ":random",
        ":status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
    ],
)

cc_library_ydf(
    name = "hyper_parameters",
    srcs = [
        "hyper_parameters.cc",
    ],
    hdrs = [
        "hyper_parameters.h",
    ],
    deps = [
        ":compatibility",
        ":logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
    ],
)

cc_library_ydf(
    name = "bitmap",
    srcs = [
        "bitmap.cc",
    ],
    hdrs = [
        "bitmap.h",
    ],
    deps = [
        ":bitmap_cc_proto",
        ":concurrency",
        ":filesystem",
        ":logging",
        ":status_macros",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:synchronization_primitives",
    ],
)

cc_library_ydf(
    name = "histogram",
    hdrs = ["histogram.h"],
    deps = [
        ":accurate_sum",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library_ydf(
    name = "adaptive_work",
    srcs = [
        "adaptive_work.cc",
    ],
    hdrs = [
        "adaptive_work.h",
    ],
    deps = [
        ":compatibility",
        ":logging",
        "@com_google_absl//absl/base:core_headers",
        "//yggdrasil_decision_forests/utils:synchronization_primitives",
    ],
)

cc_library_ydf(
    name = "usage",
    hdrs = ["usage.h"],
    deps = select({
        "//conditions:default": [":usage_default"],
    }) + [
        "@com_google_absl//absl/time",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
    ],
)

cc_library_ydf(
    name = "usage_default",
    srcs = ["usage_default.cc"],
    hdrs = ["usage.h"],
    visibility = ["//visibility:private"],
    deps = [
        "@com_google_absl//absl/time",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
    ],
)

cc_library_ydf(
    name = "feature_importance",
    srcs = [
        "feature_importance.cc",
    ],
    hdrs = [
        "feature_importance.h",
    ],
    deps = [
        ":compatibility",
        ":random",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/metric:metric_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
    ],
)

cc_library_ydf(
    name = "test_utils",
    testonly = 1,
    srcs = [
        "test_utils.cc",
    ],
    hdrs = [
        "test_utils.h",
    ],
    deps = [
        ":compatibility",
        ":concurrency",
        ":csv",
        ":distribution_cc_proto",
        ":filesystem",
        ":logging",
        ":random",
        ":test",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:synthetic_dataset",
        "//yggdrasil_decision_forests/dataset:synthetic_dataset_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/dataset:weight_cc_proto",
        "//yggdrasil_decision_forests/learner:abstract_learner",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/metric:metric_cc_proto",
        "//yggdrasil_decision_forests/metric:report",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model:model_engine_wrapper",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/model:prediction_cc_proto",
        "//yggdrasil_decision_forests/serving:example_set",
        "//yggdrasil_decision_forests/serving:fast_engine",
        "//yggdrasil_decision_forests/serving/decision_forest",
    ],
)

cc_library_ydf(
    name = "evaluation",
    srcs = ["evaluation.cc"],
    hdrs = ["evaluation.h"],
    defines = select({
        "//yggdrasil_decision_forests:use_tensorflow_io": ["YGG_TFRECORD_PREDICTIONS"],
        "//conditions:default": [
        ],
    }),
    deps = [
        ":compatibility",
        ":distribution_cc_proto",
        ":status_macros",
        ":distribution",
        "@com_google_absl//absl/status",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_writer",
        "//yggdrasil_decision_forests/dataset:formats",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model:prediction_cc_proto",
    ] + select({
        "//yggdrasil_decision_forests:use_tensorflow_io": [":sharded_io_tfrecord"],
        "//conditions:default": [
        ],
    }),
)

cc_library_ydf(
    name = "circular_buffer",
    hdrs = ["circular_buffer.h"],
    deps = [
        ":logging",
    ],
)

cc_library_ydf(
    name = "cast",
    hdrs = ["cast.h"],
)

cc_library_ydf(
    name = "random",
    hdrs = ["random.h"],
)

cc_library_ydf(
    name = "tensorflow",
    hdrs = ["tensorflow.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ] + select({
        "//yggdrasil_decision_forests:tensorflow_with_header_lib": [
            "@tensorflow_pypi//:libtensorflow_framework",
            "@tensorflow_pypi//:tf_header_lib",
        ],
        "//conditions:default": ["@org_tensorflow//tensorflow/core/platform:status"],
    }),
)

cc_library_ydf(
    name = "compatibility",
    hdrs = ["compatibility.h"],
    defines = select({
        "//yggdrasil_decision_forests:no_absl_statusor": ["YGG_ABSL_NO_STATUSOR"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/types:optional",
    ] + select({
        "//yggdrasil_decision_forests:no_absl_statusor": [
            "@com_google_absl//absl/status:status",
        ],
        "//conditions:default": ["@com_google_absl//absl/status:statusor"],
    }),
)

cc_library_ydf(
    name = "blob_sequence",
    srcs = ["blob_sequence.cc"],
    hdrs = ["blob_sequence.h"],
    deps = [
        ":bytestream",
        ":compatibility",
        ":filesystem",
        ":status_macros",
    ],
)

cc_library_ydf(
    name = "sharded_io_blob_sequence",
    hdrs = [
        "sharded_io_blob_sequence.h",
    ],
    deps = [
        ":blob_sequence",
        ":filesystem",
        ":sharded_io",
        ":status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_ydf(
    name = "snapshot",
    srcs = ["snapshot.cc"],
    hdrs = ["snapshot.h"],
    deps = [
        ":compatibility",
        ":filesystem",
        ":status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_ydf(
    name = "uid",
    hdrs = ["uid.h"],
    deps = [
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library_ydf(
    name = "math",
    hdrs = ["math.h"],
)

cc_library_ydf(
    name = "hash",
    hdrs = ["hash.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ] + [
        "@farmhash_archive//:farmhash",
    ],
)

# Bzl Libraries
# =============

bzl_library(
    name = "compile_bzl",
    srcs = ["compile.bzl"],
)

# Proto
# ========

all_proto_library(
    name = "distribution_proto",
    srcs = ["distribution.proto"],
)

all_proto_library(
    name = "fold_generator_proto",
    srcs = ["fold_generator.proto"],
)

all_proto_library(
    name = "bitmap_proto",
    srcs = ["bitmap.proto"],
)

# Tests
# =====

cc_test(
    name = "logging_test",
    srcs = ["logging_test.cc"],
    deps = [
        ":logging",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "sharded_io_test",
    srcs = ["sharded_io_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":filesystem",
        ":sharded_io",
        ":test",
        # "@com_google_googletest//:gtest_main",  # When fixed
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        ":compatibility",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "regex_test",
    srcs = ["regex_test.cc"],
    deps = [
        ":regex",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "filesystem_test",
    srcs = ["filesystem_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":distribution_cc_proto",
        ":filesystem",
        # "@com_google_googletest//:gtest_main",  # When fixed
        "@com_google_googletest//:gtest_main",
        ":test",
    ],
)

cc_test(
    name = "protobuf_test",
    srcs = ["protobuf_test.cc"],
    deps = [
        ":distribution_cc_proto",
        ":protobuf",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "registration_test",
    srcs = ["registration_test.cc"],
    deps = [
        ":registration",
        ":test",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "status_macros_test",
    srcs = ["status_macros_test.cc"],
    deps = [
        ":compatibility",
        ":logging",
        ":status_macros",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "bytestream_test",
    srcs = ["bytestream_test.cc"],
    deps = [
        ":bytestream",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "csv_test",
    srcs = ["csv_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":csv",
        # "@com_google_googletest//:gtest_main",  # When fixed
        "@com_google_googletest//:gtest_main",
        ":test",
        ":filesystem",
    ],
)

cc_test(
    name = "accurate_sum_test",
    srcs = ["accurate_sum_test.cc"],
    deps = [
        ":accurate_sum",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "concurrency_test",
    srcs = ["concurrency_test.cc"],
    deps = [
        ":concurrency",
        # "@com_google_googletest//:gtest_main",  # When fixed
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "html_test",
    srcs = ["html_test.cc"],
    deps = [
        ":html",
        # "@com_google_googletest//:gtest_main",  # When fixed
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "distribution_test",
    size = "large",
    srcs = ["distribution_test.cc"],
    deps = [
        ":distribution",
        ":distribution_cc_proto",
        ":test",
        # "@com_google_googletest//:gtest_main",  # When fixed
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
    ],
)

cc_test(
    name = "fold_generator_test",
    srcs = ["fold_generator_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":compatibility",
        ":filesystem",
        ":fold_generator",
        ":status_macros",
        ":test",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:csv_example_reader",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/metric:metric_cc_proto",
    ],
)

cc_test(
    name = "hyper_parameters_test",
    srcs = ["hyper_parameters_test.cc"],
    deps = [
        ":hyper_parameters",
        ":test",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
    ],
)

cc_test(
    name = "bitmap_test",
    size = "large",
    srcs = ["bitmap_test.cc"],
    shard_count = 12,
    deps = [
        ":bitmap",
        ":filesystem",
        ":logging",
        ":test",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "histogram_test",
    srcs = ["histogram_test.cc"],
    deps = [
        ":histogram",
        ":logging",
        ":random",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "adaptive_work_test",
    srcs = ["adaptive_work_test.cc"],
    deps = [
        ":adaptive_work",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "evaluation_test",
    srcs = ["evaluation_test.cc"],
    defines = select({
        "//yggdrasil_decision_forests:use_tensorflow_io": ["YDF_EVAL_TFRECORD"],
        "//conditions:default": [],
    }),
    deps = [
        ":evaluation",
        ":filesystem",
        ":test",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/dataset:csv_example_reader",
        "//yggdrasil_decision_forests/dataset:csv_example_writer",
        "//yggdrasil_decision_forests/dataset:example_reader",
    ] + select({
        "//yggdrasil_decision_forests:use_tensorflow_io": [":sharded_io_tfrecord"],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "circular_buffer_test",
    srcs = ["circular_buffer_test.cc"],
    deps = [
        ":circular_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "blob_sequence_test",
    srcs = ["blob_sequence_test.cc"],
    deps = [
        ":blob_sequence",
        ":filesystem",
        ":test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "snapshot_test",
    srcs = ["snapshot_test.cc"],
    deps = [
        ":filesystem",
        ":snapshot",
        ":test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "math_test",
    srcs = ["math_test.cc"],
    deps = [
        ":math",
        ":test",
        "@com_google_googletest//:gtest_main",
    ],
)
