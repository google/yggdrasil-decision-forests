load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library_ydf(
    name = "abstract_postprocessor",
    srcs = [
        "abstract_postprocessor.cc",
    ],
    hdrs = [
        "abstract_postprocessor.h",
    ],
    deps = [
        ":abstract_postprocessor_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/model:prediction_cc_proto",
        "@com_google_absl//absl/status",
    ],
)

all_proto_library(
    name = "abstract_postprocessor_proto",
    srcs = ["abstract_postprocessor.proto"],
)

cc_library_ydf(
    name = "postprocessor_library",
    srcs = [
        "postprocessor_library.cc",
    ],
    hdrs = [
        "postprocessor_library.h",
    ],
    deps = [
        ":abstract_postprocessor",
        ":abstract_postprocessor_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "postprocessor_library_test",
    srcs = ["postprocessor_library_test.cc"],
    deps = [
        ":abstract_postprocessor_cc_proto",
        ":postprocessor_library",
        "//yggdrasil_decision_forests/utils:test",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "abstract_postprocessor_test",
    srcs = ["abstract_postprocessor_test.cc"],
    deps = [
        ":abstract_postprocessor",
        ":abstract_postprocessor_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/model:prediction_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)
