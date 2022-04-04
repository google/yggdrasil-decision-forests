load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Proto
# ========

all_proto_library(
    name = "distribute_cli_proto",
    srcs = ["distribute_cli.proto"],
    deps = ["//yggdrasil_decision_forests/utils/distribute:distribute_proto"],
)

# Library
# =======

cc_library_ydf(
    name = "distribute_cli",
    srcs = ["distribute_cli.cc"],
    hdrs = ["distribute_cli.h"],
    deps = [
        ":common",
        ":distribute_cli_cc_proto",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:hash",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils:uid",
        "//yggdrasil_decision_forests/utils/distribute",
        "//yggdrasil_decision_forests/utils/distribute:distribute_cc_proto",
        "//yggdrasil_decision_forests/utils/distribute:distribute_without_implementations",
    ],
)

cc_library_ydf(
    name = "distribute_cli_worker",
    srcs = ["distribute_cli_worker.cc"],
    hdrs = ["distribute_cli_worker.h"],
    deps = [
        ":common",
        ":distribute_cli_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:protobuf",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils/distribute:core",
        "//yggdrasil_decision_forests/utils/distribute:distribute_cc_proto",
    ],
    alwayslink = 1,
)

cc_library_ydf(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:hash",
        "//yggdrasil_decision_forests/utils:logging",
    ],
)

cc_test(
    name = "distribute_cli_test",
    srcs = ["distribute_cli_test.cc"],
    deps = [
        ":distribute_cli",
        ":distribute_cli_worker",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils/distribute",
        "//yggdrasil_decision_forests/utils/distribute:distribute_without_implementations",
        "//yggdrasil_decision_forests/utils/distribute/implementations/multi_thread:multi_thread_cc_proto",
    ],
)
