load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Proto
# ========

all_proto_library(
    name = "grpc_proto",
    srcs = ["grpc.proto"],
    has_services = True,
    deps = ["//yggdrasil_decision_forests/utils/distribute:distribute_proto"],
)

# Worker binaries
# ===============

cc_binary(
    name = "grpc_worker_main",
    srcs = ["grpc_worker_main.cc"],
    deps = [
        ":grpc",
        "@com_google_absl//absl/flags:flag",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils/distribute:all_workers",
    ],
)

# Library
# =======

cc_library(
    name = "grpc",
    srcs = ["grpc.cc"],
    hdrs = ["grpc.h"],
    deps = [
        ":grpc_cc_proto",
        ":grpc_grpc_proto",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/synchronization",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils/distribute:core",
        "//yggdrasil_decision_forests/utils/distribute:distribute_cc_proto",
    ],
    alwayslink = 1,
)

# Tests
# =====

cc_test(
    name = "grpc_test",
    srcs = ["grpc_test.cc"],
    deps = [
        ":grpc",
        ":grpc_cc_proto",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils/distribute",
        "//yggdrasil_decision_forests/utils/distribute:distribute_without_implementations",
        "//yggdrasil_decision_forests/utils/distribute:test_utils",
        "//yggdrasil_decision_forests/utils/distribute:toy_worker",
    ],
)
