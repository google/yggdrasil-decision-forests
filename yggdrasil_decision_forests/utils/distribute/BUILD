load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    #features = ["-layering_check"],
    licenses = ["notice"],
)

# Groups
# ======

# All the available distribution implementations.
# This can be *very* large, and most pipelines should only depends on the one
# needed.
cc_library_ydf(
    name = "all_implementations",
    deps = [
        "//yggdrasil_decision_forests/utils/distribute/implementations/grpc",
        "//yggdrasil_decision_forests/utils/distribute/implementations/multi_thread",
    ],
    alwayslink = 1,
)

# All worker logic implementations.
cc_library_ydf(
    name = "all_workers",
    deps = [
        ":toy_worker",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_worker",
        "//yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees:worker",
        "//yggdrasil_decision_forests/learner/generic_worker",
        "//yggdrasil_decision_forests/utils/distribute_cli:distribute_cli_worker",
    ],
    alwayslink = 1,
)

# Proto
# ========

all_proto_library(
    name = "distribute_proto",
    srcs = ["distribute.proto"],
)

# Library
# =======

cc_library_ydf(
    name = "distribute",
    visibility = ["//visibility:public"],
    deps = [
        "distribute_without_implementations",
        ":all_implementations",
    ],
)

cc_library_ydf(
    name = "distribute_without_implementations",
    srcs = ["distribute.cc"],
    hdrs = ["distribute.h"],
    deps = [
        ":core",
        ":distribute_cc_proto",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
)

cc_library_ydf(
    name = "core",
    srcs = ["core.cc"],
    hdrs = ["core.h"],
    deps = [
        ":distribute_cc_proto",
        "@com_google_absl//absl/status",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:protobuf",
        "//yggdrasil_decision_forests/utils:registration",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
)

cc_library_ydf(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        ":core",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:test",
    ],
)

cc_library_ydf(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = ["//yggdrasil_decision_forests/utils:concurrency"],
)

# Tests
# =====

cc_library_ydf(
    name = "toy_worker",
    srcs = ["toy_worker.cc"],
    hdrs = ["toy_worker.h"],
    deps = [
        ":core",
        "@com_google_absl//absl/synchronization",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:synchronization_primitives",
    ],
    alwayslink = 1,
)

cc_test(
    name = "distribute_test",
    srcs = ["distribute_test.cc"],
    deps = [
        "test_utils",
        ":distribute",
        ":distribute_without_implementations",
        ":toy_worker",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils/distribute/implementations/multi_thread:multi_thread_cc_proto",
    ],
)
