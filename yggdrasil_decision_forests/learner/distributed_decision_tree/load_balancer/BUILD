load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Proto
# ========

all_proto_library(
    name = "load_balancer_proto",
    srcs = ["load_balancer.proto"],
)

# Libraries
# =========================================================

cc_library_ydf(
    name = "load_balancer",
    srcs = [
        "load_balancer.cc",
    ],
    hdrs = ["load_balancer.h"],
    deps = [
        ":load_balancer_cc_proto",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_cc_proto",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

# Test
# =========================================================

cc_test(
    name = "load_balancer_test",
    srcs = ["load_balancer_test.cc"],
    deps = [
        ":load_balancer",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
        "@com_google_googletest//:gtest_main",
    ],
)
