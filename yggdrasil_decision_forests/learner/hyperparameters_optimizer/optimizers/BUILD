load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Library
# =======

cc_library_ydf(
    name = "random",
    srcs = ["random.cc"],
    hdrs = ["random.h"],
    deps = [
        ":random_cc_proto",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/status",
        "//yggdrasil_decision_forests/learner/hyperparameters_optimizer:optimizer_interface",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:status_macros",
    ],
    alwayslink = 1,
)

# Proto
# ========

all_proto_library(
    name = "random_proto",
    srcs = ["random.proto"],
    deps = [
        "//yggdrasil_decision_forests/learner/hyperparameters_optimizer:hyperparameters_optimizer_proto",
    ],
)

# Test
# ========

cc_test(
    name = "random_test",
    srcs = ["random_test.cc"],
    deps = [
        ":random",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/learner/cart",
        "//yggdrasil_decision_forests/learner/hyperparameters_optimizer:hyperparameters_optimizer_cc_proto",
        "//yggdrasil_decision_forests/learner/hyperparameters_optimizer:optimizer_interface",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:test_utils",
    ],
)
