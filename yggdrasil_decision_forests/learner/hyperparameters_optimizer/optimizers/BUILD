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
        "//yggdrasil_decision_forests/learner/hyperparameters_optimizer:optimizer_interface",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:protobuf",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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
        "//yggdrasil_decision_forests/learner/cart",
        "//yggdrasil_decision_forests/learner/hyperparameters_optimizer:hyperparameters_optimizer_cc_proto",
        "//yggdrasil_decision_forests/learner/hyperparameters_optimizer:optimizer_interface",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:test",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
    ],
)
