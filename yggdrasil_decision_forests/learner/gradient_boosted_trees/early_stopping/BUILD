load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Library
# =======

cc_library_ydf(
    name = "early_stopping",
    srcs = ["early_stopping.cc"],
    hdrs = ["early_stopping.h"],
    deps = [
        ":early_stopping_snapshot_cc_proto",
        "//yggdrasil_decision_forests/utils:logging",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

# Proto
# =======

all_proto_library(
    name = "early_stopping_snapshot_proto",
    srcs = ["early_stopping_snapshot.proto"],
)

# Test
# =======
cc_test(
    name = "early_stopping_test",
    size = "small",
    srcs = ["early_stopping_test.cc"],
    deps = [
        ":early_stopping",
        ":early_stopping_snapshot_cc_proto",
        "//yggdrasil_decision_forests/utils:test",
        "@com_google_googletest//:gtest_main",
    ],
)
