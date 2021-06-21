load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Proto
# ========

all_proto_library(
    name = "single_thread_proto",
    srcs = ["single_thread.proto"],
    deps = ["//yggdrasil_decision_forests/utils/distribute:distribute_proto"],
)

# Library
# =======

cc_library(
    name = "single_thread",
    srcs = ["single_thread.cc"],
    hdrs = ["single_thread.h"],
    deps = [
        ":single_thread_cc_proto",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils/distribute:core",
        "//yggdrasil_decision_forests/utils/distribute:distribute_cc_proto",
    ],
    alwayslink = 1,
)
