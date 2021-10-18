load("//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========================================================

cc_library_ydf(
    name = "column_cache",
    srcs = ["column_cache.cc"],
    hdrs = ["column_cache.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils/distribute:core",
    ],
)

# Test
# =========================================================

cc_test(
    name = "column_cache_test",
    srcs = ["column_cache_test.cc"],
    deps = [
        ":column_cache",
        "@com_google_googletest//:gtest_main",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:test",
    ],
)
