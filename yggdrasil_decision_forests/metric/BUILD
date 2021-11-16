load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========================================================

cc_library_ydf(
    name = "metric",
    srcs = [
        "metric.cc",
        "uplift.cc",
    ],
    hdrs = [
        "labels.h",
        "metric.h",
        "uplift.h",
    ],
    deps = [
        ":metric_cc_proto",
        ":ranking",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@org_boost//:boost",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:distribution",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:protobuf",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:status_macros",
    ] + select({
        "//conditions:default": [
        ],
    }),
)

cc_library_ydf(
    name = "ranking_utils",
    hdrs = [
        "ranking_utils.h",
    ],
)

cc_library_ydf(
    name = "ranking",
    srcs = [
        "ranking_mrr.cc",
        "ranking_ndcg.cc",
    ],
    hdrs = [
        "ranking_mrr.h",
        "ranking_ndcg.h",
    ],
    deps = [
        ":ranking_utils",
        "//yggdrasil_decision_forests/utils:logging",
    ],
)

cc_library_ydf(
    name = "comparison",
    srcs = [
        "comparison.cc",
    ],
    hdrs = [
        "comparison.h",
        "labels.h",
    ],
    deps = [
        ":metric",
        ":metric_cc_proto",
        ":ranking",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@org_boost//:boost",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model:prediction_cc_proto",
        "//yggdrasil_decision_forests/utils:logging",
    ],
)

cc_library_ydf(
    name = "report",
    srcs = [
        "report.cc",
    ],
    hdrs = [
        "report.h",
    ],
    deps = [
        ":metric",
        ":metric_cc_proto",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:distribution",
    ],
)

# Proto
# =========================================================

all_proto_library(
    name = "metric_proto",
    srcs = ["metric.proto"],
    deps = [
        "//yggdrasil_decision_forests/dataset:data_spec_proto",
        "//yggdrasil_decision_forests/dataset:weight_proto",
        "//yggdrasil_decision_forests/model:abstract_model_proto",
        "//yggdrasil_decision_forests/model:prediction_proto",
        "//yggdrasil_decision_forests/utils:distribution_proto",
    ],
)

# Unit Tests
# =========================================================

cc_test(
    name = "metric_test",
    size = "large",
    srcs = ["metric_test.cc"],
    shard_count = 4,
    deps = [
        ":comparison",
        ":metric",
        ":metric_cc_proto",
        ":report",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:distribution",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:test",
    ],
)

cc_test(
    name = "comparison_test",
    size = "large",
    srcs = ["comparison_test.cc"],
    shard_count = 4,
    deps = [
        ":comparison",
        ":metric",
        ":metric_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@org_boost//:boost",
        "//yggdrasil_decision_forests/utils:random",
    ],
)

cc_test(
    name = "uplift_test",
    srcs = ["uplift_test.cc"],
    deps = [
        ":comparison",
        ":metric",
        ":metric_cc_proto",
        ":report",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:distribution",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:test",
    ],
)
