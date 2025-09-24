load("//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library_ydf(
    name = "java_embed",
    srcs = ["java_embed.cc"],
    hdrs = ["java_embed.h"],
    deps = [
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model/decision_tree:decision_forest_interface",
        "//yggdrasil_decision_forests/serving/embed:common",
        "//yggdrasil_decision_forests/serving/embed:utils",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
