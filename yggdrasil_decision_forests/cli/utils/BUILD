load("//yggdrasil_decision_forests/utils:compile.bzl", "cc_binary_ydf", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Binaries
# ========

cc_binary_ydf(
    name = "synthetic_dataset",
    deps = [":synthetic_dataset_lib_with_main"],
)

cc_library_ydf(
    name = "synthetic_dataset_lib_with_main",
    srcs = ["synthetic_dataset.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "//yggdrasil_decision_forests/cli:all_file_systems",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:synthetic_dataset",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
    ],
)
