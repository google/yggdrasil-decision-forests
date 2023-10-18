load("//yggdrasil_decision_forests/utils:compile.bzl", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library_ydf(
    name = "serving",
    hdrs = ["serving.h"],
    deps = [
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:all_models",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/serving:example_set",
        "//yggdrasil_decision_forests/serving:fast_engine",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
