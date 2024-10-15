# ts_library

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "learner",
    srcs = ["learner.cc"],
    hdrs = ["learner.h"],
    tags = ["manual"],
    deps = [
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/learner:abstract_learner",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/model:abstract_model",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/port/javascript/training/dataset",
        "//yggdrasil_decision_forests/port/javascript/training/model",
        "//yggdrasil_decision_forests/port/javascript/training/util:status_casters",
        "//yggdrasil_decision_forests/utils:logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
