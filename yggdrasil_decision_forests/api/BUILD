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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library_ydf(
    name = "training",
    hdrs = ["training.h"],
    deps = [
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:example_builder",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner",
        "//yggdrasil_decision_forests/learner:all_learners",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/learner/cart:cart_cc_proto",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/learner/random_forest:random_forest_cc_proto",
        "//yggdrasil_decision_forests/model:all_models",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
    ],
)
