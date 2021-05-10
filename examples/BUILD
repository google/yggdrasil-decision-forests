package(
    default_visibility = ["//third_party/tensorflow_decision_forests:users"],
    licenses = ["notice"],
)

cc_binary(
    name = "beginner_cc",
    srcs = ["beginner.cc"],
    # Because this binary is defined in the Yggdrasil project directly, the
    # following dependencies are local. In your project you will probably use
    # a "http_archive" dependency:
    # https://docs.bazel.build/versions/master/repo/http.html#http_archive*
    #
    # See TensorFlow Decision Forests for an example (TF-DF imports YDF):
    #   https://github.com/tensorflow/decision-forests
    deps = [
        "//learning/lib/ami/simple_ml/dataset:vertical_dataset_io",
        "@com_google_absl//absl/flags:flag",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/learner:all_learners",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/metric:report",
        "//yggdrasil_decision_forests/model:model_library",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
    ],
)
