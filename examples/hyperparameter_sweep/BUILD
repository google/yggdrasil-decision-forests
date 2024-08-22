load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Dependences of the workers.
cc_library(
    name = "worker_dep",
    deps = [
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
    ],
)

# The worker binary.
cc_binary(
    name = "worker_main",
    deps = [
        ":worker",
        "//yggdrasil_decision_forests/utils/distribute/implementations/grpc:grpc_worker_lib_with_main",
    ],
)

# The worker logic.
cc_library(
    name = "worker",
    srcs = ["worker.cc"],
    deps = [
        ":optimizer_cc_proto",
        ":worker_dep",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner:learner_library",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:protobuf",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils/distribute:core",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

# The main / chief / manager binary.
cc_binary(
    name = "manager_main",
    srcs = ["manager_main.cc"],
    deps = [
        ":optimizer_cc_proto",
        ":worker",  # Only for in process operation.
        ":worker_dep",  # Only for in process operation.
        "//yggdrasil_decision_forests/cli:all_file_systems",
        "//yggdrasil_decision_forests/dataset:all_dataset_formats",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/gradient_boosted_trees:gradient_boosted_trees_cc_proto",
        "//yggdrasil_decision_forests/metric",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:protobuf",
        "//yggdrasil_decision_forests/utils/distribute:distribute_cc_proto",
        "//yggdrasil_decision_forests/utils/distribute:distribute_without_implementations",
        "//yggdrasil_decision_forests/utils/distribute/implementations/grpc",
        "//yggdrasil_decision_forests/utils/distribute/implementations/multi_thread",  # Only for in process operation.
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

all_proto_library(
    name = "optimizer_proto",
    srcs = ["optimizer.proto"],
    deps = [
        "//yggdrasil_decision_forests/dataset:data_spec_proto",
        "//yggdrasil_decision_forests/learner:abstract_learner_proto",
        "//yggdrasil_decision_forests/metric:metric_proto",
    ],
)
