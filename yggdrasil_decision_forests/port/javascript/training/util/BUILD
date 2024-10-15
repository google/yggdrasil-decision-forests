# ts_library

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "status_casters",
    srcs = ["status_casters.cc"],
    hdrs = ["status_casters.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    tags = ["manual"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
