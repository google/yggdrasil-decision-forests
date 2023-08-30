"""Eigen project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    http_archive(
        name = "eigen_archive",
        urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"],
        strip_prefix = "eigen-3.4.0",
        sha256 = "1ccaabbfe870f60af3d6a519c53e09f3dcf630207321dffa553564a8e75c4fc8",
        build_file_content =
            """
cc_library(
    name = 'eigen3_internal',
    srcs = [],
    includes = ['.'],
    hdrs = glob(['Eigen/**']),
    visibility = ['//visibility:public'],
)
alias(
    name = "eigen3",
    actual = "@eigen_archive//:eigen3_internal",
    visibility = ["//visibility:public"],
)
""",
    )
