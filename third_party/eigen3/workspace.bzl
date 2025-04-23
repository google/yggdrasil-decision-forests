"""Eigen project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    http_archive(
        name = "eigen_archive",
        # This is a mirror for https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        # which has had an unstable sha, see https://gitlab.com/libeigen/eigen/-/issues/2919
        # and https://github.com/bazelbuild/bazel-central-registry/pull/4364.
        urls = [
            "https://github.com/eigen-mirror/eigen/archive/refs/tags/3.4.0.tar.gz",
        ],
        strip_prefix = "eigen-3.4.0",
        sha256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
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
