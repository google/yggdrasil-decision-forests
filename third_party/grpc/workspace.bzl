"""Google Test project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    GRPC_VERSION = "1.70.1"
    GRPC_SHA = "c4e85806a3a23fd2a78a9f8505771ff60b2beef38305167d50f5e8151728e426"
    http_archive(
        name = "com_github_grpc_grpc",
        urls = ["https://github.com/grpc/grpc/archive/refs/tags/v{version}.tar.gz".format(version = GRPC_VERSION)],
        strip_prefix = "grpc-{version}".format(version = GRPC_VERSION),
        sha256 = GRPC_SHA,
    )
