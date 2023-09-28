"""Google Test project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    GRPC_VERSION = "1.58.1"
    GRPC_SHA = "860bf758a1437a03318bf09db8e87cb8149a2f578954110ce8549e147f868b62"
    http_archive(
        name = "com_github_grpc_grpc",
        urls = ["https://github.com/grpc/grpc/archive/refs/tags/v{version}.tar.gz".format(version = GRPC_VERSION)],
        strip_prefix = "grpc-{version}".format(version = GRPC_VERSION),
        sha256 = GRPC_SHA,
    )
