"""Google Test project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    http_archive(
        name = "com_github_grpc_grpc",
        urls = ["https://github.com/grpc/grpc/archive/refs/tags/v1.50.0.zip"],
        strip_prefix = "grpc-1.50.0",
        sha256 = "01f66f8349f3fe1f5c07f992206a101090f9ecb81dca355ff254fbf09e1c62a5",
    )
