"""Protobuf project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    PROTOBUF_VERSION = "24.3"
    PROTOBUF_SHA = "07d69502e58248927b58c7d7e7424135272ba5b2852a753ab6b67e62d2d29355"

    http_archive(
        name = "com_google_protobuf",
        urls = [" https://github.com/protocolbuffers/protobuf/archive/v{version}.tar.gz".format(version = PROTOBUF_VERSION)],
        strip_prefix = "protobuf-{version}".format(version = PROTOBUF_VERSION),
        sha256 = PROTOBUF_SHA,
    )
