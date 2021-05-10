"""Protobuf project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    http_archive(
        name = "com_google_protobuf",
        #strip_prefix = "protobuf-master",
        #urls = ["https://github.com/protocolbuffers/protobuf/archive/master.zip"],
        urls = [" https://github.com/protocolbuffers/protobuf/archive/v3.14.0.zip"],
        strip_prefix = "protobuf-3.14.0",
        sha256 = "bf0e5070b4b99240183b29df78155eee335885e53a8af8683964579c214ad301",
    )
