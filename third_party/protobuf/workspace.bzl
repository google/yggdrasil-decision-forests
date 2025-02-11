"""Protobuf project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    PROTOBUF_VERSION = "29.1"
    PROTOBUF_SHA = "3d32940e975c4ad9b8ba69640e78f5527075bae33ca2890275bf26b853c0962c"

    http_archive(
        name = "com_google_protobuf",
        urls = ["https://github.com/protocolbuffers/protobuf/archive/v{version}.tar.gz".format(version = PROTOBUF_VERSION)],
        strip_prefix = "protobuf-{version}".format(version = PROTOBUF_VERSION),
        sha256 = PROTOBUF_SHA,
    )
