"""Pybind Protobuf project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    PYBIND_PROTOBUF_COMMIT_HASH = "f1b245929759230f31cdd1e5f9e0e69f817fed95"
    PYBIND_PROTOBUF_SHA = "7eeabdaa39d5b1f48f1feb0894d6b7f02f77964e2a6bc1eaa4a90fe243e0a34c"
    http_archive(
        name = "com_google_pybind11_protobuf",
        strip_prefix = "pybind11_protobuf-{commit}".format(commit = PYBIND_PROTOBUF_COMMIT_HASH),
        urls = [
            "https://github.com/pybind/pybind11_protobuf/archive/{commit}.tar.gz".format(commit = PYBIND_PROTOBUF_COMMIT_HASH),
        ],
        sha256 = PYBIND_PROTOBUF_SHA,
    )
