"""Pybind Protobuf project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    PYBIND_PROTOBUF_COMMIT_HASH = "3d7834b607758bbd2e3d210c6c478453922f20c0"
    PYBIND_PROTOBUF_SHA = "89ba0a6eb92a834dc08dc199da5b94b4648168c56d5409116f9b7699e5350f11"
    http_archive(
        name = "com_google_pybind11_protobuf",
        strip_prefix = "pybind11_protobuf-{commit}".format(commit = PYBIND_PROTOBUF_COMMIT_HASH),
        urls = [
            "https://github.com/pybind/pybind11_protobuf/archive/{commit}.tar.gz".format(commit = PYBIND_PROTOBUF_COMMIT_HASH),
        ],
        sha256 = PYBIND_PROTOBUF_SHA,
    )
