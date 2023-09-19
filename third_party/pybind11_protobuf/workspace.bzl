"""Pybind Protobuf project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# TODO: Update to latest commit.
def deps():
    http_archive(
        name = "com_google_pybind11_protobuf",
        sha256 = "2e1fa89c6afa23f65f5ff9835d195142a8a059bcdb96eaa3c367a195cb183f26",
        strip_prefix = "pybind11_protobuf-10fc4b0affe36b98b5e291008ef59f36637a1f1b",
        urls = [
            "https://github.com/pybind/pybind11_protobuf/archive/10fc4b0affe36b98b5e291008ef59f36637a1f1b.tar.gz",
        ],
    )
