"""Pybind project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    # Version 2.11.1
    PYBIND_BAZEL_COMMIT_HASH = "fd7d88857cca3d7435b06f3ac6abab77cd9983b2"
    PYBIND_COMMIT_HASH = "8a099e44b3d5f85b20f05828d919d2332a8de841"
    http_archive(
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-{commit}".format(commit = PYBIND_BAZEL_COMMIT_HASH),
        urls = ["https://github.com/pybind/pybind11_bazel/archive/{commit}.tar.gz".format(commit = PYBIND_BAZEL_COMMIT_HASH)],
    )

    http_archive(
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        strip_prefix = "pybind11_bazel-{commit}".format(commit = PYBIND_COMMIT_HASH),
        urls = ["https://github.com/pybind/pybind11/archive/{commit}.tar.gz".format(commit = PYBIND_COMMIT_HASH)],
    )
