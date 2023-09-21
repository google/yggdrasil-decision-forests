"""Pybind project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    # Version 2.11.1
    PYBIND_BAZEL_COMMIT_HASH = "fd7d88857cca3d7435b06f3ac6abab77cd9983b2"
    PYBIND_BAZEL_SHA = "34bc7959304c22ca7b37be06c6078b6be1ffd12e683961aadb1c18d28d7d9d5f"
    PYBIND_COMMIT_HASH = "8a099e44b3d5f85b20f05828d919d2332a8de841"
    PYBIND_SHA = "e7fc4519e2c59737d38751fab8de2192865a06dac4f45231b96f9628e62da5a6"
    http_archive(
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-{commit}".format(commit = PYBIND_BAZEL_COMMIT_HASH),
        urls = ["https://github.com/pybind/pybind11_bazel/archive/{commit}.tar.gz".format(commit = PYBIND_BAZEL_COMMIT_HASH)],
        sha256 = PYBIND_BAZEL_SHA,
    )

    http_archive(
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        strip_prefix = "pybind11-{commit}".format(commit = PYBIND_COMMIT_HASH),
        urls = ["https://github.com/pybind/pybind11/archive/{commit}.tar.gz".format(commit = PYBIND_COMMIT_HASH)],
        sha256 = PYBIND_SHA,
    )
