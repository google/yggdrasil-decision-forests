"""Pybind absl wrappers project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    PYBIND_ABSL_COMMIT_HASH = "fcfff8502fad281b0c1197872a1e30cdab69a323"
    http_archive(
        name = "com_google_pybind11_abseil",
        urls = ["https://github.com/pybind/pybind11_abseil/archive/{commit}.tar.gz".format(commit = PYBIND_ABSL_COMMIT_HASH)],
        strip_prefix = "pybind11_abseil-{commit}".format(commit = PYBIND_ABSL_COMMIT_HASH),
    )
