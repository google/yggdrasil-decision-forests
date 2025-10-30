"""Absl project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    VERSION = "20250814.1"
    SHA = "1692f77d1739bacf3f94337188b78583cf09bab7e420d2dc6c5605a4f86785a1"

    http_archive(
        name = "com_google_absl",
        urls = ["https://github.com/abseil/abseil-cpp/releases/download/{version}/abseil-cpp-{version}.tar.gz".format(version = VERSION)],
        strip_prefix = "abseil-cpp-{version}".format(version = VERSION),
        sha256 = SHA,
    )
