"""Absl project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    VERSION = "20240722.1"
    SHA = "40cee67604060a7c8794d931538cb55f4d444073e556980c88b6c49bb9b19bb7"

    http_archive(
        name = "com_google_absl",
        urls = ["https://github.com/abseil/abseil-cpp/releases/download/{version}/abseil-cpp-{version}.tar.gz".format(version = VERSION)],
        strip_prefix = "abseil-cpp-{version}".format(version = VERSION),
        sha256 = SHA,
    )
