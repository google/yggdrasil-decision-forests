"""Google Test project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    GOOGLETEST_VERSION = "1.15.2"
    GOOGLETEST_SHA = "7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926"
    http_archive(
        name = "com_google_googletest",
        urls = ["https://github.com/google/googletest/archive/refs/tags/v{version}.tar.gz".format(version = GOOGLETEST_VERSION)],
        strip_prefix = "googletest-{version}".format(version = GOOGLETEST_VERSION),
        sha256 = GOOGLETEST_SHA,
    )
