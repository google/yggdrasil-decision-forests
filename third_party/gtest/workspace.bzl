"""Google Test project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    # Version 1.14.0, 2023-08-02
    GOOGLETEST_VERSION = "1.14.0"
    GOOGLETEST_SHA = "8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7"
    http_archive(
        name = "com_google_googletest",
        urls = ["https://github.com/google/googletest/archive/refs/tags/v{version}.tar.gz".format(version = GOOGLETEST_VERSION)],
        strip_prefix = "googletest-{version}".format(version = GOOGLETEST_VERSION),
        sha256 = GOOGLETEST_SHA,
    )
