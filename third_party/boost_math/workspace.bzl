"""Boost math project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    BOOST_MATH_VERSION = "1.83.0"
    BOOST_MATH_SHA = "53e5f7539a66899fe0fca3080405cbd5f7959da5394ec13664746741aece1705"
    build_file_content = """
package(
  default_visibility = ["//visibility:public"],
  licenses = ["notice"],
)

cc_library(
  name = "boost_math",
  srcs = glob(["include/**/*.hpp", "include/**/*.h"]),
  includes = glob(["include"],exclude_directories=0),
)
  """
    http_archive(
        name = "org_boost_math",
        urls = ["https://github.com/boostorg/math/archive/refs/tags/boost-{version}.tar.gz".format(version = BOOST_MATH_VERSION)],
        strip_prefix = "math-boost-{version}".format(version = BOOST_MATH_VERSION),
        sha256 = BOOST_MATH_SHA,
        build_file_content = build_file_content,
    )
