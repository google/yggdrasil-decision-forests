"""Absl project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    # Note: If the following import is commented, the version of Absl injected
    # by TensorFlow will be used to compile Yggdrasil. TensorFlow uses an old
    # version of Absl.

    http_archive(
        name = "com_google_absl",
        #  Abseil LTS branch, Aug 2023
        urls = ["https://github.com/abseil/abseil-cpp/archive/29bf8085f3bf17b84d30e34b3d7ff8248fda404e.zip"],
        strip_prefix = "abseil-cpp-29bf8085f3bf17b84d30e34b3d7ff8248fda404e",
    )
