"""TensorFlow project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    TENSORFLOW_VERSION = "2.13.0"
    TENSORFLOW_SHA = "447cdb65c80c86d6c6cf1388684f157612392723eaea832e6392d219098b49de"
    http_archive(
        name = "org_tensorflow",
        sha256 = TENSORFLOW_SHA,
        strip_prefix = "tensorflow-{version}".format(version = TENSORFLOW_VERSION),
        urls = ["https://github.com/tensorflow/tensorflow/archive/refs/tags/v{version}.zip".format(version = TENSORFLOW_VERSION)],
    )
