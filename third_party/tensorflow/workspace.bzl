"""TensorFlow project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    http_archive(
        name = "org_tensorflow",
        sha256 = "9f2dac244e5af6c6a13a7dad6481e390174ac989931942098e7a4373f1bccfc2",
        strip_prefix = "tensorflow-2.9.1",
        urls = ["https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.9.1.zip"],
    )
