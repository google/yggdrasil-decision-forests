"""TensorFlow project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    http_archive(
        name = "org_tensorflow",
        sha256 = "e3d0ee227cc19bd0fa34a4539c8a540b40f937e561b4580d4bbb7f0e31c6a713",
        strip_prefix = "tensorflow-2.5.0",
        urls = ["https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.5.0.zip"],
    )
