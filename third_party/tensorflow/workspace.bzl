"""TensorFlow project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    http_archive(
        name = "org_tensorflow",
        # 2.4.1
        #sha256 = "b91ec194ddf6c4a5a2f9d1db4af4daab0b187ff691e6f88142413d2c7e77a3bb",
        #strip_prefix = "tensorflow-2.4.1",
        #urls = ["https://github.com/tensorflow/tensorflow/archive/v2.4.1.zip"],
        # v2.5.0-rcx
        sha256 = "f7ad0a488559ee01f042a967c065482b074e11afd0299facbd8dc0cba9ae3fa9",
        strip_prefix = "tensorflow-2.5.0-rc3",
        urls = ["https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.5.0-rc3.zip"],
        # head
        #urls = ["https://github.com/tensorflow/tensorflow/archive/master.zip"],
        #strip_prefix = "tensorflow-master",
    )
