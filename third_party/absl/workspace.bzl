"""Absl project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    # Note: If the following import is commented, the version of Absl injected
    # by TensorFlow will be used to compile Yggdrasil. TensorFlow uses an old
    # version of Absl. In this case, you will have to use the following defines:
    # --define=no_absl_statusor=1

    http_archive(
        name = "com_google_absl",
        urls = ["https://github.com/abseil/abseil-cpp/archive/master.zip"],
        strip_prefix = "abseil-cpp-master",
    )
