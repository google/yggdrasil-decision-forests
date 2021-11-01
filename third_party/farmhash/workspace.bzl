"""TensorFlow project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps(prefix = ""):
    http_archive(
        # The name should match TF's name for farmhash lib.
        name = "farmhash_archive",
        build_file = prefix + "//third_party/farmhash:farmhash.BUILD",
        strip_prefix = "farmhash-master",
        # Does not have any release.
        urls = ["https://github.com/google/farmhash/archive/master.zip"],
        sha256 = "d27a245e59f5e10fba10b88cb72c5f0e03d3f2936abbaf0cb78eeec686523ec1",
    )
