"""Google Test project."""

#load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    # An old version of GRPC is injected by TensorFlow itself.
    # For compatibility reason with TF-DF, we use the TensorFlow version.

    #http_archive(
    #    name = "com_github_grpc_grpc",
    #    urls = ["https://github.com/grpc/grpc/archive/master.zip"],
    #    strip_prefix = "grpc-master",
    #)
    pass
