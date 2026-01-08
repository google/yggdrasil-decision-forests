"""Highway project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    http_archive(
        name = "com_google_highway",
        urls = ["https://github.com/google/highway/releases/download/1.3.0/highway-1.3.0.tar.gz"],
        strip_prefix = "highway-1.3.0",
        sha256 = "e8d696900b45f4123be8a9d6866f4e7b6831bf599f4b9c178964d968e6a58a69",
    )
