load("@rules_python//python:py_test.bzl", "py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Checks the content of the wheel (see //:wheel), imports it (without the
# sources in the runfiles) and, on Linux, checks its manylinux tag with
# auditwheel.
py_test(
    name = "wheel_test",
    srcs = ["wheel_test.py"],
    args = ["--wheel=$(location //:wheel)"],
    data = ["//:wheel"],
    deps = [
        "@abseil-py//absl/flags",
        "@abseil-py//absl/testing:absltest",
        "@pypi_ydf//numpy",
        "@pypi_ydf//protobuf",
    ] + select({
        "@platforms//os:linux": ["@pypi_ydf//auditwheel"],
        "//conditions:default": [],
    }),
)
