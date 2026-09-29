load("@com_google_protobuf//bazel/toolchains:proto_lang_toolchain.bzl", "proto_lang_toolchain")
load("@rules_python//python/uv:lock.bzl", "lock")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Universal lock file of the third-party Python packages, for all the supported
# Python versions and platforms.
#
# Update with:
#   bazel run //:requirements.update
lock(
    name = "requirements",
    srcs = [
        "dev_requirements.txt",
        "requirements.txt",
    ],
    out = "requirements_lock.txt",
    args = [
        "--universal",
        "--python-version=3.10",
    ],
)

# Runtime of `py_proto_library`: The protobuf package from PyPI, i.e. the same
# as used by the users of the ydf package.
proto_lang_toolchain(
    name = "python_proto",
    command_line = "--python_out=%s",
    progress_message = "Generating Python proto_library %{label}",
    runtime = "@pypi_ydf//protobuf",
    toolchain_type = "@com_google_protobuf//bazel/private:python_toolchain_type",
)
