"""Utilities for the compilation of code."""

load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

def cc_library_ydf(**attrs):
    native.cc_library(**attrs)

def cc_binary_ydf(**attrs):
    native.cc_binary(**attrs)

def all_proto_library(
        name = None,
        deps = [],
        option_deps = [],
        srcs = [],
        compile_cc = True,
        compile_py = True,
        visibility = None,
        has_services = False,
        exports = None):
    """Create the set of proto, cc proto and py proto targets.

    Usage example:
        all_proto_library(name="toy_proto",srcs=[...])

        cc_library_ydf(deps=[":toy_cc_proto"], ...)
        py_library(deps=[":toy_py_proto"], ...)

    Args:
      name: Name of the proto rule. Should end with "_proto".
      deps: Dependencies of the proto rule.
      option_deps: Option dependencies of the proto rule.
      srcs: Sources of the proto rule.
      compile_cc: If true, generate a cc proto rule.
      compile_py: If true, generate a py proto rule.
      visibility: Visibility of the rules.
      has_services: The proto has a grpc service.
      exports: List of proto_library targets that can be referenced via "import public".
    """

    suffix = "_proto"
    if not name.endswith(suffix):
        fail("Rule name should ends with _proto")
    base_name = name[0:-len(suffix)]

    proto_library(
        name = name,
        srcs = srcs,
        deps = deps,
        visibility = visibility,
    )

    if has_services:
        cc_grpc_library(
            name = base_name + "_grpc_proto",
            srcs = [":" + name],
            deps = [base_name + "_cc_proto"],
            visibility = visibility,
            grpc_only = True,
        )

    if compile_cc:
        cc_proto_library(
            name = base_name + "_cc_proto",
            deps = [":" + name],
            visibility = visibility,
        )

    if compile_py:
        py_proto_library(
            name = base_name + "_py_proto",
            deps = [":" + name],
            visibility = visibility,
        )

register_extension_info(
    extension = all_proto_library,
    label_regex_map = {
        "deps": "deps:{extension_name}",
        "option_deps": "option_deps:{extension_name}",
    },
)
