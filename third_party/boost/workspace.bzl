"""Boost project."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def deps(prefix = ""):
    branch = "boost-1.75.0"
    # branch = "master"
    # branch = "develop"

    build_file_content = """
package(
  default_visibility = ["//visibility:public"],
  licenses = ["notice"],
)

cc_library(
  name = "boost",
  srcs = glob(["libs/*/include/**/*.hpp", "libs/*/include/**/*.h", "libs/*/*/include/**/*.hpp", "libs/*/*/include/**/*.h"]),
  includes = glob(["libs/*/include", "libs/*/*/include"],exclude_directories=0),
)
  """

    new_git_repository(
        name = "org_boost",
        branch = branch,
        build_file_content = build_file_content,
        init_submodules = True,
        recursive_init_submodules = True,
        remote = "https://github.com/boostorg/boost",
    )

    # Bazel has a 10 minutes timeout on most commands. If getting Boost
    # times-out (boost is composed of many sub repositories), do the following:
    #
    # PS: Make sure this is not your anti-virus that is slowing things down.
    #
    # 1. Create a new directory in your homespace (e.g. "dependencies").
    # 2. Open a shell in this directory and run:
    #    git clone --branch boost-1.75.0 https://github.com/boostorg/boost.git
    #    cd boost
    #    git submodule update --init --checkout --force
    # 3. Comment the "new_git_repository" rule above, and uncomment the
    #    "new_local_repository" rule below.

    # native.new_local_repository(
    #   name = "org_boost",
    #   path = "../boost",
    #   build_file_content = build_file_content,
    #  )
