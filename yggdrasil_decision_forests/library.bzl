"""Inject YDF dependencies."""

load("//third_party/absl:workspace.bzl", absl = "deps")
load("//third_party/boost_math:workspace.bzl", boost_math = "deps")
load("//third_party/eigen3:workspace.bzl", eigen = "deps")
load("//third_party/farmhash:workspace.bzl", farmhash = "deps")
load("//third_party/grpc:workspace.bzl", grpc = "deps")

# Third party libraries
load("//third_party/gtest:workspace.bzl", gtest = "deps")
load("//third_party/nlohmann_json:workspace.bzl", nlohmann_json = "deps")
load("//third_party/protobuf:workspace.bzl", protobuf = "deps")
load("//third_party/zlib:workspace.bzl", zlib = "deps")

def load_dependencies(repo_name = "", exclude_repo = []):
    if "gtest" not in exclude_repo:
        gtest()

    if "absl" not in exclude_repo:
        absl()

    if "protobuf" not in exclude_repo:
        protobuf()

    if "zlib" not in exclude_repo:
        zlib()

    if "farmhash" not in exclude_repo:
        farmhash(prefix = repo_name)

    if "boost_math" not in exclude_repo:
        boost_math()

    if "grpc" not in exclude_repo:
        grpc()

    if "eigen" not in exclude_repo:
        eigen()

    if "nlohmann_json" not in exclude_repo:
        nlohmann_json()
