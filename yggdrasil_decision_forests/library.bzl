# Third party libraries
load("//third_party/gtest:workspace.bzl", gtest = "deps")
load("//third_party/absl:workspace.bzl", absl = "deps")
load("//third_party/protobuf:workspace.bzl", protobuf = "deps")
load("//third_party/zlib:workspace.bzl", zlib = "deps")
load("//third_party/tensorflow:workspace.bzl", tensorflow = "deps")
load("//third_party/farmhash:workspace.bzl", farmhash = "deps")
load("//third_party/boost:workspace.bzl", boost = "deps")
load("//third_party/grpc:workspace.bzl", grpc = "deps")
load("//third_party/rapidjson:workspace.bzl", rapidjson = "deps")
load("//third_party/eigen3:workspace.bzl", eigen = "deps")

def load_dependencies(repo_name = "", exclude_repo = []):
    if "gtest" not in exclude_repo:
        gtest()

    if "absl" not in exclude_repo:
        absl()

    if "protobuf" not in exclude_repo:
        protobuf()

    if "zlib" not in exclude_repo:
        zlib()

    if "tensorflow" not in exclude_repo:
        tensorflow()

    if "farmhash" not in exclude_repo:
        farmhash(prefix = repo_name)

    if "boost" not in exclude_repo:
        boost(prefix = repo_name)

    if "grpc" not in exclude_repo:
        grpc()

    if "rapidjson" not in exclude_repo:
        rapidjson(prefix = repo_name)

    if "eigen" not in exclude_repo:
        eigen()

# Placeholder to use until bazel supports py_strict_binary.
def py_strict_binary(name, **kwargs):
    native.py_binary(name = name, **kwargs)

# Placeholder to use until bazel supports py_strict_library.
def py_strict_library(name, **kwargs):
    native.py_library(name = name, **kwargs)

# Placeholder to use until bazel supports pytype_strict_binary.
def pytype_strict_binary(name, **kwargs):
    native.py_binary(name = name, **kwargs)

# Placeholder to use until bazel supports pytype_strict_library.
def pytype_strict_library(name, **kwargs):
    native.py_library(name = name, **kwargs)

# Placeholder to use until bazel supports pytype_library.
def pytype_library(name, **kwargs):
    native.py_library(name = name, **kwargs)
