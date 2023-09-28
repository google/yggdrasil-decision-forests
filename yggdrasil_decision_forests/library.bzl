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
load("//third_party/pybind11:workspace.bzl", pybind11 = "deps")
load("//third_party/pybind11_abseil:workspace.bzl", pybind11_abseil = "deps")
load("//third_party/pybind11_protobuf:workspace.bzl", pybind11_protobuf = "deps")

def load_dependencies(repo_name = "", exclude_repo = []):
    if "pybind11" not in exclude_repo:
        pybind11()

    if "pybind11_abseil" not in exclude_repo:
        pybind11_abseil()

    if "pybind11_protobuf" not in exclude_repo:
        pybind11_protobuf()

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
