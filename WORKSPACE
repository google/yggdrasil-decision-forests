workspace(name = "yggdrasil_decision_forests")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Load third party libraries
load("//third_party/gtest:workspace.bzl", gtest = "deps")
load("//third_party/absl:workspace.bzl", absl = "deps")
load("//third_party/protobuf:workspace.bzl", protobuf = "deps")
load("//third_party/zlib:workspace.bzl", zlib = "deps")
load("//third_party/tensorflow:workspace.bzl", tensorflow = "deps")
load("//third_party/farmhash:workspace.bzl", farmhash = "deps")
load("//third_party/boost:workspace.bzl", boost = "deps")
load("//third_party/grpc:workspace.bzl", grpc = "deps")

gtest()
absl()
protobuf()
zlib()
tensorflow()
farmhash()
boost()
grpc()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

# Injected by TensoFlow
#load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
#grpc_deps()
#load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
#grpc_extra_deps()

# TensorFlow is used to read and write TFRecord and IO if
# use_tensorflow_io=1. Only a small fraction of TF will be compiled.
load("@org_tensorflow//tensorflow:workspace3.bzl", tf1="workspace")
tf1()
load("@org_tensorflow//tensorflow:workspace2.bzl", tf2="workspace")
tf2()
load("@org_tensorflow//tensorflow:workspace1.bzl", tf3="workspace")
tf3()
load("@org_tensorflow//tensorflow:workspace0.bzl", tf4="workspace")
tf4()

# Temporary solution to the "no such target '@platforms//cpu:wasm32'" error in
# Absl.
# TODO(gbm): Replace with a "bazel_skylib" rule when supported (see
# https://github.com/abseil/abseil-cpp/commit/eb3db08cb3a4faf2aa09a2ba4a30b442457f36cf).
http_archive(
        name = "platforms",
        sha256 = "b601beaf841244de5c5a50d2b2eddd34839788000fa1be4260ce6603ca0d8eb7",
        strip_prefix = "platforms-98939346da932eef0b54cf808622f5bb0928f00b",
        urls = ["https://github.com/bazelbuild/platforms/archive/98939346da932eef0b54cf808622f5bb0928f00b.zip"],
    )