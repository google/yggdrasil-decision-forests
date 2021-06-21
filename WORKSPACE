workspace(name = "yggdrasil_decision_forests")

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
