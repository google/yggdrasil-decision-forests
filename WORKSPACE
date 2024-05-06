# Bazel workspace configuration for Yggdrasil Decision Forest
#
# For support with TensorFlow file formats or TensorFlow IO, replace this file
# with WORKSPACE_WITH_TF, and use the "--config=use_tensorflow_io" flag. See
# tools/test_bazel_with_tf.sh for more details.
#
#
# WARNING: This file is a copy of WORKSPACE_NO_TF. Keep both files in sync.
#

workspace(name = "yggdrasil_decision_forests")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Load the dependencies of YDF.
load("//yggdrasil_decision_forests:library.bzl", ydf_load_deps = "load_dependencies")
ydf_load_deps(exclude_repo=["tensorflow"])

# Load the dependencies of YDF that cannot be loaded above.
# Protobuf deps are loaded from grpc or Tensorflow. 
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

# Emscripten
http_archive(
  name = "emsdk",
  strip_prefix = "emsdk-3.1.15/bazel",
  url = "https://github.com/emscripten-core/emsdk/archive/refs/tags/3.1.15.zip",
  sha256 = "bc06feb66e376f0184929e603d7c02dcd07646ed6f89281bf1478bb8947fbb0f",
)
load("@emsdk//:deps.bzl", emsdk_deps = "deps")
emsdk_deps()
load("@emsdk//:emscripten_deps.bzl", emsdk_emscripten_deps = "emscripten_deps")
emsdk_emscripten_deps()
