# Bazel workspace configuration for Yggdrasil Decision Forest
#
# For support with TensorFlow file formats or TensorFlow IO, replace this file
# with WORKSPACE_WITH_TF, and use the "--config=use_tensorflow_io" flag. See
# tools/test_bazel_with_tf.sh for more details.
#
# WARNING: This file is a copy of WORKSPACE_NO_TF. Keep both files in sync.
#

workspace(name = "yggdrasil_decision_forests")

# Load the dependencies of YDF.
load("//yggdrasil_decision_forests:library.bzl", ydf_load_deps = "load_dependencies")
ydf_load_deps(exclude_repo=["tensorflow", "protobuf"])

# Load the dependencies of YDF that cannot be loaded above.
# Protobuf is loaded from pybind11_protobuf or Tensorflow. 
# TODO: Unify these dependencies.
# load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
# protobuf_deps()
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")