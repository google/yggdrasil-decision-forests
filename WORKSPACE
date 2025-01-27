# Bazel workspace configuration for Yggdrasil Decision Forest C++ and JS.
# For Yggdrasil Decision Forest Python, see port/python/WORKSPACE.

workspace(name = "yggdrasil_decision_forests")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Load the dependencies of YDF
# ============================

load("//yggdrasil_decision_forests:library.bzl", ydf_load_deps = "load_dependencies")
ydf_load_deps()

# Initialize dependencies that need manual initialization
# ========================================================

# gRPC; needed for any distributed computation capability (e.g., distributed training).
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()


# TODO: Enbable CUDA build in OSS
# load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")
# rules_cuda_dependencies()
# register_detected_cuda_toolchains()

# Emscripten; needed for the Web build.
http_archive(
    name = "emsdk",
    sha256 = "ee008c9aff9a633f6da37f53283b40654635f1739e0f1bcde9d887d895803157",
    strip_prefix = "emsdk-3.1.74/bazel",
    url = "https://github.com/emscripten-core/emsdk/archive/refs/tags/3.1.74.zip",
)
load("@emsdk//:deps.bzl", emsdk_deps = "deps")
emsdk_deps()
load("@emsdk//:emscripten_deps.bzl", emsdk_emscripten_deps = "emscripten_deps")
emsdk_emscripten_deps()
