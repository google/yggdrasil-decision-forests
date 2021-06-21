# Third party libraries
load("//third_party/gtest:workspace.bzl", gtest = "deps")
load("//third_party/absl:workspace.bzl", absl = "deps")
load("//third_party/protobuf:workspace.bzl", protobuf = "deps")
load("//third_party/zlib:workspace.bzl", zlib = "deps")
load("//third_party/tensorflow:workspace.bzl", tensorflow = "deps")
load("//third_party/farmhash:workspace.bzl", farmhash = "deps")
load("//third_party/boost:workspace.bzl", boost = "deps")
load("//third_party/grpc:workspace.bzl", grpc = "deps")

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

    # Inject transitive dependencies
    # Note: Bazel does not automatically inject transitive dependencies.

    # Protobuf.
    # Needs to be called before tensorflow because of a collision with Six.
    #load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
    #protobuf_deps()

    # TensorFlow.
    #load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
    #tf_workspace(tf_repo_name = "org_tensorflow")
