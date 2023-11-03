"""Helpers to initialize third-party dependencies."""

load("//oss_third_party/pybind11:workspace.bzl", pybind11 = "deps")
load("//oss_third_party/pybind11_protobuf:workspace.bzl", pybind11_protobuf = "deps")

def load_dependencies(exclude_repo = []):
    """Initialize our third-party dependencies of PYDF.

    Args:
        exclude_repo: Repositories to not be loaded by PYDF (e.g. those already loaded elsewhere).
    """
    if "pybind11" not in exclude_repo:
        pybind11()

    if "pybind11_protobuf" not in exclude_repo:
        pybind11_protobuf()
