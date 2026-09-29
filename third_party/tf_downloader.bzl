"""Configures the TensorFlow dependency for the open-source build.

This code is only required for injecting the TensorFlow header dependency in the
open-source build. It downloads the TensorFlow wheel from PyPI to get the
headers and the shared library.

Environment variables:
  YDF_TF_VERSION: Version of TensorFlow, e.g. "2.16.1". Required.
"""

_DEFAULT_PY_VERSION = "3.12"

def _error(msg):
    fail("\n\n\033[31m[Error]\033[0m " + msg + "\n")

def _target_os_and_arch(ctx):
    """Returns the OS and architecture as used in the wheel platform tags."""
    os_name = ctx.os.name.lower()
    arch = ctx.os.arch
    if arch == "amd64":
        arch = "x86_64"
    if "linux" in os_name:
        return "linux", arch
    if "mac" in os_name:
        if arch == "aarch64":
            arch = "arm64"
        return "macos", arch
    _error("Unsupported OS for the TensorFlow wheel: " + os_name)
    return None

def _select_wheel(files, py_tag, target_os, arch):
    """Returns the PyPI file entry of the matching wheel, or None."""
    for f in files:
        filename = f["filename"]
        if not filename.endswith(".whl"):
            continue

        # {distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl
        parts = filename[:-len(".whl")].split("-")
        if len(parts) < 5:
            continue
        python_tags = parts[-3].split(".")
        platform_tag = parts[-1]
        if py_tag not in python_tags:
            continue
        if target_os == "linux" and ("manylinux" not in platform_tag or arch not in platform_tag):
            continue
        if target_os == "macos" and ("macosx" not in platform_tag or arch not in platform_tag):
            continue
        return f
    return None

def _find_framework_lib(ctx, target_os):
    """Returns the path (relative to the repo) of libtensorflow_framework."""
    if target_os == "macos":
        expected = "libtensorflow_framework.2.dylib"
        prefix = "libtensorflow_framework"
        suffix = ".dylib"
    else:
        expected = "libtensorflow_framework.so.2"
        prefix = "libtensorflow_framework.so"
        suffix = ""
    if ctx.path("tensorflow/" + expected).exists:
        return "tensorflow/" + expected
    for entry in ctx.path("tensorflow").readdir():
        if entry.basename.startswith(prefix) and entry.basename.endswith(suffix):
            return "tensorflow/" + entry.basename
    return None

def _tf_downloader_impl(ctx):
    tf_version = ctx.os.environ.get("YDF_TF_VERSION")
    if not tf_version:
        _error("Environment variable 'YDF_TF_VERSION' is missing.\n" +
               "Please set it to the desired TensorFlow version (e.g., '2.16.1').")
    py_version = _DEFAULT_PY_VERSION
    py_tag = "cp" + py_version.replace(".", "")
    target_os, arch = _target_os_and_arch(ctx)

    # Find the wheel with the PyPI JSON API.
    ctx.download(
        url = "https://pypi.org/pypi/tensorflow/{}/json".format(tf_version),
        output = "pypi_tensorflow.json",
    )
    pypi_data = json.decode(ctx.read("pypi_tensorflow.json"))
    ctx.delete("pypi_tensorflow.json")

    wheel = _select_wheel(pypi_data.get("urls", []), py_tag, target_os, arch)
    if not wheel:
        _error("No wheel found for TF {} / Py {} on {} {}".format(
            tf_version,
            py_version,
            target_os,
            arch,
        ))

    ctx.download_and_extract(
        url = wheel["url"],
        sha256 = wheel["digests"]["sha256"],
        type = "zip",
    )

    framework_lib = _find_framework_lib(ctx, target_os)
    if not framework_lib:
        _error("Downloaded {} but could not find libtensorflow_framework inside.".format(
            wheel["filename"],
        ))

    ctx.file("BUILD", content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tensorflow_headers",
    hdrs = glob(["tensorflow/include/**"]),
    includes = ["tensorflow/include"],
)

cc_import(
    name = "tensorflow_lib",
    shared_library = "{}",
)

cc_library(
    name = "tensorflow",
    deps = [
        ":tensorflow_headers",
        ":tensorflow_lib",
    ],
)
""".format(framework_lib))

tf_downloader = repository_rule(
    implementation = _tf_downloader_impl,
    environ = ["YDF_TF_VERSION"],
)

# Bzlmod Extension Wrapper
def _tf_downloader_extension_impl(_):
    tf_downloader(name = "local_config_tf")

tf_downloaded_header_extension = module_extension(
    implementation = _tf_downloader_extension_impl,
)
