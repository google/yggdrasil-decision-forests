"""Configures the TensorFlow dependency for the open-source build.

This code is only required for injecting the TensorFlow header dependency in the
open-source build. It queries the installed TensorFlow package to find the
headers and shared library.
"""

def _tf_downloader_impl(ctx):
    tf_version = ctx.os.environ.get("YDF_TF_VERSION")
    py_version = ctx.os.environ.get("YDF_PY_VERSION")

    if not tf_version:
        fail("\n\n\033[31m[Error]\033[0m Environment variable 'YDF_TF_VERSION' is missing.\n" +
             "Please set it to the desired TensorFlow version (e.g., '2.16.1').\n")

    if not py_version:
        fail("\n\n\033[31m[Error]\033[0m Environment variable 'YDF_PY_VERSION' is missing.\n" +
             "Please set it to the Python version (e.g., '3.11').\n")

    os_name = ctx.os.name
    arch = ctx.os.arch

    if arch == "amd64":
        arch = "x86_64"
    elif arch == "aarch64":
        arch = "aarch64"

    target_os = "unknown"
    if "linux" in os_name:
        target_os = "linux"
    elif "mac" in os_name:
        target_os = "macos"
        if arch == "aarch64":
            arch = "arm64"

    # 3. Python Script to Query PyPI
    # We use a helper script to parse the JSON from PyPI and find the correct wheel URL.
    # This is much more robust than trying to construct the URL string manually.
    python_bin = ctx.which("python3")
    if not python_bin:
        fail("Could not find python3 in PATH")

    pypi_script = """
import sys
import json
import urllib.request

def main():
    if len(sys.argv) < 5:
        print("Usage: resolve_tf.py <tf_ver> <py_ver> <os> <arch>", file=sys.stderr)
        sys.exit(1)

    tf_ver = sys.argv[1]
    py_ver = sys.argv[2]
    target_os = sys.argv[3]
    target_arch = sys.argv[4]

    url = f"https://pypi.org/pypi/tensorflow/{tf_ver}/json"

    try:
        with urllib.request.urlopen(url) as r:
            data = json.load(r)
    except Exception as e:
        print("Error fetching PyPI data:", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    found_url = ""
    found_sha256 = ""
    
    # Clean python version for wheel filename matching (e.g. 3.9 -> cp39)
    py_ver_clean = "cp" + py_ver.replace(".", "")

    for file in data.get("urls", []):
        name = file["filename"]
        if not name.endswith(".whl"):
            continue
        
        if py_ver_clean not in name:
            continue

        if target_os == "linux":
            if "manylinux" not in name:
                continue
            if target_arch not in name:
                continue
        elif target_os == "macos":
            if "macosx" not in name:
                continue
            if target_arch not in name:
                continue

        found_url = file["url"]
        found_sha256 = file["digests"]["sha256"]
        break

    if not found_url:
        print(f"No wheel found for TF {tf_ver} / Py {py_ver} on {target_os} {target_arch}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps({"url": found_url, "sha256": found_sha256}))

if __name__ == "__main__":
    main()
"""

    ctx.file("resolve_tf.py", pypi_script)
    result = ctx.execute([
        python_bin,
        "resolve_tf.py",
        tf_version,
        py_version,
        target_os,
        arch,
    ])

    if result.return_code != 0:
        fail("Failed to resolve TensorFlow URL:\n" + result.stderr)

    tf_data = json.decode(result.stdout)
    tf_url = tf_data["url"]
    tf_sha256 = tf_data["sha256"]

    ctx.download_and_extract(
        url = tf_url,
        sha256 = tf_sha256,
        type = "zip",
    )

    if target_os == "macos":
        # Specifically look for the .2.dylib on macOS as requested
        search_pattern = "libtensorflow_framework.2.dylib"
    else:
        # Fallback to standard .so for Linux
        search_pattern = "libtensorflow_framework.so*"

    # We search specifically inside the 'tensorflow' directory to avoid deps
    find_cmd = ["find", "tensorflow", "-name", search_pattern]
    find_result = ctx.execute(find_cmd)
    libs = find_result.stdout.strip().split("\n")

    selected_lib = None
    for lib in libs:
        # We want the one inside the tensorflow folder, not local_config symlinks
        if "tensorflow/libtensorflow_framework" in lib and not lib.endswith(".deps"):
            selected_lib = lib
            break

    if not selected_lib:
        fail("Downloaded wheel but could not find libtensorflow_framework inside.")

    # 6. Generate BUILD file
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
""".format(selected_lib))

tf_downloader = repository_rule(
    implementation = _tf_downloader_impl,
    local = True,
    environ = ["YDF_TF_VERSION", "YDF_PY_VERSION", "PATH"],
)

# Bzlmod Extension Wrapper
def _tf_downloader_extension_impl(_):
    tf_downloader(name = "local_config_tf")

tf_downloaded_header_extension = module_extension(
    implementation = _tf_downloader_extension_impl,
)
