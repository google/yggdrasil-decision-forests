load("@com_google_protobuf//bazel/toolchains:proto_lang_toolchain.bzl", "proto_lang_toolchain")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("@rules_python//python/uv:lock.bzl", "lock")
load("@ydf_wheel_config//:config.bzl", "PACKAGE_NAME", "VERSION")
load("//tools:wheel.bzl", "py_package_init_files")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Universal lock file of the third-party Python packages, for all the supported
# Python versions and platforms.
#
# Update with:
#   bazel run //:requirements.update
lock(
    name = "requirements",
    srcs = [
        "dev_requirements.txt",
        "requirements.txt",
    ],
    out = "requirements_lock.txt",
    args = [
        "--universal",
        "--python-version=3.10",
    ],
)

# Runtime of `py_proto_library`: The protobuf package from PyPI, i.e. the same
# as used by the users of the ydf package.
proto_lang_toolchain(
    name = "python_proto",
    command_line = "--python_out=%s",
    progress_message = "Generating Python proto_library %{label}",
    runtime = "@pypi_ydf//protobuf",
    toolchain_type = "@com_google_protobuf//bazel/private:python_toolchain_type",
)

# Pip package
# ===========
#
# Build the wheel for a given Python version with:
#   bazel build --config=linux_cpp17 \
#     --@rules_python//python/config_settings:python_version=3.12 //:wheel
#
# The name and version of the package are defined in the `ydf_wheel_config`
# repository (see MODULE.bazel and tools/wheel_config.bzl).

# Python versions of the wheels. Must match PYTHON_VERSIONS in MODULE.bazel.
WHEEL_PYTHON_VERSIONS = [
    "3.10",
    "3.11",
    "3.12",
    "3.13",
    "3.14",
]

# Python packages in the wheel.
WHEEL_PACKAGES = [
    "ydf",
    "yggdrasil_decision_forests",
]

# The Python files, generated protos (from the `py_proto_library` targets),
# native extension (ydf/cc/ydf.so) and data files (e.g. plotter.js) of the
# package. Third-party dependencies (e.g. the protobuf or numpy packages in the
# runfiles) are excluded by `packages`.
py_package(
    name = "wheel_files",
    packages = WHEEL_PACKAGES,
    deps = [
        "//ydf:api",
        "//ydf/learner:worker_main_lib",
    ],
)

py_package_init_files(
    name = "wheel_init_files",
    packages = WHEEL_PACKAGES,
    deps = [":wheel_files"],
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

config_setting(
    name = "macos_x86_64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "macos_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:aarch64",
    ],
)

# Note: `py_wheel` does not check the platform tag. On Linux x86_64, the
# hermetic C++ toolchain targets glibc 2.27 (see .bazelrc), and the tag is
# verified by `//tools:wheel_test` with auditwheel.
py_wheel(
    name = "wheel",
    abi = select({
        "@rules_python//python/config_settings:is_python_" + v: "cp" + v.replace(".", "")
        for v in WHEEL_PYTHON_VERSIONS
    }),
    author = "Mathieu Guillame-Bert, Richard Stotz, Jan Pfeifer",
    author_email = "decision-forests-contact@google.com",
    classifiers = [
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ] + ["Programming Language :: Python :: " + v for v in WHEEL_PYTHON_VERSIONS],
    description_content_type = "text/markdown",
    description_file = "README.md",
    distribution = PACKAGE_NAME,
    extra_distinfo_files = {
        "@ydf_wheel_config//:LICENSE": "licenses/LICENSE",
    },
    extra_requires = {
        "pandas": ["pandas"],
        "tensorflow": ["ydf-tf"],
    },
    homepage = "https://github.com/google/yggdrasil-decision-forests",
    license = "Apache 2.0",
    platform = select(
        {
            ":linux_x86_64": "manylinux_2_27_x86_64",
            ":linux_aarch64": "manylinux_2_27_aarch64",
            ":macos_x86_64": "macosx_10_15_x86_64",
            ":macos_arm64": "macosx_12_0_arm64",
        },
        no_match_error = "Unsupported platform for the YDF wheel.",
    ),
    project_urls = {
        "Documentation": "https://ydf.readthedocs.io/",
        "Source": "https://github.com/google/yggdrasil-decision-forests.git",
        "Tracker": "https://github.com/google/yggdrasil-decision-forests/issues",
    },
    python_requires = ">=" + WHEEL_PYTHON_VERSIONS[0],
    python_tag = select({
        "@rules_python//python/config_settings:is_python_" + v: "cp" + v.replace(".", "")
        for v in WHEEL_PYTHON_VERSIONS
    }),
    requires = [
        "numpy",
        "absl_py",
        "protobuf>=6.31.1,<8.0.0",
    ],
    strip_path_prefixes = ["wheel_init_files/"],
    summary = (
        "YDF (short for Yggdrasil Decision Forests) is a library for training," +
        " serving, evaluating and analyzing decision forest models such as" +
        " Random Forest and Gradient Boosted Trees."
    ),
    version = VERSION,
    deps = [
        ":wheel_files",
        ":wheel_init_files",
    ],
)
