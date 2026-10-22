# Pip package (wheel) of YDF.
#
# Build and test the wheel for a given Python version with:
#   bazel test --@rules_python//python/config_settings:python_version=3.12 \
#     //tools:wheel_test

load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("@rules_python//python:py_test.bzl", "py_test")
load("@ydf_wheel_config//:config.bzl", "PACKAGE_NAME", "PYTHON_VERSIONS", "VERSION")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_package(
    name = "wheel_files",
    packages = [
        "ydf",
        "yggdrasil_decision_forests",
    ],
    deps = ["//ydf:api"],
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
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

# Python and ABI tag of the wheel, e.g. "cp312" (see PYTHON_VERSIONS in
# MODULE.bazel).
PYTHON_TAG = select(
    {
        "@rules_python//python/config_settings:is_python_" + v: "cp" + v.replace(".", "")
        for v in PYTHON_VERSIONS
    },
    no_match_error = "Unsupported Python version. See PYTHON_VERSIONS in MODULE.bazel.",
)

# The wheel (bazel-bin/tools/<name>-<version>-<tags>.whl).
#
# Note: `py_wheel` does not check the platform tag.
py_wheel(
    name = "wheel",
    abi = PYTHON_TAG,
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
    ] + ["Programming Language :: Python :: " + v for v in PYTHON_VERSIONS],
    description_content_type = "text/markdown",
    description_file = "//:README.md",
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
            ":macos_arm64": "macosx_12_0_arm64",
        },
        no_match_error = "Unsupported platform for the YDF wheel.",
    ),
    project_urls = {
        "Documentation": "https://ydf.readthedocs.io/",
        "Source": "https://github.com/google/yggdrasil-decision-forests.git",
        "Tracker": "https://github.com/google/yggdrasil-decision-forests/issues",
    },
    python_requires = ">=" + PYTHON_VERSIONS[0],
    python_tag = PYTHON_TAG,
    requires = [
        "numpy",
        "absl_py",
        "protobuf>=6.31.1,<8.0.0",
    ],
    # `py_wheel` has no `keywords` attribute; append the `Keywords:` header line
    # after `Summary:`.
    summary = (
        "YDF (short for Yggdrasil Decision Forests) is a library for training," +
        " serving, evaluating and analyzing decision forest models such as" +
        " Random Forest and Gradient Boosted Trees." +
        "\nKeywords: " + ",".join([
            "machine learning",
            "decision forests",
            "random forest",
            "gradient boosted decision trees",
            "CART",
            "isolation forests",
            "classification",
            "regression",
            "ranking",
            "uplift",
            "boosting",
        ])
    ),
    version = VERSION,
    deps = [":wheel_files"],
)

py_test(
    name = "wheel_test",
    srcs = ["wheel_test.py"],
    args = [
        "--wheel=$(rootpath :wheel)",
    ],
    data = [
        ":wheel",
    ],
    deps = [
        "@abseil-py//absl/flags",
        "@abseil-py//absl/testing:absltest",
        "@pypi_ydf//numpy",
        "@pypi_ydf//packaging",
        "@pypi_ydf//protobuf",
    ] + select({
        "@platforms//os:linux": ["@pypi_ydf//auditwheel"],
        "//conditions:default": [],
    }),
)
