# Building and installing YDF

## Install from PyPi

To install YDF, run:

```
pip install ydf --upgrade
```

## Building

### Linux x86_64

After changing `requirements.txt` or `dev_requirements.txt`, update the lock
file with `bazel run //:requirements.update`.

#### Release script

The C++ toolchain (Clang with a glibc 2.27 sysroot) and the Python interpreters
are hermetic. The script `tools/release.sh` builds the wheels of all the
supported Python versions end-to-end. You can find the wheels in the `dist/`
subdirectory.

#### Manual build

Note that we may not be able to help with issues during manual builds.

**Requirements**

*   Bazel - version as specified in `.bazelversion`,
    [Bazelisk](https://github.com/bazelbuild/bazelisk) recommended


**Steps**

1.  Compile and test the code with

    ```shell
    # PYTHON_VERSION is one of the supported versions (see MODULE.bazel).
    # Default: 3.12.
    PYTHON_VERSION=3.12 RUN_TESTS=1 ./tools/build_test_linux.sh
    ```

    To use the C++ compiler of the host instead of the hermetic toolchain (e.g.
    for debugging), set `BAZEL_FLAGS=--config=local_cc`. The resulting wheels
    may not be manylinux-compatible.

1.  Build and test the Pip package

    ```shell
    bazel test --@rules_python//python/config_settings:python_version=3.12 \
      //tools:wheel //tools:wheel_test
    ```

    The wheel is in the `bazel-bin/tools/` directory.

### MacOS

**Requirements**

*   Bazel (version as specified in `.bazelversion`,
    [Bazelisk](https://github.com/bazelbuild/bazelisk) recommended)
*   XCode command line tools

**Building for all supported Python versions**

Simply run

```shell
./tools/release.sh
```

This will build a MacOS wheel for every supported Python version on the current
architecture. See the contents of this script for details about the build.

### Windows

Windows builds are not yet officially supported.
