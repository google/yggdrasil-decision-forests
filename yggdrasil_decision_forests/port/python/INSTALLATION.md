# Building and installing YDF

## Install from PyPi

To install YDF, run:

```
pip install ydf --upgrade
```

## Building

### Pre-work

Use `tools/change_version.sh` to update the version number (if needed) and
remember to update `CHANGELOG.md`.

### Linux x86_64

After changing `requirements.txt` or `dev_requirements.txt`, update the lock
file with `bazel run //:requirements.update`.

#### Release script

The C++ toolchain (Clang with a glibc 2.27 sysroot) and the Python interpreters
are hermetic. The script `tools/release_linux.sh` builds the wheels of all the
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
    # PYTHON_VERSION is one of 3.10, 3.11, 3.12 (default), 3.13, 3.14.
    PYTHON_VERSION=3.12 RUN_TESTS=1 ./tools/build_test_linux.sh
    ```

    To use the C++ compiler of the host instead of the hermetic toolchain (e.g.
    for debugging), set `BAZEL_FLAGS=--config=local_cc`. The resulting wheels
    are not manylinux-compatible.

1.  Build the Pip package

    ```shell
    # Create a virtual environment with the (hermetic) Python interpreter of the
    # Bazel build.
    bazel run --@rules_python//python/config_settings:python_version=3.12 \
      @rules_python//python/bin:python -- -m venv --clear /tmp/venv
    source /tmp/venv/bin/activate
    PYTHON_VERSION=3.12 RUN_TESTS=0 ./tools/build_test_linux.sh
    PYTHON_VERSION=3.12 ./tools/package_linux.sh
    deactivate
    ```

### Linux ARM64

This build configuration is experimental at this time and may break.

#### Docker

For building manylinux_2_28-compatible packages, you can use an appropriate
Docker image. The pre-configured build script at
`tools/build_linux_aarch64_release_in_docker.sh` starts a container and builds
the wheels end-to-end. You can find the wheels in the `dist/`subdirectory.

For details and configuration options, please consult the corresponding scripts.

### MacOS

**Requirements**

*   Bazel (version as specified in `.bazelversion`,
    [Bazelisk](https://github.com/bazelbuild/bazelisk) recommended)
*   XCode command line tools

**Building for all supported Python versions**

Simply run

```shell
./tools/release_macos.sh
```

This will build a MacOS wheel for every supported Python version on the current
architecture. See the contents of this script for details about the build.

### MacOS cross-compilation

We have not tested MacOS cross-compilation (Intel <-> ARM) for YDF yet, though
it is on our roadmap.

### AArch64

We have not tested AArch64 compilation for YDF yet.

### Windows

Windows builds are not yet officially supported.

See `tools\release_windows.bat` for details.

**Requirements**

-   MSys2
-   Python versions installed in "C:\Python<version>" e.g. C:\Python310.
-   Bazel - version as specified in `.bazelversion`
    [Bazelisk](https://github.com/bazelbuild/bazelisk) recommended
-   Visual Studio (tested with VS2019 and VS2022).

**Steps**

Simply run

```shell
tools\release_windows.bat
```

This will build a Windows wheel for every supported Python version on the
current architecture. See the contents of this script for details about the
build.

**Optionally**, edit `tools\release_windows.bat` to

-   Only compile YDF for a specific version of Windows.
-   Configure the paths to MSys2, Python, and VS.

**Issues**

-   `tools\release_windows.bat` compiles and tests YDF with various compatible
    libraries such as TensorFlow and Jax. Compiling YDF does not need those
    library. Notably, you can disable those libraries in `dev_requirements.txt`
    if they fail to install on your environment.
-   If the compilation fails with the error `error C2475:
    'upbc::kRepeatedFieldArrayGetterPostfix': redefinition; 'constinit'
    specifier mismatch`, you are seeing an issue caused by an old version of
    protobuf (`upb` to be precise). To solve this error, make the following
    changes:
    -   In `bazel-python\external\upb\upbc\names.c`, add the two missing
        `ABSL_CONST_INIT`. More precisely, replace `const absl::string_view
        kRepeatedFieldArrayGetterPostfix = ...;` with `ABSL_CONST_INIT const
        absl::string_view kRepeatedFieldArrayGetterPostfix = ...;`. Do the same
        for `kRepeatedFieldMutableArrayGetterPostfix`.
