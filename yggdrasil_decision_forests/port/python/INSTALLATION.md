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

#### Docker

For building manylinux2014-compatible packages, you can use an appropriate
Docker image. The pre-configured build script at
`tools/release_linux_in_docker.sh` starts a container and builds the wheels
end-to-end. You can find the wheels in the `dist/`subdirectory.

#### Manual build

Note that we may not be able to help with issues during manual builds.

**Requirements**

*   Bazel - version as specified in `.bazelversion`,
    [Bazelisk](https://github.com/bazelbuild/bazelisk) recommended
*   GCC >= 9 or Clang >= 14
*   rsync
*   Python headers (e.g. `python-dev` package on Ubuntu)
*   Python virtualenv

**Steps**

1.  Compile and test the code with

    ```shell
    # Create a virtual environment where Python dependencies will be installed.
    python -m venv myvenv
    RUN_TESTS=1 ./tools/build_test_linux.sh
    deactivate
    ```

    Substitute for your compiler name / version

1.  Build the Pip package

    ```shell
    PYTHON_BIN=python
    ./tools/package_linux.sh $PYTHON_BIN
    ```

    If you want to build with [Pyenv](https://github.com/pyenv/pyenv) for all
    supported Python versions, run

    ```shell
    ./tools/package_linux.sh ALL_VERSIONS
    ```

### Linux ARM64

This build configuration is experimental at this time and may break.

#### Docker

For building manylinux2014-compatible packages, you can use an appropriate
Docker image. The pre-configured build script at
`tools/build_linux_aarch64_release_in_docker.sh` starts a container and builds
the wheels end-to-end. You can find the wheels in the `dist/`subdirectory.

For details and configuration options, please consult the corresponding scripts.

### MacOS

**Requirements**

*   Bazel (version as specified in `.bazelversion`,
    [Bazelisk](https://github.com/bazelbuild/bazelisk) recommended)
*   XCode command line tools
*   [Pyenv](https://github.com/pyenv/pyenv)

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
