# Building and installing YDF

## Install from PyPi

To install YDF, run:

```
pip install ydf --upgrade
```

## Building

### Pre-work

Use `tools/update_version.sh` to update the version number (if needed) and
remember to update `CHANGELOG.md`.

### Linux

#### Docker

For building manylinux2014-compatible packages, you can use an appropriate
Docker image. The pre-configured build script at
`tools/build_linux_release_in_docker.sh` starts a container and builds the
wheels end-to-end. You can find the wheels in the `dist/`subdirectory.

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
    RUN_TESTS=1 ./tools/test_pydf.sh
    deactivate
    ```

    Substitute for your compiler name / version

1. Build the Pip package

    ```shell
    PYTHON_BIN=python
    ./tools/build_pydf.sh $PYTHON_BIN
    ```

    If you want to build with [Pyenv](https://github.com/pyenv/pyenv) for all supported Python versions, run

    ```shell
    ./tools/build_pydf.sh ALL_VERSIONS
    ```

### MacOS

**Requirements**

*   Bazel (version as specified in `.bazelversion`, 
    [Bazelisk](https://github.com/bazelbuild/bazelisk) recommended)
*   XCode command line tools
*   [Pyenv](https://github.com/pyenv/pyenv)

**Building for all supported Python versions**

Simply run

```shell
./tools/build_macos_release.sh
```
This will build a MacOS wheel for every supported Python version on the current
architecture. See the contents of this script for details about the build.

### MacOS cross-compilation

We have not tested MacOS cross-compilation (Intel <-> ARM) for YDF yet, though
it is on our roadmap.

### AArch64

We have not tested AArch64 compilation for YDF yet.

### Windows

TODO, see `tools/build.bat`.
