# Install the CLI API

This page explains how to install the CLI (Command line interface) of Yggdrasil
Decision Forests (YDF). Once installed, models can be trained and evaluated
using shell commands.

## Pre-compiled binaries

Linux and Windows pre-compiled binaries are available on the
[Github release page](https://github.com/google/yggdrasil-decision-forests/releases).
Look for the files named `cli_linux.zip` or `cli_windows.zip` corresponding to
the latest release.

## Compile from source

Instead of downloading pre-compiled binaries, you can compile YDF from the
source code.

### Requirements

The following libraries/tools are required to compile YDF.

-   Bazel or Bazelisk
-   Python >= 3.8
-   Git
-   Python's numpy
-   Docker (optional)

**Linux or Macos**

-   GCC or Clang. GCC>=9 recommended.

**Windows**

-   Microsoft Visual Studio >= 2019
-   MSYS2

### Common

First, clone the YDF GitHub repository:

```shell
git clone https://github.com/google/yggdrasil-decision-forests.git
cd yggdrasil-decision-forests
```

Then, follow the instructions corresponding to your OS.

### On Linux

Docker makes it easy to compile YDF without caring about dependencies. To build
YDF **with** Docker run the following command:

```shell
./tools/build_binary_release_in_docker.sh
```

Instead, to build YDF **without** Docker run:

```shell
INSTALL_DEPENDENCIES=1 BUILD=1./tools/build_binary_release.sh
```

Building options (e.g., select the compiler, disable support for TensorFlow,
etc.) are available by calling Bazel directly. See `tools/test_bazel.sh` for
examples. For example, the following command compiles YDF with c++17, AVX2
support and using GCC9.

```shell
bazel build //yggdrasil_decision_forests/cli:all \
  --config=linux_cpp17 \
  --config=linux_avx2 \
  --repo_env=CC=gcc-9
```

### On MacOS

To compile YDF on MacOS, run the following command:

```shell
bazel build //yggdrasil_decision_forests/cli:all --config=macos
```

### On Windows

To compile YDF on Windows, run the following command:

```shell
# Set the python path.
# Note: The python path should not contain spaces.
set PYTHON_BIN_PATH=C:\Python38\python.exe

bazel build //yggdrasil_decision_forests/cli:all \
  --config=windows_cpp17 \
  --config=windows_avx2
```

If multiple version of Visual Studio are installed on your computer, use the
`BAZEL_VC_FULL_VERSION` variable to specify the version to use. For example:

```shell
# Set a specific version of visual studio.
# The exact version can be found in `Program Files (x86)`
# e.g. C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29910.
set BAZEL_VC_FULL_VERSION=14.28.29910
```

## Running a minimal example

Once the compilation is done, the binaries are available in
`bazel-bin/yggdrasil_decision_forests/cli`. Those are the same binaries that you
can download in the pre-compiled binaries.

The scripts `./examples/beginner.sh` and `./examples/beginner.bat` are minimal
examples for Linux/MacOs and Windows.

On linux / MacOS, run:

```shell
./examples/beginner.sh
```

On Windows, run:

```shell
examples\beginner.bat
```

## Installing

Once compiled or downloaded, the YDF binaries can be added to your PATH to make
then easily accessible. For example, instead of typing
`bazel-bin/yggdrasil_decision_forests/cli/train`, you will type `train`.

On Linux / MacOS, run:

```shell
echo "export PATH=\"$(pwd)/bazel-bin/yggdrasil_decision_forests/cli:\$PATH\"" >> ~/.bashrc
source ~/.bashrc
```

On Windows:

-   Run the command `echo %cd%\bazel-bin\yggdrasil_decision_forests\cli`
-   Go to *Advanced System Settings (Win+Pause) > Environment Variables > path
    (either in user or system variables) > New* and add the result of the
    command.

## TensorFlow support

YDF can be compiled with or without TensorFlow. Compiling with TensorFlow adds
the following features:

-   Use TensorFlow for all IO operations. Without TensorFlow, IO operations are
    done using `<filesystem>` introduced in C++17.

-   Support of TensorFlow Record dataset format.

``` {note}
Compiling with TensorFlow IO is currently (Sept 2022) a solution to compile
YDF with C++14. However, this compatibility might be dropped at time.
```

To enable TensorFlow support:

1.  Runs:

```shell
cp -f WORKSPACE_WITH_TF WORKSPACE
```

1.  Then, add the following flag to the build command:

```
--config=use_tensorflow_io
```

## Compile YDF for Raspberry Pi

Compiling YDF on and for a Raspberry Pi is similar to the Linux compilation with
some exceptions:

-   The Bazel team does not publish pre-compiled binaries for ARM CPUs (the type
    of CPU used in Raspberry Pi), therefore Bazel needs to be compiled from
    source on Raspberry Pi. Note that compiling of Bazel takes more time that
    compiling YDF.
-   Bazel lacks configuration for Arm processors. We will have to set it
    manually.

The following instructions have been tested successfully on a Raspberry Pi 4
Model B Rev 1.4.

### Install requirements

On the Raspberry Pi, install GCC and Java JDK:

```shell
sudo apt-get update
sudo apt-get install openjdk-8-jdk gcc-8
```

### Compile Bazel

Detailed instructions and troubleshooting for compiling Bazel on Raspberry Pi is
available in the following guides:
[1](https://github.com/samjabrahams/tensorflow-on-raspberry-pi/blob/master/GUIDE.md)
,
[2](https://gitlab.com/arm-hpc/packages/-/wikis/packages/tensorflow?version_id=9cb9ae0120827dfccf609b3d316cc357c04d4e93)
,
[3](https://github.com/koenvervloesem/bazel-on-arm/blob/v3.4.0/patches/bazel-3.4.0-arm.patch)
. Refer to them in case of issues.

First, download the source code of Bazel 4:

```shell
wget https://github.com/bazelbuild/bazel/releases/download/4.2.2/bazel-4.2.2-dist.zip
unzip bazel-4.2.2-dist.zip -d bazel
```

Makes the following modifications in the Bazel source code:

-   In `bazel/tools/cpp/lib_cc_configure.bzl`:

    -   Around line 180, make the function `get_cpu_value` return `"arm"`
        independently of its parameters i.e. add `return "arm"` at the top of
        `get_cpu_value`'s body.

-   In `bazel/tools/cpp/unix_cc_configure.bzl`:

    -   Around line 392, replace `bazel_linkopts = "-lstdc++:-lm"` with
        `bazel_linkopts = "-lstdc++:-lm -latomic"`.

-   In `bazel/tools/jdk/BUILD`:

    -   Around line 142, replace `"//conditions:default": [],` with
        `//conditions:default": [":jni_md_header-linux"],`.

    -   Around line 153, replace `"//conditions:default": [],` with
        `"//conditions:default": ["include/linux"],`.

Compile Bazel:

```shell
cd bazel
EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh
```

This compilation stage takes a bit less than an hour.

Remember the location of bazel:

```shell
BAZEL=$(pwd)/output/bazel
```

### Compile YDF

Download the YDF source code:

```shell
git clone https://github.com/google/yggdrasil-decision-forests.git
cd yggdrasil-decision-forests
```

Compile YDF:

```shell
${BAZEL} build //yggdrasil_decision_forests/cli/...:all \
  --config=linux_cpp17 --features=-fully_static_link --host_javabase=@local_jdk//:jdk
```

### Test YDF

You can run the beginner example that train, evaluate and benchmark the
inference speed of a model:

```shell
./examples/beginner.sh
```

At the end of its execution, this script prints the inference speed of the
model. For example, on a Raspberry Pi 4 Model B Rev 1.4, I obtained.

```
batch_size : 100  num_runs : 20
time/example(us)  time/batch(us)  method
----------------------------------------
          12.754          1275.4  GradientBoostedTreesQuickScorerExtended [virtual interface]
          20.413          2041.2  GradientBoostedTreesGeneric [virtual interface]
          76.803          7680.3  Generic slow engine
----------------------------------------
```

For comparison, running `./examples/beginner.sh` on an Intel Xeon W-2135 returns
the following benchmark:

```
batch_size : 100  num_runs : 20
time/example(us)  time/batch(us)  method
----------------------------------------
          1.2968          129.68  GradientBoostedTreesQuickScorerExtended [virtual interface]
          6.9953          699.52  GradientBoostedTreesGeneric [virtual interface]
          16.108          1610.8  Generic slow engine
----------------------------------------
```

The speed difference is between 3x and 10x.
