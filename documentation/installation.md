# Installation

Yggdrasil Decision Forests (YDF) is available as a C++ library and a
Command-line-interface (CLI). Learners and models are compatible with both
interfaces.

## Table of Contents

<!--ts-->

*   [Installation](#installation)
    *   [Table of Contents](#table-of-contents)
    *   [Installation pre-compiled command-line-interface](#installation-pre-compiled-command-line-interface)
    *   [Compile command-line-interface from source](#compile-command-line-interface-from-source)
        *   [Linux / MacOS](#linux--macos)
        *   [Windows](#windows)
    *   [Running a minimal example](#running-a-minimal-example)
    *   [Compilation on and for Raspberry Pi](#compilation-on-and-for-raspberry-pi)
        *   [Install requirements](#install-requirements)
        *   [Compile Bazel](#compile-bazel)
        *   [Compile YDF](#compile-ydf)
        *   [Test YDF](#test-ydf)
    *   [Using the C++ library](#using-the-c-library)
    *   [Troubleshooting](#troubleshooting)

<!--te-->

## Installation pre-compiled command-line-interface

Pre-compiled binaries are available as
[github releases](https://github.com/google/yggdrasil-decision-forests/releases)
.

## Compile command-line-interface from source

**Requirements**

-   Microsoft Visual Studio >= 2019 (Windows)
-   GCC or Clang (Linux)
-   Bazel >= 3.7.2
-   Python >= 3
-   Git
-   Python's numpy
-   MSYS2 (Windows)

First install [Bazel](https://docs.bazel.build). Versions 3.7.2 and 4.0.0 are
supported:

-   On linux: `sudo apt update && sudo apt install bazel`

-   On windows: Follow
    [the guide](https://docs.bazel.build/versions/4.0.0/install-windows.html).

-   On Mac: Follow
    [the guide](https://docs.bazel.build/versions/master/install-os-x.html#install-with-installer-mac-os-x)
    or install bazel / bazelisk with [homebrew](https://brew.sh/): `brew install
    bazel` or `brew install bazelisk`. We recommend bazelisk, as it will select
    an appropriate version of bazel for the project automatically.

For more details (and troubleshooting), see the
[Bazel installation guide](https://docs.bazel.build/versions/4.0.0/install.html)
.

Once Bazel is installed, clone the github repository and start the compilation:

```shell
git clone https://github.com/google/yggdrasil-decision-forests.git
cd yggdrasil-decision-forests

bazel build //yggdrasil_decision_forests/...:all --config=<platform config>
```

For example:

### Linux / MacOS

```shell
git clone https://github.com/google/yggdrasil-decision-forests.git
cd yggdrasil-decision-forests

bazel build //yggdrasil_decision_forests/cli/...:all --config=linux_cpp17 --config=linux_avx2
```

*Note:* You can specify the compiler with `--repo_env=CC`. For example:

```shell
# Compile with GCC9
... --repo_env=CC=gcc-9

# Compile with Clang
... --repo_env=CC=clang
```

*Note:* On MacOS you may need to `brew install numpy` if a version installed via
pip is not available on the paths used by bazel.

### Windows

```shell

# Note: The python path should not contain spaces.
set PYTHON_BIN_PATH=C:\Python38\python.exe

git clone https://github.com/google/yggdrasil-decision-forests.git
cd yggdrasil-decision-forests

bazel build //yggdrasil_decision_forests/cli/...:all --config=windows_cpp17 --config=windows_avx2
```

*Note:* If multiple version of visual studio are installed, use
`BAZEL_VC_FULL_VERSION`. For example:

```shell
# Set a specific version of visual studio.
# The exact version can be found in `Program Files (x86)` e.g. C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29910.
set BAZEL_VC_FULL_VERSION=14.28.29910
```

**Important remarks**

-   The `.bazelrc` file contains implicit options used by the build.
-   By default, the binaries will not be compiled with support for TensorFlow IO
    and dataset formats. Support can be added with `--config=use_tensorflow_io`.
    In this case, a small fraction of TensorFlow is compiled. Some of the unit
    test require `--config=use_tensorflow_io`.
-   TensorFlow does not support C++17 on Windows. If using `use_tensorflow_io`,
    you have to use C++14 i.e. `--config=windows_cpp14`.

## Running a minimal example

The CLI binaries are now be available in the
`bazel-bin/yggdrasil_decision_forests/cli` directory. For example,
`bazel-bin/yggdrasil_decision_forests/cli/train` trains a model. Alternatively,
if you downloaded the binaries, extract them to the project root. Then, to run
the end-to-end example::

**On linux / MacOS:**

```shell
./examples/beginner.sh
```

**On windows:**

```shell
examples\beginner.bat
```

Optionally add the `bazel-bin/yggdrasil_decision_forests/cli` directory to your
`PATH`, to make all YDF binaries readily available.

**On linux/MacOS:**

-   Run the following commands:

```shell
echo "export PATH=\"$(pwd)/bazel-bin/yggdrasil_decision_forests/cli:\$PATH\"" >> ~/.bashrc
source ~/.bashrc
```

**On Windows:**

-   Run the command `echo %cd%\bazel-bin\yggdrasil_decision_forests\cli`
-   Go to *Advanced System Settings (Win+Pause) > Environment Variables > path
    (either in user or system variables) > New* and add the result of the
    command.

At this point, typing `train` in a shell will call the training code.

## Compilation on and for Raspberry Pi

Compiling YDF on and for a Raspberry Pi is similar to the linux compilation
except for:

-   Compiling Bazel from source: The Bazel team does not publish recompiled
    binaries for Arm CPUs (the type of CPU used in Raspberry Pi), therefore
    Bazel needs to be compiled from source. Note that compiling of Bazel takes
    more time that compiling YDF.
-   Bazel lacks configuration for Arm processors. We will have to set it
    manually.

The following instructions have been tested successfully on a Raspberry Pi 4
Model B Rev 1.4.

### Install requirements

Install GCC and Java JDK:

```shell
sudo apt-get update
sudo apt-get install openjdk-8-jdk gcc-8
```

### Compile Bazel

Note: Instructions for the compilation of Bazel is available in those guides:
[1](https://github.com/samjabrahams/tensorflow-on-raspberry-pi/blob/master/GUIDE.md)
,
[2](https://gitlab.com/arm-hpc/packages/-/wikis/packages/tensorflow?version_id=9cb9ae0120827dfccf609b3d316cc357c04d4e93)
,
[3](https://github.com/koenvervloesem/bazel-on-arm/blob/v3.4.0/patches/bazel-3.4.0-arm.patch)
. Refer to them in case of issues.

Download the source code of Bazel 4:

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

## Using the C++ library

Yggdrasil Decision Forests is accessible as a Bazel repository. In your Bazel
WORKSPACE adds the following lines:

**WORKSPACE**

```py
http_archive(
    name="ydf",
    strip_prefix="yggdrasil_decision_forests-master",
    urls=[
        "https://github.com/google/yggdrasil_decision_forests/archive/master.zip"],
)

load("@ydf//yggdrasil_decision_forests:library.bzl",
     ydf_load_deps="load_dependencies")
ydf_load_deps(repo_name="@ydf")
```

**Remarks :** `ydf_load_deps` injects the required dependencies (e.g. absl,
boost). Using the `exclude_repo`, you can disable to automated injection of some
of the dependencies (e.g. `exclude_repo = ["absl"]`).

YDF is now configured as an
[external dependency](https://docs.bazel.build/versions/master/external.html)
and can be used in your code. For example:

**BUILD**

```py
cc_binary(
    name="main",
    srcs=["main.cc"],
    deps=[
        "@ydf//yggdrasil_decision_forests/model/learner:learner_library",
        "@com_google_absl//absl/status",
    ],
)
```

**main.cc**

```c++
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "absl/status/status.h"

namespace ydf = yggdrasil_decision_forests;

void f() {
 ydf:: model::proto::TrainingConfig config;
 std::unique_ptr<ydf::model::AbstractLearner> learner;
 absl::Status status = model::GetLearner(config, &learner);
 // ...
}
```

An example is available at [examples/beginner.cc](../examples/beginner.cc) and
[examples/BUILD](../examples/BUILD).

## Troubleshooting

**`Time out` during building**

Bazel has a 10 minutes limit on each "command". If downloading one of the
dependency takes more than 10 minutes (e.g. with a slow connection, antivirus
scanning the files), you might get a timeout. Boost is the most likely culprit.
Follow the instructions in `third_party/boost/workspace.bzl` to download Boost
manually.

**`Repository command failed` with `execute(repository_ctx, [python_bin, "-c",
cmd])`**

Make sure python3 is installed and accessible through the PATH e.g. `SET
BAZEL_PYTHON=C:\Python38`. Alternatively, set `PYTHON_BIN_PATH` to the python
binary e.g. `SET PYTHON_BIN_PATH=C:\Python38\python.exe`.

**[Windows] `Error in fail: Repository command failed` with
`which(repository_ctx, "bash")`**

Make sure msys64 is installed and registered in the the PATH. Alternatively, set
`BAZEL_SH` to a bash e.g. `set BAZEL_SH=C:\tools\msys64\usr\bin\bash.exe`.

**[Windows] `external/nsync\version(1): error C2059: syntax error: 'constant'`**

This one is a bit awkward: It is due to an incompatibility between Boost and
Nsync (used by TensorFlow). Go to bazel `external` directory and remove the file
`external/nsync/version`. The error should be resolved until the next `bazel
clean --expunge`.

**[Windows; Library] `Compiling protos on window: command is longer than
CreateProcessW's limit (32767 characters)`**

This error is due to a bug in an early version of Protobuffer. Make sure you are
using Yggdrasil or TensorFlow version of protobuffer. If you are using Skylib,
make sure to use a recent version.

**[Linux] usr/bin/env: 'python': no such file or directory**

Bazel calls `python` during the compilation. Check which version of python you
have available and create an alias `sudo ln -s /usr/bin/python3 /usr/bin/python`
.

**[Windows] `fatal error LNK1120: 6 unresolved externals` +
`yggdrasil_decision_forests::serving::decision_forest::Idendity`**

You are using a non supported version of Visual Studio. Install VS>=2019
(VS>=14). If multiple version of VS are installed, specify the one used by Bazel
with `BAZEL_VC_FULL_VERSION`.

**Segmentation fault when any program starts on `std::filesystem::~path()`**

`lstdc++fs` is not linked. You are likely using GCC8 without TensorFlow. Update
to GCC>=9 or use TensorFlow for IO (`--config=use_tensorflow_io`).

**[MacOS] `ld: illegal thread local variable reference to regular symbol
__ZN4absl13base_internal19thread_identity_ptrE for architecture x86_64`**

Bazel can have issues with Clang and GRPC (see
[details](https://github.com/bazelbuild/bazel/issues/4341#issuecomment-758361769))
. Adding `--features=-supports_dynamic_linker` to the Bazel build command will
solve the issue.
