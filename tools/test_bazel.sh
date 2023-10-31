#!/bin/bash
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Compile and runs the unit tests.
#
# By default, build and run the compatible test of YDF without tensorflow
# support. Alternatively, if TF_SUPPORT is set to 1, build and run YDF
# with tensorflow support.
#
# Options:
#  CPP_VERSIONS: C++ Versions to build, separated by semicolon. Can be 14 or 17.
#                Defaults to 17
#  RUN_TESTS: Run the unit tests, 0 or 1 (default).
#  TF_SUPPORT: Whether or not to build with Tensorflow support. Can be "ON",
#              "OFF" (default) or "BOTH" (for building both variants)
#  COMPILERS: Compilers to build, separated by semicolon. Defaults to gcc-9
#  INSTALL_DEPENDENCIES: Installs required dependencies.
#
# Usage example:
#
#   # Compilation without TF support, GCC 9, C++17. running tests.
#   ./tools/test_bazel.sh
#
#   # Compilation with TF support, C++17 and C++14, Clang and gcc-12, run all tests.
#   TF_SUPPORT="ON" COMPILERS="gcc-12" CPP_VERSIONS="17" RUN_TESTS=1 ./tools/test_bazel.sh
#
#   # Compilation with TF support, C++14 and C++17, Clang and gcc-9, no tests run.
#   TF_SUPPORT="ON" COMPILERS="clang;gcc-9" CPP_VERSIONS="14;17" RUN_TESTS=0 ./tools/test_bazel.sh
#
set -xev

build_and_maybe_test () {
   echo "Building YDF the following settings:"
   echo "   C++ Version: $1"
   echo "   Compiler : $2"
   echo "   Tensorflow support: $3"

    BAZEL=bazel
    ${BAZEL} version

    local flags="--config=linux_cpp${1} --config=linux_avx2 --features=-fully_static_link --repo_env=CC=${2}"
    # Not all tests can be run without TF support
    local testable_components=""
    local buildable_cli_components=""
    # Do not build the PYDF targets, use tools/test_pydf.sh instead
    local pydf_targets="-//yggdrasil_decision_forests/port/python/...:all"
    if [ "$3" = 0 ]; then
      # No tensorflow support
      cp -f WORKSPACE_NO_TF WORKSPACE
      buildable_cli_components=":all"
      testable_components="metric/...:all"
    else 
      cp -f WORKSPACE_WITH_TF WORKSPACE
      flags="${flags} --config=use_tensorflow_io"
      buildable_cli_components="/...:all"
      testable_components="...:all"
    fi

    time ${BAZEL} build ${flags} -- //yggdrasil_decision_forests/cli${buildable_cli_components} ${pydf_targets} //examples:beginner_cc
    if [ "$RUN_TESTS" = 1 ]; then
      time ${BAZEL} test ${flags} -- //yggdrasil_decision_forests/${testable_components} ${pydf_targets} //examples:beginner_cc
    fi
    echo "Building and maybe testing YDF complete."
} 

main () {
  # Set default values
  : "${CPP_VERSIONS:=17}"
  : "${COMPILERS:="gcc-9"}"
  : "${RUN_TESTS:=1}"
  : "${TF_SUPPORT:="OFF"}"

  local cpp_version_array=(${CPP_VERSIONS//;/ })
  local compilers_array=(${COMPILERS//;/ })

  local tf_supports;
  if [ "${TF_SUPPORT}" = "ON" ]; then
    tf_supports=(1)
  elif [ "${TF_SUPPORT}" = "OFF" ]; then
    tf_supports=(0)
  elif [ "${TF_SUPPORT}" = "BOTH" ]; then
    tf_supports=(0 1)
  else 
    echo "ERROR: Invalid value for TF_SUPPORT ${TF_SUPPORT}. Allowed values are \"ON\", \"OFF\", \"BOTH\""
    exit 1;
  fi

  for cpp_version in ${cpp_version_array[@]}; do
    for compiler in ${compilers_array[@]}; do
      for tf_support in ${tf_supports[@]}; do
        build_and_maybe_test $cpp_version $compiler $tf_support
      done
    done
  done
}

# Install the build dependencies
if [[ ! -z ${INSTALL_DEPENDENCIES+z} ]]; then

  # If the script is running as root (e.g. in the build docker), don't use sudo.
  if [ $(id -u) -eq 0 ]
  then
    SUDO=""
  else
    SUDO=sudo
  fi

  $SUDO apt-get update
  $SUDO apt-get -y --no-install-recommends install \
    ca-certificates \
    build-essential \
    g++-10 \
    clang-12 \
    git \
    python3 \
    python3-pip \
    python3-dev

  python3 -m pip install numpy

  wget -O bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.14.0/bazelisk-linux-amd64
  chmod +x bazel
  PATH="$(pwd):$PATH"
fi

main