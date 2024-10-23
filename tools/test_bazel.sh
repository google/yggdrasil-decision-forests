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
# Options:
#  CPP_VERSIONS: C++ Versions to build, separated by semicolon. Can be 14 or 17.
#                Defaults to 17
#  RUN_TESTS: Run the unit tests, 0 or 1 (default).
#  COMPILERS: Compilers to build, separated by semicolon. Defaults to gcc-9
#  INSTALL_DEPENDENCIES: Installs required dependencies.
#
# Usage example:
#
#   # Compilation GCC 9, C++17. running tests.
#   ./tools/test_bazel.sh
#
set -xev

build_and_maybe_test () {
   echo "Building YDF the following settings:"
   echo "   C++ Version: $1"
   echo "   Compiler : $2"

    BAZEL=bazel
    ${BAZEL} version

    local flags="--config=linux_cpp${1} --config=linux_avx2 --features=-fully_static_link --repo_env=CC=${2} --build_tag_filters=-tf_dep --test_tag_filters=-tf_dep"

    # Do not build the PYDF targets, use tools/build_test_linux.sh instead
    local exclude_pydf_targets="-//yggdrasil_decision_forests/port/python/...:all"

    time ${BAZEL} build ${flags} -- //yggdrasil_decision_forests/...:all ${exclude_pydf_targets} //examples:beginner_cc
    if [ "$RUN_TESTS" = 1 ]; then
      time ${BAZEL} test ${flags} -- //yggdrasil_decision_forests/...:all ${exclude_pydf_targets} //examples:beginner_cc
    fi
    echo "Building and maybe testing YDF complete."
} 

main () {
  # Set default values
  : "${CPP_VERSIONS:=17}"
  : "${COMPILERS:="gcc-9"}"
  : "${RUN_TESTS:=1}"

  local cpp_version_array=(${CPP_VERSIONS//;/ })
  local compilers_array=(${COMPILERS//;/ })

  for cpp_version in ${cpp_version_array[@]}; do
    for compiler in ${compilers_array[@]}; do
        build_and_maybe_test $cpp_version $compiler
    done
  done
}

# Install the build dependencies
if [[ ! -z ${INSTALL_DEPENDENCIES+z} ]]; then

  # If the script is running as root (e.g. in the build docker), don't use sudo.
  if [ $(id -u) -eq 0 ]; then
    MAYBE_SUDO=""
  else
    MAYBE_SUDO=sudo
  fi

  ${MAYBE_SUDO} apt-get update
  ${MAYBE_SUDO} apt-get -y --no-install-recommends install \
    ca-certificates \
    build-essential \
    g++-10 \
    clang-12 \
    git \
    python3 \
    python3-pip \
    python3-dev \
    wget

  python3 -m pip install numpy

  wget -O bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.14.0/bazelisk-linux-amd64
  chmod +x bazel
  PATH="$(pwd):$PATH"
fi

main