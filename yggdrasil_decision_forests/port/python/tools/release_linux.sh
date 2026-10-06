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


# Builds the pip packages of all the supported Python versions for release on
# PyPI. The packages are in the dist/ directory.
#
# The C++ toolchain (see MODULE.bazel) and the Python interpreters are
# hermetic, so the packages are manylinux_2_27 compatible when built on any
# Linux x86_64 host (no Docker image needed).
#
# Requirements: Bazelisk (as `bazel`) and rsync.
#
# Usage example:
#   # Release all the Python versions.
#   ./tools/release_linux.sh
#
#   # Release some Python versions.
#   PYTHON_VERSIONS="3.12 3.13" ./tools/release_linux.sh
#
#   # Run the unit tests before packaging.
#   RUN_TESTS=1 ./tools/release_linux.sh

set -vex

: "${PYTHON_VERSIONS:=3.10 3.11 3.12 3.13 3.14}"
: "${RUN_TESTS:=0}"
: "${VENV_ROOT:=/tmp/ydf_release_venv}"

function build_py() {
  local version=$1
  local venv="${VENV_ROOT}/${version}"
  echo "Build YDF for Python ${version}"

  # Virtual environment of the hermetic Python interpreter of the Bazel build.
  bazel run --@rules_python//python/config_settings:python_version=${version} \
    @rules_python//python/bin:python -- -m venv --clear "${venv}"

  (
    source "${venv}/bin/activate"
    export PYTHON_VERSION=${version}
    RUN_TESTS=${RUN_TESTS} ./tools/build_test_linux.sh
    ./tools/package_linux.sh
  )
}

function main() {
  for version in ${PYTHON_VERSIONS}; do
    build_py "${version}"
  done
}

main
