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


# Builds and tests the pip packages of all the supported Python versions for
# release on PyPI, for the platform of the host (Linux x86_64 or macOS arm64).
# The packages are in the dist/ directory.
#
# On Linux, the C++ toolchain (see MODULE.bazel) and the Python interpreters
# are hermetic, so the packages are manylinux_2_27 compatible when built on any
# Linux x86_64 host (no Docker image needed).
#
# Requirements: Bazelisk (as `bazel`), and the XCode command line tools on
# macOS.
#
# Usage example:
#   # Release all the Python versions.
#   ./tools/release.sh
#
#   # Release some Python versions.
#   PYTHON_VERSIONS="3.12 3.13" ./tools/release.sh
#
#   # Run the unit tests before packaging.
#   RUN_TESTS=1 ./tools/release.sh
#
# Options:
#  PYTHON_VERSIONS: Python versions of the wheels. Default: all the supported
#    versions (see MODULE.bazel).
#  RUN_TESTS: Run the unit tests before packaging, 0 (default) or 1.
#  BAZEL_FLAGS: Additional Bazel flags. Default: "".
#  YDF_PACKAGE_NAME: Distribution name, e.g. "ydf-staging". Default: "ydf".
#  YDF_VERSION_SUFFIX: Appended to the version, e.g. ".dev202610141400".

set -vex

: "${PYTHON_VERSIONS:=3.10 3.11 3.12 3.13 3.14}"
: "${RUN_TESTS:=0}"

function build_py() {
  local version=$1
  echo "Build YDF for Python ${version}"
  # The OS specific flags are in .bazelrc.
  local flags="--@rules_python//python/config_settings:python_version=${version} ${BAZEL_FLAGS}"

  if [[ "${RUN_TESTS}" = 1 ]]; then
    PYTHON_VERSION=${version} RUN_TESTS=1 ./tools/build_test_linux.sh
  fi

  # Builds and tests the wheel (see tools/BUILD). The wheel is a top-level
  # target so that Bazel also creates it if the test result is cached, e.g.
  # from a disk or remote cache.
  bazel test ${flags} --test_output=errors //tools:wheel //tools:wheel_test
  local wheel
  wheel=$(bazel cquery ${flags} --output=files //tools:wheel)
  mkdir -p dist
  install -m 644 "${wheel}" dist/
}

function main() {
  for version in ${PYTHON_VERSIONS}; do
    build_py "${version}"
  done
}

main
