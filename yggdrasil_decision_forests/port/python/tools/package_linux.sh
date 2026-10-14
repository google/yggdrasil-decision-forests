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


# Builds and tests the YDF pip package (wheel) with Bazel, and copies it to the
# dist/ directory.
#
# The wheel is the `//:wheel` target (see BUILD). `//tools:wheel_test` checks
# its content, imports it and checks its manylinux tag with auditwheel.
#
# Options:
#  PYTHON_VERSION: Python version of the wheel, e.g. "3.11". Default: "3.12".
#  BAZEL_FLAGS: Additional Bazel flags. Default: "".
#  YDF_PACKAGE_NAME: Distribution name, e.g. "ydf-staging". Default: "ydf".
#  YDF_VERSION_SUFFIX: Appended to the version, e.g. ".dev202610141400".
#
# Usage example:
#   # Generate the pip package for Python 3.11
#   PYTHON_VERSION=3.11 ./tools/package_linux.sh

set -vex

: "${PYTHON_VERSION:=3.12}"

function check_directory() {
  local cur_dir=${PWD##*/}
  if [ ! -f "MODULE.bazel" ] || [ "${cur_dir}" != "python" ]; then
    echo "This script should be run from the root directory of the Python port of Yggdrasil Decision Forests (i.e. probably under port/python)."
    exit 1
  fi
}

function main() {
  check_directory

  local flags="--config=linux_cpp17 --features=-fully_static_link --@rules_python//python/config_settings:python_version=${PYTHON_VERSION} ${BAZEL_FLAGS}"

  # Builds and tests the wheel.
  bazel test ${flags} --test_output=errors //tools:wheel_test

  # Copies the wheel to dist/.
  local wheel=$(bazel cquery ${flags} --output=files //:wheel 2>/dev/null)
  mkdir -p dist
  cp -f "${wheel}" dist/
  chmod u+w "dist/$(basename "${wheel}")"
  echo "Created dist/$(basename "${wheel}")"
}

main
