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



# Compile and runs the unit tests for the Python port of YDF
#
#
# Options:
#  RUN_TESTS: Run the unit tests, 0 or 1 (default).
#  PYTHON_VERSION: Python version to compile (and test) for, e.g. "3.11".
#    Default: "3.12".
#  BAZEL_FLAGS: Additional Bazel flags. Default: "".
#
# The C++ toolchain is hermetic (see MODULE.bazel) and ignores CC / CXX.
#
# Usage example:
#
#   # Compilation without tests
#   RUN_TESTS=0 ./tools/build_test_linux.sh
#
#   # Compilation with the C++ compiler of the host
#   BAZEL_FLAGS="--config=local_cc" RUN_TESTS=0 ./tools/build_test_linux.sh
#
#   # Compilation and tests with Python 3.11
#   PYTHON_VERSION=3.11 ./tools/build_test_linux.sh
#
set -vex

build_and_maybe_test () {
   echo "Building PYDF the following settings:"
   echo "   Flags    : $BAZEL_FLAGS"
   echo "   Python   : $PYTHON_VERSION"

    bazel version

    local flags="--config=linux_cpp17 --features=-fully_static_link --@rules_python//python/config_settings:python_version=${PYTHON_VERSION} ${BAZEL_FLAGS}"

    if [[ "$RUN_TESTS" = 0 ]]; then
      # OSS builds don't check with Pytype, but we need to compile all targets
      # to ensure protos are compiled for Python. The targets depending on
      # Tensorflow or Grain are only tests / benchmarks.
      bazel build ${flags} --build_tag_filters=-tf_dep,-grain_dep -- //ydf/...:all
    else
      local tag_filters=""
      case "${PYTHON_VERSION}" in
        3.13|3.14)
          # TensorFlow and Grain are not available for these Python versions.
          tag_filters="--build_tag_filters=-tf_dep,-grain_dep --test_tag_filters=-tf_dep,-grain_dep"
          ;;
      esac
      time bazel build ${flags} ${tag_filters} -- //ydf/...:all
      time bazel test ${flags} ${tag_filters} --test_output=errors -- //ydf/...:all
    fi
}

main () {
  # Set default values
  : "${RUN_TESTS:=1}"
  : "${PYTHON_VERSION:=3.12}"

  build_and_maybe_test
}

main