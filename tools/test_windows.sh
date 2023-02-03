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


# Compile and runs the unit tests on Windows and C++20.
#
# Build and run the compatible test of YDF without tensorflow support. 
#
# This script is tested with MSVC 2017 and C++20. Other compilers and C++
# versions are not supported as of yet, but patches are welcome. Notably,
# earlier C++ versions are not compatible with MSVC and the current abseil due
# to its use of designated initializers.
#
# Options:
#  RUN_TESTS: Run the unit tests, 0 or 1 (default).
#
# Usage example:
#
#   # Compilation with C++20. running tests.
#   ./tools/test_windows.sh
#
set -xev

build_and_maybe_test () {
   echo "Start building YDF"

    BAZEL=bazel
    ${BAZEL} version

    local flags="--config=windows_cpp20 --config=windows_avx2 --features=-fully_static_link"
    # Not all tests can be run without TF support
    local testable_components=":all"
    local buildable_cli_components="metric/...:all"
    # No tensorflow support
    cp -f WORKSPACE_NO_TF WORKSPACE

    time ${BAZEL} build //yggdrasil_decision_forests/cli${buildable_cli_components}  //examples:beginner_cc ${flags}
    if [ "$RUN_TESTS" = 1 ]; then
      time ${BAZEL} test //yggdrasil_decision_forests/${testable_components}  //examples:beginner_cc ${flags}
    fi
    echo "Building and maybe testing YDF complete."
} 

main () {
  # Set default values
  : "${RUN_TESTS:=1}"

  # Adjust bazelrc to prevent long paths issue
  echo "startup --output_user_root=${TMPDIR}" >> ".bazelrc"

  build_and_maybe_test $cpp_version
}

main