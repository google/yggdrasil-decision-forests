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


# Compile and runs the unit tests on Windows
#
# Build and run the compatible test of YDF without tensorflow support. 
#
# Options:
#  CPP_VERSIONS: C++ Versions to build, separated by semicolon. Can be 14 or 17.
#                Defaults to 17
#  RUN_TESTS: Run the unit tests, 0 or 1 (default).
#
# Usage example:
#
#   # Compilation with C++17. running tests.
#   ./tools/test_bazel.sh
#    
#   # Compilation with C++14 and C++17, no tests run.
#   CPP_VERSIONS="14;17" RUN_TESTS=0 ./tools/test_bazel.sh
#
set -xev

build_and_maybe_test () {
   echo "Building YDF the following settings:"
   echo "   C++ Version: $1"

    BAZEL=bazel
    ${BAZEL} version

    local flags="--config=windows_cpp${1} --config=windows_avx2 --features=-fully_static_link --repo_env=CC=${2}"
    # Not all tests can be run without TF support
    local testable_components=""
    local buildable_cli_components=""
    # No tensorflow support
    cp -f WORKSPACE_NO_TF WORKSPACE
    buildable_cli_components=":all"
    testable_components="metric/...:all"

    time ${BAZEL} build //yggdrasil_decision_forests/cli${buildable_cli_components}  //examples:beginner_cc ${flags}
    if [ "$RUN_TESTS" = 1 ]; then
      time ${BAZEL} test //yggdrasil_decision_forests/${testable_components}  //examples:beginner_cc ${flags}
    fi
    echo "Building and maybe testing YDF complete."
} 

main () {
  # Set default values
  : "${CPP_VERSIONS:=17}"
  : "${RUN_TESTS:=1}"

  # Adjust bazelrc to prevent long paths issue
  echo "startup --output_user_root=${TMPDIR}" >> ".bazelrc"

  local cpp_version_array=(${CPP_VERSIONS//;/ })

  for cpp_version in ${cpp_version_array[@]}; do
    build_and_maybe_test $cpp_version
  done
}

main