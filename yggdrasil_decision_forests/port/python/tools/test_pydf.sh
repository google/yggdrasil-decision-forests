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
#  COMPILERS: Compilers to build, separated by semicolon. Defaults to gcc-9
#
# Usage example:
#
#   # Compilation with GCC 9, C++17. Running tests.
#   ./tools/test_pydf.sh
#
#   # Compilation with Clang 14, without tests
#   COMPILERS="clang-14" RUN_TESTS=0 ./tools/test_pydf.sh
#
set -xev

build_and_maybe_test () {
   echo "Building PYDF the following settings:"
   echo "   Compiler : $1"

    BAZEL=bazel
    ${BAZEL} version

    local flags="--config=linux_cpp17 --config=linux_avx2 --features=-fully_static_link --repo_env=CC=${1}"
    local pydf_targets="//ydf/...:all"
    # Install PYDF components
    python -m pip install -r requirements.txt
    python -m pip install -r dev_requirements.txt

    time ${BAZEL} build ${flags} -- ${pydf_targets}
    if [[ "$RUN_TESTS" = 1 ]]; then
      time ${BAZEL} test ${flags} --test_output=errors -- ${pydf_targets}
    fi
    echo "PYDF build / test complete."
} 

main () {
  # Set default values
  : "${COMPILERS:="gcc-9"}"
  : "${RUN_TESTS:=1}"

  local compilers_array=(${COMPILERS//;/ })

for compiler in ${compilers_array[@]}; do
  build_and_maybe_test "$compiler"
done
}

main