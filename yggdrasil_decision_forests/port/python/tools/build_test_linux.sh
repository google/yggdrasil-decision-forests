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
#
# Usage example:
#
#   # Compilation with Clang 14, without tests
#   CC="clang-14" RUN_TESTS=0 ./tools/build_test_linux.sh
#
set -vex

build_and_maybe_test () {
   echo "Building PYDF the following settings:"
   echo "   Compiler : $CC"

    bazel version
    local ARCHITECTURE=$(uname --m)

    local flags="--config=linux_cpp17 --features=-fully_static_link"
    if [ "$ARCHITECTURE" == "x86_64" ]; then
        flags="$flags --config=linux_avx2"
    fi
    local pydf_targets="//ydf/...:all"
    # Install PYDF components
    python -m pip install -r requirements.txt
    python -m pip install -r dev_requirements.txt

    time bazel build ${flags} -- ${pydf_targets}
    if [[ "$RUN_TESTS" = 1 ]]; then
      time bazel test ${flags} --test_output=errors -- ${pydf_targets}
    fi
    echo "PYDF build / test complete."
} 

main () {
  # Set default values
  : "${RUN_TESTS:=1}"

  build_and_maybe_test
}

main