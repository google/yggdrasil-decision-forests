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
    local ARCHITECTURE=$(uname -m)

    local flags="--config=linux_cpp17 --features=-fully_static_link --copt=-DYDF_USE_DYNAMIC_DISPATCH"
    python -m pip install -r requirements.txt

    if [[ "$RUN_TESTS" = 0 ]]; then
      # OSS builds don't check with Pytype, but we need to compile all targets
      # to ensure protos are compiled for Python.
      bazel build ${flags} -- //ydf/...:all
    else
      python -m pip install -r dev_requirements.txt
      time bazel build ${flags} -- //ydf/...:all
      time bazel test ${flags} --test_output=errors -- //ydf/...:all
    fi
} 

main () {
  # Set default values
  : "${RUN_TESTS:=1}"

  build_and_maybe_test
}

main