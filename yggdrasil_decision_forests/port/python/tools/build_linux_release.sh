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


# Builds all python versions for release on Pypi

PYTHON_VERSIONS=( 3.8 3.9 3.10 3.11 )

function build_py() {
  local PYTHON="python"$1
  echo "Starting build with " $PYTHON
  $PYTHON -m venv /tmp/venv_$PYTHON
  source /tmp/venv_$PYTHON/bin/activate
  bazel clean --expunge
  COMPILERS="gcc" ./tools/test_pydf.sh
  ./tools/build_pydf.sh python
}

function main() {
  set -e
  for ver in ${PYTHON_VERSIONS[*]} 
  do
    build_py $ver 
  done
  set +e
}



main