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

set -vex

PYTHON_VERSIONS=( 3.8 3.9 3.10 3.11 3.12 )

function build_py() {
  VERSION=$1
  echo "Build YDF for python $VERSION"

  # Note: set RUN_TESTS=1 to run the tests.
  CMD="python$VERSION -m venv /tmp/venv && source /tmp/venv/bin/activate && RUN_TESTS=0 ./tools/build_test_linux.sh && ./tools/package_linux.sh"

  # You can also start an interactive shell with:
  # CMD="python$VERSION -m venv /tmp/venv && source /tmp/venv/bin/activate && /bin/bash"
  # In the interactive shell, you can run commands such as "RUN_TESTS=1 ./tools/build_test_linux.sh"

  docker run \
    -v pydf_venv_cache_$VERSION:/tmp/venv \
    -v pydf_bazel_cache_$VERSION:/root/.cache \
    -v $(pwd)/../../../:/src \
    -w /src/yggdrasil_decision_forests/port/python \
    -it --rm \
    build_pydf \
    "${CMD}"
}

function main() {
  # Build docker image
  docker build -t build_pydf .

  # Build ydf for each compatible python version
  for VERSION in ${PYTHON_VERSIONS[*]} ; do
    build_py $VERSION 
  done
}

main
