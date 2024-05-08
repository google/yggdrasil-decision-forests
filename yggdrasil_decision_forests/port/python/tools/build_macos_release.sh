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


set -vex

declare -a python_versions=("3.8" "3.9" "3.10" "3.11" "3.12")

for pyver in "${python_versions[@]}"
do
  pyenv install -s $pyver
  export PYENV_VERSION=$pyver
  rm -rf ${TMPDIR}venv
  python -m venv ${TMPDIR}venv
  source ${TMPDIR}venv/bin/activate
  pip install --upgrade pip

  echo "Building with $(python3 -V 2>&1)"
  
  bazel clean --expunge
  RUN_TESTS=0 CC="clang" ./tools/test_pydf.sh
  ./tools/build_pydf.sh python
  deactivate
done