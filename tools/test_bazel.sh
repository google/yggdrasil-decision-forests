#!/bin/bash
# Copyright 2021 Google LLC.
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


# Compile and runs the unit tests.

set -x
set -e

# A specific version of GCC or Clang can be set as follow:
# export CC=gcc-9
# export CXX=g++-9

BAZEL=bazel-3.7.2
FLAGS="--config=linux_cpp17 --config=linux_avx2"

${BAZEL} build //yggdrasil_decision_forests/cli/...:all ${FLAGS}

${BAZEL} build //yggdrasil_decision_forests/cli/...:all ${FLAGS} --config=use_tensorflow_io

targets=""
subdirs=(cli dataset learner metric model serving)
for dir in "${subdirs[@]}"
do
  targets="$targets //yggdrasil_decision_forests/${dir}/...:all"
done

# Note: use_tensorflow_io is required for some of the unit test.
${BAZEL} test ${targets} ${FLAGS} --config=use_tensorflow_io
