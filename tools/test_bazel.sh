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

set -xev

BAZEL=bazel
${BAZEL} version

echo "====================================================="
echo "1. With TensorFlow IO, c++14 and without abslStatusOr"
echo "====================================================="

# With TensorFlow IO, without StatusOr (c++14 with gcc8)
FLAGS="--config=linux_cpp14 --config=linux_avx2 --features=-fully_static_link --config=use_tensorflow_io --define=no_absl_statusor=1 --repo_env=CC=gcc-8"
time ${BAZEL} build //yggdrasil_decision_forests/cli/...:all ${FLAGS}
time ${BAZEL} test //yggdrasil_decision_forests/{cli,dataset,learner,metric,model,serving}/...:all ${FLAGS}


echo "====================================================="
echo "2. Without TensorFlow IO, c++17 and with abslStatusOr"
echo "====================================================="

# Without TensorFlow IO (c++17)
FLAGS="--config=linux_cpp17 --config=linux_avx2 --features=-fully_static_link"
time ${BAZEL} build //yggdrasil_decision_forests/cli/...:all ${FLAGS}
time ${BAZEL} test //yggdrasil_decision_forests/{cli,metric,model,serving,utils}/...:all //examples:beginner_cc ${FLAGS}

echo "=================================================="
echo "3. With TensorFlow IO, c++17 and with abslStatusOr"
echo "=================================================="

# With TensorFlow IO (c++17)
FLAGS="--config=linux_cpp17 --config=linux_avx2 --features=-fully_static_link --config=use_tensorflow_io"
time ${BAZEL} build //yggdrasil_decision_forests/cli/...:all ${FLAGS}
time ${BAZEL} test //yggdrasil_decision_forests/...:all ${FLAGS}
