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



set -x
set -e

# Compile the example.
#
# See the "Using the C++ library" section in the user manual for more details
# about the API. See the "Compile command-line-interface from source" section in
# the user manual for more details about the compilation flags.
#
# Add "--config=use_tensorflow_io" to support TFRecord format natively.

bazel build --config=linux_cpp17 --config=linux_avx2 //:beginner_cc

# Run the example.
# The "dataset_dir" should contains the "adult_train.csv" and "adult_test.csv"
# files. See "beginner.cc"'s header for detailed explanations about the example.

./bazel-bin/beginner_cc \
  --dataset_dir=../../yggdrasil_decision_forests/test_data/dataset \
  --output_dir=/tmp/yggdrasil_decision_forest
