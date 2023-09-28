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

# Make sure your cpus are configured for high performance.
#
# On linux, run:
#   sudo apt install linux-cpupower
#   sudo cpupower frequency-set --governor performance

bazel build -c opt //third_party/yggdrasil_decision_forests/cli/monitoring:benchmark_training

bazel-bin/third_party/yggdrasil_decision_forests/cli/monitoring/benchmark_training --output=/tmp/train_benchmark_new.csv

# Let's assume you generated a benchmark in /tmp/train_benchmark_old.csv
bazel run -c opt //third_party/yggdrasil_decision_forests/cli/monitoring:compare_benchmark -- \
 --old=/tmp/train_benchmark_old.csv \
 --new=/tmp/train_benchmark_new.csv
