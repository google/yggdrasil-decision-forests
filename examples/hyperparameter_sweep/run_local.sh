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

COMPILE="bazel build -c opt --copt=-mavx2 --config=linux_cpp17 --repo_env=CC=clang-10"
CODE=examples/hyperparameter_sweep
cp -f WORKSPACE_NO_TF WORKSPACE
PROJECT=~/project/hp_sweep

${COMPILE} //${CODE}:manager_main

bazel-bin/${CODE}/manager_main \
  --alsologtostderr \
  --num_repetitions=1 \
  --max_num_runs=5 \
  --work_dir=${PROJECT}/work_dir \
  --output_dir=${PROJECT}/output_dir \
  --label=binary_class \
  --dataset_train=csv:${PROJECT}/dataset/hls4ml_HLF_binary_train.csv \
  --dataset_test=csv:${PROJECT}/dataset/hls4ml_HLF_binary_test.csv \
  '--distribute_config=implementation_key:"MULTI_THREAD" [yggdrasil_decision_forests.distribute.proto.multi_thread] { num_workers:1 }'
