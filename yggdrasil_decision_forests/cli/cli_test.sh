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



set -x
set -e

ROOT_DIR="$TEST_SRCDIR/yggdrasil_decision_forests/yggdrasil_decision_forests"

# Paths
TRAIN_DS="csv:$ROOT_DIR/test_data/dataset/adult_train.csv"
TEST_DS="csv:$ROOT_DIR/test_data/dataset/adult_test.csv"

CLI="$ROOT_DIR/cli"

# Generate the dataspec for the training dataspec.
DATASPEC="$TEST_TMPDIR/dataspec.pbtxt"
$CLI/infer_dataspec --dataset=$TRAIN_DS --output=$DATASPEC --alsologtostderr

# Human description of the dataspec.
DATASPEC_INFO="$TEST_TMPDIR/dataspec.txt"
$CLI/show_dataspec --dataspec=$DATASPEC --alsologtostderr | tee $DATASPEC_INFO

# Create a training configuration i.e. the hyper-parameters.
TRAINING_CONFIG="$TEST_TMPDIR/train_config.pbtxt"
cat <<EOF >>$TRAINING_CONFIG
task: CLASSIFICATION
label: "income"
learner: "GRADIENT_BOOSTED_TREES"

# Change learner specific hyper-parameters.
[yggdrasil_decision_forests.model.gradient_boosted_trees.proto.gradient_boosted_trees_config] {
  num_trees: 200
}
EOF

# Train the model.
# Note that there are not validation dataset. Some learners might extract a
# validation data from the training dataset.
MODEL="$TEST_TMPDIR/model"
$CLI/train \
  --dataset=$TRAIN_DS \
  --dataspec=$DATASPEC \
  --config=$TRAINING_CONFIG \
  --output=$MODEL \
  --alsologtostderr

# Display information about the model.
MODEL_INFO="$TEST_TMPDIR/model/description.txt"
$CLI/show_model --model=$MODEL --engines --alsologtostderr | tee $MODEL_INFO

# Evaluate the model on the test dataset.
EVALUATION="$TEST_TMPDIR/evaluation.txt"
$CLI/evaluate --dataset=$TEST_DS --model=$MODEL --alsologtostderr | tee $EVALUATION

# Export the predictions of the model.
PREDICTIONS="csv:$TEST_TMPDIR/prediction_test.csv"
$CLI/predict --dataset=$TEST_DS --model=$MODEL --output=$PREDICTIONS --alsologtostderr

# Benchmark the inference speed of the model.
BENCHMARK="$TEST_TMPDIR/benchmark.txt"
$CLI/benchmark_inference --dataset=$TEST_DS --model=$MODEL --alsologtostderr | tee $BENCHMARK

# Show the content of the working directory.
ls -l $TEST_TMPDIR
