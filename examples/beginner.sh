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



# Example showing the training, evaluation and inference benchmark of a model.
#
# It will create a directory ${HOME}/yggdrasil_decision_forests_beginner and
# store the trained models there.
#
# Before running this script, make sure to compile or download the Yggdrasil
# Decision Forests binaries.
#

set -vex

# Find Yggdrasil Decision Forests
if command -v ./infer_dataspec &> /dev/null; then
  echo "Using Yggdrasil Decision Forest from the PATH"
  CLI=.
  DATASET_DIR=yggdrasil_decision_forests/test_data/dataset

elif command -v ../bazel-bin/yggdrasil_decision_forests/cli/infer_dataspec  &> /dev/null ;then
  echo "Using Yggdrasil Decision Forest from the example directory."
  CLI=../bazel-bin/yggdrasil_decision_forests/cli
  DATASET_DIR=../yggdrasil_decision_forests/test_data/dataset

elif command -v bazel-bin/yggdrasil_decision_forests/cli/infer_dataspec  &> /dev/null ;then
  echo "Using Yggdrasil Decision Forest from the root directory."
  CLI=bazel-bin/yggdrasil_decision_forests/cli
  DATASET_DIR=yggdrasil_decision_forests/test_data/dataset

else
  echo "Yggdrasil Decision Forest was not found. Compile it and add the
  bazel-bin/yggdrasil_decision_forests/cli directory to the PATH, or run this
  command from the Yggdrasil's example directory (i.e. the directory containing
  this file)."
  exit 1
fi

# Start of the tutorial.

# Datasets
TRAIN_DS="csv:$DATASET_DIR/adult_train.csv"
TEST_DS="csv:$DATASET_DIR/adult_test.csv"

# Directory containing the artifacts of the project.
PROJECT="${HOME}/yggdrasil_decision_forests_beginner"
mkdir -p $PROJECT

# Generate the dataspec for the training dataspec.
DATASPEC="$PROJECT/dataspec.pbtxt"
$CLI/infer_dataspec --dataset=$TRAIN_DS --output=$DATASPEC --alsologtostderr

# Human description of the dataspec.
DATASPEC_INFO="$PROJECT/dataspec.txt"
$CLI/show_dataspec --dataspec=$DATASPEC --alsologtostderr | tee $DATASPEC_INFO

# Create a training configuration i.e. the hyper-parameters.
TRAINING_CONFIG="$PROJECT/train_config.pbtxt"
cat <<EOF > $TRAINING_CONFIG
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
MODEL="$PROJECT/model"
$CLI/train \
  --dataset=$TRAIN_DS \
  --dataspec=$DATASPEC \
  --config=$TRAINING_CONFIG \
  --output=$MODEL \
  --alsologtostderr

# Display information about the model.
MODEL_INFO="$PROJECT/model/description.txt"
$CLI/show_model --model=$MODEL --engines --alsologtostderr | tee $MODEL_INFO

# Analyse the model
$CLI/experimental_analyze_model_and_dataset --dataset=$TEST_DS --model=$MODEL --alsologtostderr --output="$PROJECT/analyse_report"

# Evaluate the model on the test dataset.
EVALUATION="$PROJECT/evaluation.txt"
$CLI/evaluate --dataset=$TEST_DS --model=$MODEL --alsologtostderr | tee $EVALUATION

# Export the predictions of the model.
PREDICTIONS="csv:$PROJECT/prediction_test.csv"
$CLI/predict --dataset=$TEST_DS --model=$MODEL --output=$PREDICTIONS --alsologtostderr

# Benchmark the inference speed of the model.
BENCHMARK="$PROJECT/benchmark.txt"
$CLI/benchmark_inference --dataset=$TEST_DS --model=$MODEL --alsologtostderr | tee $BENCHMARK

# Show the content of the working directory.
ls -l $PROJECT
