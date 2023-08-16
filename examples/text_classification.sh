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



# Example of binary text classification using categorical-set features.
#
# See https://arxiv.org/abs/2009.09991 for more details on categorical-set
# features.


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
TRAIN_DS="csv:$DATASET_DIR/sst_binary_train.csv"
TEST_DS="csv:$DATASET_DIR/sst_binary_test.csv"

# Directory containing the artifacts of the project.
PROJECT="${HOME}/yggdrasil_decision_forests_text_classification"
mkdir -p $PROJECT

# Generate the dataspec for the training dataspec.
DATASPEC_GUIDE="$PROJECT/dataspec_guide.pbtxt"
cat <<EOF > $DATASPEC_GUIDE
# Make sure the label is detected as categorical (and not boolean).
column_guides {
  column_name_pattern: "^label$"
  type: CATEGORICAL
  }
# Dictionary and tokenizer of the text feature.
column_guides {
  column_name_pattern: "^sentence$"
  type: CATEGORICAL_SET
  categorial {
    min_vocab_frequency: 5
    max_vocab_count: 2000
    }
  tokenizer {
    tokenizer {
      splitter: SEPARATOR
      separator: " ;,"
      }
    }
  }
EOF
DATASPEC="$PROJECT/dataspec.pbtxt"
$CLI/infer_dataspec --guide=$DATASPEC_GUIDE --dataset=$TRAIN_DS --output=$DATASPEC --alsologtostderr

# Human description of the dataspec.
DATASPEC_INFO="$PROJECT/dataspec.txt"
$CLI/show_dataspec --dataspec=$DATASPEC --alsologtostderr | tee $DATASPEC_INFO

# Create a training configuration i.e. the hyper-parameters.
TRAINING_CONFIG="$PROJECT/train_config.pbtxt"
cat <<EOF > $TRAINING_CONFIG
task: CLASSIFICATION
label: "label"
learner: "RANDOM_FOREST"

# Change the hyper-parameters.
[yggdrasil_decision_forests.model.random_forest.proto.random_forest_config] {
  num_trees: 100 # Add more trees (e.g. 300-2k) for a real model.
  decision_tree {
    categorical_set_greedy_forward {
      sampling: 0.1
      max_selected_items: -1
      }
    }
  }
EOF

# Train the model.
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
