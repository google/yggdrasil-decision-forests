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



# Example showing the hyper-parameter tuning and evaluation of a model.
# The model will be tuned on the local machine. For distributed hyper-parameter
# tuning, see
# third_party/yggdrasil_decision_forests/examples/example_template/run.sh.
#
# This example is similar to the "beginner.sh" example, with the exception for
# configuration of the training config (see "TRAINING_CONFIG").
#
# Check the documentation as:
# https://ydf.readthedocs.io/en/latest/cli_user_manual.html#automated-hyper-parameter-tuning
#
# It will create a directory ${HOME}/yggdrasil_decision_forests_hptuning and
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
PROJECT="${HOME}/yggdrasil_decision_forests_hptuning"
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
learner: "HYPERPARAMETER_OPTIMIZER"
label: "income"

[yggdrasil_decision_forests.model.hyperparameters_optimizer_v2.proto
      .hyperparameters_optimizer_config] {

  # Do not retrained the model after the hyper-parameter tuning.
  retrain_final_model: false

  # How to tune the learner.
  optimizer {
    # Random exploration
    optimizer_key: "RANDOM"
    [yggdrasil_decision_forests.model.hyperparameters_optimizer_v2.proto.random] {
    # For serious models, you need significantly more trials e.g., 100-1000.
    num_trials: 25
    }
  }

  # The learner to tune.
  base_learner {
    learner: "GRADIENT_BOOSTED_TREES"
    [yggdrasil_decision_forests.model.gradient_boosted_trees.proto.gradient_boosted_trees_config] {
      # You can specify the hyper-parameters NOT to optimize here.
      num_trees: 300
      }
  }

  # The computing resources to train the individual learners.
  # In this case, each model is trained with a single thread.
  base_learner_deployment {
    # The multi-threading is done at the optimizer level.
    num_threads: 1
  }

  # Hyper-parameter space to optimize.
  #
  # The space of hyper-paramter to optimize can be defined automatically or
  # manually.

  # Automatically set the hyper-parameters to optimize.
  predefined_search_space {}

  # Alternatively, manually set the hyper-parameters to optimize.
  #
  # Check the following links for explanations and advices on how to configure
  # hyper-parameters for automated tuning:
  # https://ydf.readthedocs.io/en/latest/hyper_parameters.html
  # https://ydf.readthedocs.io/en/latest/improve_model.html
  #
  # search_space {
  #   fields {
  #     name: "num_candidate_attributes_ratio"
  #     discrete_candidates {
  #       possible_values { real: 1.0 }
  #       possible_values { real: 0.8 }
  #       possible_values { real: 0.6 }
  #     }
  #   }
  #
  #   fields {
  #     name: "use_hessian_gain"
  #     discrete_candidates {
  #       possible_values { categorical: "true" }
  #       possible_values { categorical: "false" }
  #     }
  #   }
  #
  #   fields {
  #     name: "growing_strategy"
  #     discrete_candidates {
  #       possible_values { categorical: "LOCAL" }
  #       possible_values { categorical: "BEST_FIRST_GLOBAL" }
  #     }
  #
  #     children {
  #       parent_discrete_values {
  #         possible_values { categorical: "LOCAL" }
  #       }
  #       name: "max_depth"
  #       discrete_candidates {
  #         possible_values { integer: 4 }
  #         possible_values { integer: 5 }
  #         possible_values { integer: 6 }
  #         possible_values { integer: 7 }
  #       }
  #     }
  #
  #     children {
  #       parent_discrete_values {
  #         possible_values { categorical: "BEST_FIRST_GLOBAL" }
  #       }
  #       name: "max_num_nodes"
  #       discrete_candidates {
  #         possible_values { integer: 16 }
  #         possible_values { integer: 32 }
  #         possible_values { integer: 64 }
  #         possible_values { integer: 128 }
  #       }
  #     }
  #   }
  # }
  
}
EOF

# Train the model.
# Note that there is not validation dataset. Some learners might extract a
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

# Evaluate the model on the test dataset.
EVALUATION="$PROJECT/evaluation.txt"
$CLI/evaluate --dataset=$TEST_DS --model=$MODEL --alsologtostderr | tee $EVALUATION

# Show the content of the working directory.
ls -l $PROJECT
