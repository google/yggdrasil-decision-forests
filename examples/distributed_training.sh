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



# Example showing the distributed training and evaluation of a Gradient Boosted
# Decision Trees model.
#
# For the sake of the examples, the workers are running on the local host.
# However in practice,  workers should be run on separate remote machines.
#
# The worked directory and results are created in
# ${HOME}/yggdrasil_decision_forests_distributed_training
#
# Before running this script, make sure to compile or download the Yggdrasil
# Decision Forests binaries.
#
# See https://github.com/google/yggdrasil-decision-forests/blob/main/documentation/user_manual.md#distributed-training for more details.
#

set -x
set -e

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
#
# To speed-up distributed training, it is recommended to use a sharded dataset
# with at least as much shards as workers. If the dataset is not sharded, part
# of the distributed training could be slowest tha ideal (depending on the
# learning algorithm).
TRAIN_DS="csv:$DATASET_DIR/adult_train.csv"
TEST_DS="csv:$DATASET_DIR/adult_test.csv"

# Directory containing the artifacts of the project.
PROJECT="${HOME}/yggdrasil_decision_forests_distributed_training"
mkdir -p $PROJECT

# Generate the dataspec for the training dataspec.
# The dataspec should be inferred from a subset of the training dataset if the
# training dataset is large.
DATASPEC="$PROJECT/dataspec.pbtxt"
$CLI/infer_dataspec --dataset=$TRAIN_DS --output=$DATASPEC --alsologtostderr

# Create a training configuration i.e. specify the hyper-parameters.
TRAINING_CONFIG="$PROJECT/train_config.pbtxt"
cat <<EOF > $TRAINING_CONFIG
task: CLASSIFICATION
label: "income"
learner: "DISTRIBUTED_GRADIENT_BOOSTED_TREES"

# Change learner specific hyper-parameters.
[yggdrasil_decision_forests.model.distributed_gradient_boosted_trees.proto.distributed_gradient_boosted_trees_config] {
  gbt {
    num_trees: 200
  }
}
EOF

# Create a deployment configuration i.e. specify the computing resources
# available for the training.
DEPLOYMENT_CONFIG="$PROJECT/deployment_config.pbtxt"
cat <<EOF > $DEPLOYMENT_CONFIG
cache_path: "${PROJECT}/cache"
num_threads: 2 # Each worker will run 2 threads.
try_resume_training: true # Allow training to be interrupted and resumed.

distribute {
  implementation_key: "GRPC"
  [yggdrasil_decision_forests.distribute.proto.grpc] {
    socket_addresses {
      # Configure the 3 workers.
      addresses { ip: "localhost" port: 2001 }
      addresses { ip: "localhost" port: 2002 }
      addresses { ip: "localhost" port: 2003 }
      }
    }
  }
EOF

# Start the workers locally.
#
# Note: The worker binary "grpc_worker_main" is also available at the root of
# of the recompiled binary package.
WORKER_BIN="$CLI/../utils/distribute/implementations/grpc/grpc_worker_main"
$WORKER_BIN --port=2001 &
$WORKER_BIN --port=2002 &
$WORKER_BIN --port=2003 &

# Train the model.
# Note that there are not validation dataset. Some learners might extract a
# validation data from the training dataset.
MODEL="$PROJECT/model"
$CLI/train \
  --dataset=$TRAIN_DS \
  --dataspec=$DATASPEC \
  --config=$TRAINING_CONFIG \
  --output=$MODEL \
  --deployment=$DEPLOYMENT_CONFIG \
  --alsologtostderr

# Stop the workers (because they run locally).
killall grpc_worker_main
wait

# Display information about the model.
MODEL_INFO="$PROJECT/model/description.txt"
$CLI/show_model --model=$MODEL --engines --alsologtostderr | tee $MODEL_INFO

# Evaluate the model on the test dataset.
EVALUATION="$PROJECT/evaluation.txt"
$CLI/evaluate --dataset=$TEST_DS --model=$MODEL --alsologtostderr | tee $EVALUATION

# Show the content of the working directory.
ls -l $PROJECT
