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

r"""Distributed Training of a ranking model using Python YDF.

Introduction
============

This tutorial shows how to train a ranking model with distributed training on
Borg. The tutorial assumes you are familiar with the non-distributed training
ranking tutorial:
https://ydf.readthedocs.io/en/latest/tutorial/ranking/

Dataset
=======

In this example, we use a synthetic dataset with 1 million examples, 150
features, and 50 examples per groups. This dataset is small enough to be
trained without distributed training, but it is still a good example. The
dataset was generated with the
//yggdrasil_decision_forests/cli/utils:synthetic_dataset tool. A
copy of this dataset is available at:
/cns/is-d/home/gbm/ml_dataset_repository/synthetic_ranking_1m/

Examine the first five examples:
gqui from /cns/is-d/home/gbm/ml_dataset_repository/synthetic_ranking_1m/train@60 proto tensorflow.Example limit 5

**Details:**

The synthetic dataset was generated with:

```
bazel run -c opt --copt=-mavx2 //yggdrasil_decision_forests/cli/utils:synthetic_dataset -- \
    --alsologtostderr \
    --options=/tmp/synthetic_ranking_config.pbtxt\
    --train=recordio+tfe:/tmp/dataset/train@60 \
    --valid=recordio+tfe:/tmp/dataset/valid@20 \
    --test=recordio+tfe:/tmp/dataset/test@20 \
    --ratio_test=0.2 \
    --ratio_valid=0.2 \
    --one_shard_at_a_time
```

with the following config file `/tmp/synthetic_ranking_config.pbtxt`:

```
ranking { group_size: 50 }
num_examples: 1000000
num_examples_per_shards: 10000
num_numerical: 50
num_categorical: 50
num_boolean: 50
num_categorical_set: 0
```

Start distributed training
==========================

# Compile the training manager (this file) and the generic worker.
#
# Note: If your training dataset contains more than 1 billions examples, add the
# flag "--define=ydf_example_idx_num_bits=64" to the "bazel build" command.
#
bazel build -c opt --copt=-mfma --copt=-mavx2 --copt=-mavx \
--aspects=TemporalMpmAspect \
//yggdrasil_decision_forests/examples/distributed_training_ranking_python:main.par \
//yggdrasil_decision_forests/port/python/ydf/learner:worker_main.par

# Start the training
#
# Note: Check yggdrasil_decision_forests/port/python/ydf.borg for
# all the options.
#
borgcfg --vars= --skip_confirmation \
yggdrasil_decision_forests/port/python/ydf.borg \
--vars \
cell=is,charged_user=simple-ml-accounting,name=my_ydf_training,num_workers=10,cpu=30,manager_args=\"--project_dir=/cns/is-d/home/$USER/tmp/ttl=15d/ydf_training\",manager_binary=//yggdrasil_decision_forests/examples/distributed_training_ranking_python:main.par \
reload

# For larger datasets, allocate compute resources. For examples:
# - more workers (e.g. 100)
# - more cpu per workers ("cpu" argument in BorgConfig and "num_threads"
#   argument in the "DistributedGradientBoostedTreesLearner" configuration)
# - While training support worker interruptions, Borg alloate higher memory
#   bandwidth to jobs with higher priority and higher appclass (see BorgConfig)
#   which can improve the training speed significantly.
# - Increase the RAM if workers are running out of memory.

# Optionally, monitor the training directly in your shell.
btail my_ydf_training.manager

# You can stop training at anytime with:
borg --borg=is --user=$USER canceljob my_ydf_training.manager

# Before the training start, the dataset will be indexed. Once the training
# starts, you will see lines as follow in the terminal:
num-trees:126/200 valid-loss:0.000000 train-loss:-0.705096 train-NDCG@5:0.705096


# When the training is done, you will see the model evaluation:
Evaluation:
NDCG: 0.694741
num examples: 197250
num examples (weighted): 197250

# You can then load the model into a colab for further analysis, such as model
# understanding, prediction understanding, and benchmarking the model speed.
# See https://ydf.readthedocs.io for details.
"""

import os
from absl import app
from absl import flags
from absl import logging
import ydf

_WORKER_BNS = flags.DEFINE_string(
    "worker_bns",
    None,
    "BNS address (~ dynamic group of ips + ports) of the workers.",
)
_NUM_WORKERS = flags.DEFINE_integer("num_workers", None, "Number of workers")
_PROJECT_DIR = flags.DEFINE_string(
    "project_dir",
    None,
    "Directory containing the checkpoints and the final model",
)


def main(argv) -> None:
  # To show (a lot of) training logs
  ydf.verbose(2)

  # Configure the learning.
  learner = ydf.DistributedGradientBoostedTreesLearner(
      label="LABEL",
      ranking_group="GROUP",
      task=ydf.Task.RANKING,
      # Number of threads used by each worker.
      num_threads=30,
      # The working directory contains the model checkpoints. Change the working
      # directory when training a new model.
      working_dir=os.path.join(_PROJECT_DIR.value, "work_dir"),
      # Resume training if the manager or one of the worker is interrupted.
      resume_training=True,
      # Hyper-parameters can be set here.
      # See https://ydf.readthedocs.io/en/latest/py_api/DistributedGradientBoostedTreesLearner
      num_trees=200,
      # force_numerical_discretization=True drastically improves the training
      # speed and generally does not reduce the quality of the model.
      force_numerical_discretization=True,
  )
  learner.set_worker_bns(_WORKER_BNS.value, _NUM_WORKERS.value)

  # Train the model.
  model = learner.train(
      "recordio+tfe:/cns/is-d/home/gbm/ml_dataset_repository/synthetic_ranking_1m/train@60",
      valid="recordio+tfe:/cns/is-d/home/gbm/ml_dataset_repository/synthetic_ranking_1m/valid@20",
  )

  # Save the model to disk.
  model.save(os.path.join(_PROJECT_DIR.value, "model"))

  # Evaluate the model.
  # Note: The model evaluation is not distributed.
  evaluation = model.evaluate(
      "recordio+tfe:/cns/is-d/home/gbm/ml_dataset_repository/synthetic_ranking_1m/test@20"
  )
  logging.info("Evaluation:\n%s", evaluation)


if __name__ == "__main__":
  app.run(main)
