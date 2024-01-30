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

r"""Distributed Training using Python YDF.

Introduction
============

This tutorial demonstrates how to tune a model with distributed computation on
Borg. It assumes you are familiar with the generated distributed training
and tuning tutorials:
https://ydf.readthedocs.io/en/latest/tutorial/distributed_training
https://ydf.readthedocs.io/en/latest/tutorial/tuning

This tutorial is similar to the distributed training tutorial (yggdrasil_decision_forests/examples/vizier_tuning_python/main.py).
However, instead of using a distributed learner (i.e. learning algorithm), we
use a non-distributed learner, but distribute the hyper-parameter search.

Dataset
=======

In this example, we use the adult dataset available at:
/cns/is-d/home/gbm/ml_dataset_repository/others/adult/adult_train.riotfe
/cns/is-d/home/gbm/ml_dataset_repository/others/adult/adult_test.riotfe

The files are RecordIO containing TensorFlow Example protos.

Examine the first five examples:
gqui from /cns/is-d/home/gbm/ml_dataset_repository/others/adult/adult_train.riotfe proto tensorflow.Example limit 5

Test binary locally
===================

You can run tuning locally to ensure everything is well configured with:

bazel run -c opt --copt=-mfma --copt=-mavx2 --copt=-mavx \
//yggdrasil_decision_forests/examples/vizier_tuning_python:main \
-- --alsologtostderr --project_dir=/tmp/test_tuner

Note that running the tuning locally will be much slower.

Start distributed computation
=============================

# Compile the training manager (this file) and the generic worker.
bazel build -c opt --copt=-mfma --copt=-mavx2 --copt=-mavx \
--aspects=TemporalMpmAspect \
//yggdrasil_decision_forests/examples/vizier_tuning_python:main.par \
//yggdrasil_decision_forests/port/python/ydf/learner:worker_main.par

# Start the training
#
# Note: Check yggdrasil_decision_forests/port/python/ydf.borg for
# all the options.
borgcfg --vars= --skip_confirmation \
yggdrasil_decision_forests/port/python/ydf.borg \
--vars \
cell=is,charged_user=simple-ml-accounting,name=my_ydf_training,num_workers=10,manager_args=\"--project_dir=/cns/is-d/home/$USER/tmp/ttl=15d/ydf_training\",manager_binary=//yggdrasil_decision_forests/examples/vizier_tuning_python:main.par \
reload

# Optionally, monitor the tuning logs directly in your shell.
btail my_ydf_training.manager

# If you use the Vizier tuner, you can also monitor the tuning progress at:
https://vizier.corp.google.com/

# You can stop training at anytime with:
borg --borg=is --user=$USER canceljob my_ydf_training.manager

# When the training is done, you will see the model evaluation:
  accuracy: 0.678026
  confusion matrix:
      label (row) \ prediction (col)
      +-------+-------+-------+
      |       |   C_0 |   C_1 |
      +-------+-------+-------+
      |   C_0 | 67509 | 32394 |
      +-------+-------+-------+
      |   C_1 | 31826 | 67728 |
      +-------+-------+-------+
  characteristics:
      name: 'C_1' vs others
      ROC AUC: 0.740736
      PR AUC: 0.733204
      Num thresholds: 10000
  loss: 0.605254
  num examples: 199457
  num examples (weighted): 199457
  
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

  # Configure the tuning
  #
  # NOTE: The VizierTuner produce good results but it takes time to run. For
  # instance, Vizier can take up to 30s to generate a hyper-parameter
  # configuration (look for "Waiting for Suggestions" in the manager logs). If
  # training a single model is short (<5 minutes), using the Vizier
  # tuner will take most of the computation time (in this example, the dataset
  # contains 30k examples and takes ~3s seconds to train). In such case, using
  # the ydf.RandomSearchTuner instead of ydf.VizierTuner might be more
  # efficient.
  tuner = ydf.VizierTuner(
      # Total number of hyper-parameter configurations to test.
      num_trials=200,
      # Total number of trials to run in parallel. It is best to use a multiple
      # of the number of workers.
      parallel_trials=20,
      # You can specify manually the search space, or use the preconfigured one
      # with automatic_search_space=True.
      automatic_search_space=True,
  )

  # Configure the learning.
  learner = ydf.GradientBoostedTreesLearner(
      label="income",
      # Number of threads to train each individual model.
      num_threads=20,
      # The working directory contains the model checkpoints. Change the working
      # directory when training a new model.
      working_dir=os.path.join(_PROJECT_DIR.value, "work_dir"),
      # Non optimized hyper-parameters can be set here.
      num_trees=200,
      # Give the tuner.
      tuner=tuner,
  )
  if _WORKER_BNS.value is not None:
    learner.set_worker_bns(_WORKER_BNS.value, _NUM_WORKERS.value)

  # Tune the model
  model = learner.train(
      "recordio+tfe:/cns/is-d/home/gbm/ml_dataset_repository/others/adult/adult_train.riotfe",
      # If you have a validation dataset, set it here. If you don't provide a
      # validation dataset, 10% (by default) of the training dataset will be
      # used for validation
      # valid=... path to validation dataset
  )

  # Save the model to disk.
  model.save(os.path.join(_PROJECT_DIR.value, "model"))

  # Evaluate the model.
  # Note: The model evaluation is not distributed.
  evaluation = model.evaluate(
      "recordio+tfe:/cns/is-d/home/gbm/ml_dataset_repository/others/adult/adult_test.riotfe"
  )
  logging.info("Evaluation:\n%s", evaluation)


if __name__ == "__main__":
  app.run(main)
