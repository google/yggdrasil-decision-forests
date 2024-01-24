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

"""Definitions for Random Forest models."""

import dataclasses
from typing import Optional, Sequence
from yggdrasil_decision_forests.metric import metric_pb2
from yggdrasil_decision_forests.model.random_forest import random_forest_pb2
from ydf.cc import ydf
from ydf.metric import metric
from ydf.model.decision_forest_model import decision_forest_model


@dataclasses.dataclass(frozen=True)
class OutOfBagEvaluation:
  """A collection of out-of-bag metrics.

  Attributes:
    number_of_trees: Number of trees when the evaluation was created.
    evaluation: Rich evaluation object containing the OOB evaluation metrics.
  """

  number_of_trees: int
  evaluation: metric.Evaluation


class RandomForestModel(decision_forest_model.DecisionForestModel):
  """A Random Forest model for prediction and inspection."""

  _model: ydf.RandomForestCCModel

  def out_of_bag_evaluations(self) -> Sequence[OutOfBagEvaluation]:
    """Returns the Out-Of-Bag evaluations of the model, if available.

    Each tree in a random forest is only trained on a fraction of the training
    examples. Out-of-bag (OOB) evaluations evaluate each training example on the
    trees that have not seen it in training. This creates a self-evaluation
    method that does not require a training dataset. See
    https://developers.google.com/machine-learning/decision-forests/out-of-bag
    for details.

    Computing OOB metrics slows down training and requires hyperparameter
    `compute_oob_performances` to be set. The learner then computes the OOB
    evaluation at regular intervals during the training. The returned list of
    evaluations is sorted by the number of trees and its last element is the OOB
    evaluation of the full model.

    If no OOB evaluations have been computed, an empty list is returned.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train model
    train_ds = pd.read_csv("train.csv")
    learner = ydf.RandomForestLearner(label="label",
                                      compute_oob_performances=True)
    model = learner.train(train_ds)

    oob_evaluations = model.out_of_bag_evaluations()
    # In an interactive Python environment, print a rich evaluation report.
    oob_evaluations[-1].evaluation
    ```
    """
    raw_evaluations: Sequence[random_forest_pb2.OutOfBagTrainingEvaluations] = (
        self._model.out_of_bag_evaluations()
    )
    return [
        OutOfBagEvaluation(
            number_of_trees=evaluation_proto.number_of_trees,
            evaluation=metric.Evaluation(evaluation_proto.evaluation),
        )
        for evaluation_proto in raw_evaluations
    ]

  def winner_takes_all(self) -> bool:
    """Returns if the model uses a winner-takes-all strategy for classification.

    This parameter determines how to aggregate individual tree votes during
    inference in a classification random forest. It is defined by the
    `winner_take_all` Random Forest learner hyper-parameter,

    If true, each tree votes for a single class, which is the traditional random
    forest inference method. If false, each tree outputs a probability
    distribution across all classes.

    If the model is not a classification model, the return value of this
    function is arbitrary and does not influence model inference.
    """
    return self._model.winner_takes_all()

  def self_evaluation(self) -> Optional[metric.Evaluation]:
    """Returns the model's self-evaluation.

    For Random Forest models, the self-evaluation is out-of-bag evaluation on
    the full model. Note that the Random Forest models do not use a validation
    dataset. If out-of-bag evaluation is not enabled, no self-evaluation is
    computed.

    Different models use different methods for self-evaluation. Notably,
    Gradient Boosted Trees use the evaluation on the validation dataset.
    Therefore, self-evaluations are not comparable between different model
    types.

    Returns None if no self-evaluation has been computed.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train model
    train_ds = pd.read_csv("train.csv")
    learner = ydf.RandomForestLearner(label="label",
                                    compute_oob_performances=True)
    model = learner.train(train_ds)

    self_evaluation = model.self_evaluation()
    # In an interactive Python environment, print a rich evaluation report.
    self_evaluation
    ```
    """
    oob_evaluation = self.out_of_bag_evaluations()
    if oob_evaluation:
      return oob_evaluation[-1].evaluation
    # Return an empty evaluation object.
    return None
