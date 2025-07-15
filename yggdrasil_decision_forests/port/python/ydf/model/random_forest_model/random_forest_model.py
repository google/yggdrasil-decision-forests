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
from typing import List, Optional, Sequence, Tuple
from yggdrasil_decision_forests.model.random_forest import random_forest_pb2
from ydf.cc import ydf
from ydf.metric import metric
from ydf.model import generic_model
from ydf.model.decision_forest_model import decision_forest_model


class RandomForestModel(decision_forest_model.DecisionForestModel):
  """A Random Forest model for prediction and inspection."""

  _model: ydf.RandomForestCCModel

  def out_of_bag_evaluations(self) -> Sequence[generic_model.TrainingLogEntry]:
    """Alias for `training_logs()` for Random Forest models."""
    return self.training_logs()

  def training_logs(self) -> List[generic_model.TrainingLogEntry]:
    """Returns the Out-of-Bag evaluation logs for the Random Forest model.

      For Random Forests, the training logs contain performance metrics
      calculated periodically during training using the Out-of-Bag (OOB) data.
      Each tree in a random forest is trained on a bootstrap sample of the
      training data. The OOB evaluation uses each training example as a test
      case for the subset of trees that were not trained on it. This method
      provides an unbiased estimate of the model's performance without requiring
      a separate validation set.

      To generate these logs, the `compute_oob_performances` hyperparameter must
      be set to `True` (which is the default). Please note that enabling this
      can slightly slow down training.

      The OOB evaluation is not computed after every single tree. Instead, the
      learner calculates it periodically when one of the following is true:
        - The most recently trained tree is the final tree of the model.
        - More than 10 seconds have passed since the last OOB evaluation.
        - More than 10 trees have been trained since the last OOB evaluation.

      The returned list of `TrainingLogEntry` objects is sorted by iteration,
      allowing you to easily plot the model's learning curve. The training
      iteration is equal to the number of trees when the model was trained. The
      last entry in the list represents the final OOB evaluation for the fully
      trained model.

      For more details, see the [explanation of OOB
      evaluation](https://developers.google.com/machine-learning/decision-forests/out-of-bag).

      Random Forest models do not return a `training_evalution`.

      For CART models, the training logs have a single entry, containing the
      evaluation on the validation dataset.

      Usage example:

      ```python
      import pandas as pd
      import ydf

      # Train model
      train_ds = pd.read_csv("train.csv")
      model = ydf.RandomForestLearner(label="label").train(train_ds)

      # Get the training logs
      logs = model.training_logs()

      # Plot the accuracy.
      plt.plot(
          [log.iteration for log in logs],
          [log.evaluation.accuracy for log in logs]
      )
      ```

    Returns:
      A list of `TrainingLogEntry` objects, each containing the OOB evaluation
      metrics and the number of trees in the model at that point in training.
      Returns an empty list if logs were not generated.
    """
    raw_evaluations: Sequence[random_forest_pb2.OutOfBagTrainingEvaluations] = (
        self._model.out_of_bag_evaluations()
    )
    return [
        generic_model.TrainingLogEntry(
            iteration=evaluation_proto.number_of_trees,
            evaluation=metric.Evaluation(evaluation_proto.evaluation),
            training_evaluation=None,
        )
        for evaluation_proto in raw_evaluations
    ]

  def winner_takes_all(self) -> bool:
    """Returns if the model uses a winner-takes-all strategy for classification.

    This parameter determines how to aggregate individual tree votes during
    inference in a classification random forest. It is defined by the
    `winner_take_all` Random Forest learner hyper-parameter.

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
