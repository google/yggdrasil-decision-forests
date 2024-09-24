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

"""Definitions for Gradient Boosted Trees models."""

import math
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

from yggdrasil_decision_forests.metric import metric_pb2
from yggdrasil_decision_forests.model.gradient_boosted_trees import gradient_boosted_trees_pb2
from ydf.cc import ydf
from ydf.learner import custom_loss
from ydf.metric import metric
from ydf.model.decision_forest_model import decision_forest_model


class GradientBoostedTreesModel(decision_forest_model.DecisionForestModel):
  """A Gradient Boosted Trees model for prediction and inspection."""

  _model: ydf.GradientBoostedTreesCCModel

  def validation_loss(self) -> Optional[float]:
    """Returns loss on the validation dataset if available."""
    loss = self._model.validation_loss()
    return loss if not math.isnan(loss) else None

  def initial_predictions(self) -> npt.NDArray[float]:
    """Returns the model's initial predictions (i.e. the model bias)."""
    return self._model.initial_predictions()

  def set_initial_predictions(self, initial_predictions: Sequence[float]):
    """Sets the model's initial predictions (i.e. the model bias)."""
    return self._model.set_initial_predictions(
        np.asarray(initial_predictions, np.float32)
    )

  def validation_evaluation(self) -> Optional[metric.Evaluation]:
    """Returns the validation evaluation of the model, if available.

    Gradient Boosted Trees use a validation dataset for early stopping.

    Returns None if no validation evaluation been computed or it has been
    removed from the model.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train model
    train_ds = pd.read_csv("train.csv")
    model = ydf.GradientBoostedTreesLearner(label="label").train(train_ds)

    validation_evaluation = model.validation_evaluation()
    # In an interactive Python environment, print a rich evaluation report.
    validation_evaluation
    ```
    """
    validation_evaluation_proto = self._model.validation_evaluation()
    # There is no canonical way of checking if a proto is empty. This workaround
    # just checks if the evaluation proto is valid.
    if not validation_evaluation_proto.HasField("task"):
      return None
    return metric.Evaluation(self._model.validation_evaluation())

  def self_evaluation(self) -> Optional[metric.Evaluation]:
    """Returns the model's self-evaluation.

    For Gradient Boosted Trees models, the self-evaluation is the evaluation on
    the validation dataset. Note that the validation dataset is extracted
    automatically if not explicitly given. If the validation dataset is
    deactivated, no self-evaluation is computed.

    Different models use different methods for self-evaluation. Notably, Random
    Forests use the last Out-Of-Bag evaluation. Therefore, self-evaluations are
    not comparable between different model types.

    Returns None if no self-evaluation has been computed.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train model
    train_ds = pd.read_csv("train.csv")
    model = ydf.GradientBoostedTreesLearner(label="label").train(train_ds)

    self_evaluation = model.self_evaluation()
    # In an interactive Python environment, print a rich evaluation report.
    self_evaluation
    ```
    """
    return self.validation_evaluation()

  def num_trees_per_iteration(self) -> int:
    """The number of trees trained per gradient boosting iteration."""

    return self._model.num_trees_per_iter()

  def activation(self) -> custom_loss.Activation:
    """The model activation function."""
    loss = self._model.loss()
    if loss in [
        gradient_boosted_trees_pb2.Loss.BINOMIAL_LOG_LIKELIHOOD,
        gradient_boosted_trees_pb2.Loss.BINARY_FOCAL_LOSS,
    ]:
      return custom_loss.Activation.SIGMOID
    elif loss == gradient_boosted_trees_pb2.Loss.MULTINOMIAL_LOG_LIKELIHOOD:
      return custom_loss.Activation.SOFTMAX
    elif loss in [
        gradient_boosted_trees_pb2.Loss.SQUARED_ERROR,
        gradient_boosted_trees_pb2.Loss.LAMBDA_MART_NDCG,
        gradient_boosted_trees_pb2.Loss.LAMBDA_MART_NDCG5,
        gradient_boosted_trees_pb2.Loss.XE_NDCG_MART,
        gradient_boosted_trees_pb2.Loss.POISSON,
        gradient_boosted_trees_pb2.Loss.MEAN_AVERAGE_ERROR,
    ]:
      return custom_loss.Activation.IDENTITY
    else:
      raise ValueError(f"No activation registered for loss: {loss!r}")
