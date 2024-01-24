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
from typing import Optional

import numpy.typing as npt

from ydf.cc import ydf
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

  def validation_evaluation(self) -> metric.Evaluation:
    """Returns the validation evaluation of the model, if available.

    Gradient Boosted Trees use a validation dataset for early stopping.

    If no validation evaluation been computed or has been removed, the
    evaluation object may be missing fields.

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
    return metric.Evaluation(self._model.validation_evaluation())
