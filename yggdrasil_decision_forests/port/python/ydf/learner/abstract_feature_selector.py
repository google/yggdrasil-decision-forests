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

"""Configures the automatic selection of the input features."""

import abc
import dataclasses
from typing import Any, Dict, Optional, Tuple, Union

from ydf.dataset import dataset
from ydf.metric import metric as metric_lib
from ydf.model import generic_model


@dataclasses.dataclass
class AbstractFeatureSelector(abc.ABC):
  """Base class for feature selectors.

  Attributes:
    objective_metric: Name of the metric optimized by the feature selection.
      Should be one of the keys of `model.self_evaluation().to_dict()` or
      `model.evaluation(...)`. If None, a reasonable `objective_metric` is
      selected automatically.
    maximize_objective: If true, `objective_metric` will be maximized. If False,
      `objective_metric` will be minimized. Should be None if and only if
      `objective_metric` is None.
  """

  objective_metric: Optional[str] = None
  maximize_objective: Optional[bool] = None

  @abc.abstractmethod
  def run(
      self,
      learner: Any,  # generic_learner_lib.GenericLearner
      ds: dataset.InputDataset,
      valid: Optional[dataset.InputDataset] = None,
      verbose: Optional[Union[int, bool]] = None,
  ) -> generic_model.ModelType:
    """Runs the feature selector."""

  def get_objective_score(
      self, evaluation: Dict[str, Any]
  ) -> Tuple[float, str]:
    """Gets the score of a metric to optimize. The greater the better."""

    if self.objective_metric:
      # The user specified the metric to optimize.
      if self.objective_metric not in evaluation:
        raise ValueError(
            f"The metric {self.objective_metric} is unknown. The available"
            f" metrics are {evaluation!r}"
        )
      return (
          evaluation[self.objective_metric]
          * (1.0 if self.maximize_objective else -1.0),
          self.objective_metric,
      )

    if self.maximize_objective is not None:
      raise ValueError(
          "maximize_objective should be None if objective_metric is None"
      )

    for candidate_name, maximize_candidate in [
        ("loss", False),
        ("accuracy", True),
        ("rmse", False),
        ("ndcg", True),
        ("auuc", True),
    ]:
      if candidate_name in evaluation:
        return (
            evaluation[candidate_name] * (1.0 if maximize_candidate else -1.0),
            candidate_name,
        )

    raise ValueError(
        f"No metric to optimizer. The available metrics are {evaluation!r}"
    )
