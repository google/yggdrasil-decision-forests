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

"""Implementations for custom metric containers."""

import dataclasses
from typing import Callable

import numpy as np
import numpy.typing as npt

from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.cc import ydf


@dataclasses.dataclass(frozen=True)
class AbstractCustomMetric:
  """Abstract Base class for custom metrics."""

  name: str

  def check_is_compatible_task(self, task: abstract_model_pb2.Task) -> None:
    raise NotImplementedError("Not implemented")

  def _to_cc(self):
    raise NotImplementedError("Not implemented")

  def __post_init__(self):
    if not isinstance(self.name, str):
      raise ValueError(
          "The name of a custom metric must be a string. Received"
          f" {type(self.name)}"
      )


@dataclasses.dataclass(frozen=True)
class RegressionMetric(AbstractCustomMetric):
  """A user-provided secondary metric for regression problems.

  Evaluation functions may never reference their arguments outside after
  returning:
  Bad:
  ```
  mylabels = None
  def evaluation_func(labels, predictions, weights):
    nonlocal mylabels
    mylabels = labels  # labels is now referenced outside the function
  ```
  Good:
  ```
  mylabels = None
  def evaluation_func(labels, predictions, weights):
    nonlocal mylabels
    mylabels = np.copy(labels)  # mylabels is a copy, not a reference.
  ```

  Attributes:
    name: The name of the custom metric. This will be displayed in the training
      logs and made available as secondary metric in the model.
    evaluation_func: The evaluation function receives the predictions (without
      activation function), labels, and weights, and must output the metric
      value as a float.
  """

  evaluation_func: Callable[
      [
          npt.NDArray[np.float32],
          npt.NDArray[np.float32],
          npt.NDArray[np.float32],
      ],
      np.float32,
  ]

  def check_is_compatible_task(self, task: abstract_model_pb2.Task) -> None:
    """Raises an error if the given task is incompatible with this metric type."""
    if task != abstract_model_pb2.REGRESSION:
      raise ValueError(
          "A RegressionMetric is only compatible with REGRESSION"
          f" tasks. Received task {abstract_model_pb2.Task.Name(task)}"
      )

  def _to_cc(self):
    return ydf.CCRegressionMetric(self.name, self.evaluation_func)


@dataclasses.dataclass(frozen=True)
class BinaryClassificationMetric(AbstractCustomMetric):
  """A user-provided secondary metric for binary classification problems.

  Note that the labels are binary but 1-based, i.e. the positive class is 2, the
  negative class is 1.

  Evaluation functions may never reference their arguments outside after
  returning:
  Bad:
  ```
  mylabels = None
  def evaluation_func(labels, predictions, weights):
    nonlocal mylabels
    mylabels = labels  # labels is now referenced outside the function
  ```
  Good:
  ```
  mylabels = None
  def evaluation_func(labels, predictions, weights):
    nonlocal mylabels
    mylabels = np.copy(labels)  # mylabels is a copy, not a reference.
  ```

  Attributes:
    name: The name of the custom metric. This will be displayed in the training
      logs and made available as secondary metric in the model.
    evaluation_func: The evaluation function receives the predictions (without
      activation function), binary labels, and weights, and must output the
      metric value as a float.
  """

  evaluation_func: Callable[
      [
          npt.NDArray[np.int32],
          npt.NDArray[np.float32],
          npt.NDArray[np.float32],
      ],
      np.float32,
  ]

  def check_is_compatible_task(self, task: abstract_model_pb2.Task) -> None:
    """Raises an error if the given task is incompatible with this metric type."""
    if task != abstract_model_pb2.CLASSIFICATION:
      raise ValueError(
          "A BinaryClassificationMetric is only compatible with CLASSIFICATION"
          f" tasks. Received task {abstract_model_pb2.Task.Name(task)}"
      )

  def _to_cc(self):
    return ydf.CCBinaryClassificationMetric(self.name, self.evaluation_func)


@dataclasses.dataclass(frozen=True)
class MultiClassificationMetric(AbstractCustomMetric):
  """A user-provided secondary metric for multi-class problems.

  Note that the labels are 1-based. Predictions are given in an 2D
  array with one row per example.

  Evaluation functions may never reference their arguments outside after
  returning:
  Bad:
  ```
  mylabels = None
  def evaluation_func(labels, predictions, weights):
    nonlocal mylabels
    mylabels = labels  # labels is now referenced outside the function
  ```
  Good:
  ```
  mylabels = None
  def evaluation_func(labels, predictions, weights):
    nonlocal mylabels
    mylabels = np.copy(labels)  # mylabels is a copy, not a reference.
  ```

  Attributes:
    name: The name of the custom metric. This will be displayed in the training
      logs and made available as secondary metric in the model.
    evaluation_func: The evaluation function receives the predictions (2D array,
      without activation function), labels (1-based), and weights, and must
      output the metric value as a float.
  """

  evaluation_func: Callable[
      [
          npt.NDArray[np.int32],
          npt.NDArray[np.float32],
          npt.NDArray[np.float32],
      ],
      np.float32,
  ]

  def check_is_compatible_task(self, task: abstract_model_pb2.Task) -> None:
    """Raises an error if the given task is incompatible with this metric type."""
    if task != abstract_model_pb2.CLASSIFICATION:
      raise ValueError(
          "A MultiClassificationMetric is only compatible with CLASSIFICATION"
          f" tasks. Received task {abstract_model_pb2.Task.Name(task)}"
      )

  def _to_cc(self):
    return ydf.CCMultiClassificationMetric(self.name, self.evaluation_func)
