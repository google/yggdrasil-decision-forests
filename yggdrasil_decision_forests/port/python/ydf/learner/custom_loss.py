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

"""Implementations for custom loss containers."""

import dataclasses
import enum
from typing import Any, Callable, Tuple

import numpy as np
import numpy.typing as npt

from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.cc import ydf


class Activation(enum.Enum):
  """Activation functions for custom losses.

  Not all activation functions are supported for all custom losses. Activation
  function IDENTITY (i.e., no activation function applied) is always supported.
  """

  IDENTITY = "IDENTITY"
  SIGMOID = "SIGMOID"
  SOFTMAX = "SOFTMAX"


@dataclasses.dataclass(frozen=True)
class AbstractCustomLoss:
  """Abstract Base class for custom losses."""

  activation: Any

  initial_predictions: Callable[
      [npt.NDArray[Any], npt.NDArray[np.float32]],
      Any,
  ]
  loss: Callable[
      [
          npt.NDArray[Any],
          npt.NDArray[np.float32],
          npt.NDArray[np.float32],
      ],
      np.float32,
  ]
  gradient_and_hessian: Callable[
      [npt.NDArray[Any], npt.NDArray[np.float32]],
      Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
  ]

  may_trigger_gc: bool = True

  def check_is_compatible_task(self, task: abstract_model_pb2.Task) -> None:
    raise NotImplementedError("Not implemented")

  def _to_cc(self):
    raise NotImplementedError("Not implemented")


@dataclasses.dataclass(frozen=True)
class RegressionLoss(AbstractCustomLoss):
  """A user-provided loss function for regression problems.

  Loss functions may never reference their arguments outside after returning:
  Bad:
  ```
  mylabels = None
  def initial_predictions(labels, weights):
    nonlocal mylabels
    mylabels = labels  # labels is now referenced outside the function
  ```
  Good:
  ```
  mylabels = None
  def initial_predictions(labels, weights):
    nonlocal mylabels
    mylabels = np.copy(labels)  # mylabels is a copy, not a reference.
  ```

  Attributes:
    initial_predictions: The bias / initial predictions of the GBT model.
      Receives the label values and the weights, outputs the initial prediction
      as a float.
    loss: The loss function controls the early stopping. The loss function
      receives the labels, the current predictions and the current weights and
      must output the loss as a float. Note that the predictions provided to the
      loss functions have not yet had an activation function applied to them.
    gradient_and_hessian: Gradient and hessian of the current predictions. Note
      that only the diagonal of the hessian must be provided. Receives as input
      the labels and the current predictions (without activation) and returns a
      tuple of the gradient and the hessian.
    activation: Activation function to be applied to the model. Regression
      models are expected to return a value in the same space as the labels
      after applying the activation function.
    may_trigger_gc: If True (default), YDF may trigger Python's garbage
      collection to determine if a Numpy array that is backed by YDF-internal
      data is used after its lifetime has ended. If False, checks for illegal
      memory accesses are disabled. This can be useful when training many small
      models or if the observed impact of triggering GC is large. If
      `may_trigger_gc=False`, it is very important that the user validate
      manuallythat no memory leakage occurs.
  """

  initial_predictions: Callable[
      [npt.NDArray[np.float32], npt.NDArray[np.float32]],
      np.float32,
  ]
  loss: Callable[
      [
          npt.NDArray[np.float32],
          npt.NDArray[np.float32],
          npt.NDArray[np.float32],
      ],
      np.float32,
  ]
  gradient_and_hessian: Callable[
      [npt.NDArray[np.float32], npt.NDArray[np.float32]],
      Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
  ]

  activation: Activation

  def __post_init__(self):
    if self.activation != Activation.IDENTITY:
      raise ValueError(
          "Only activation function IDENTITY is supported for RegressionLoss."
      )

  def check_is_compatible_task(self, task: abstract_model_pb2.Task) -> None:
    """Raises an error if the given task is incompatible with this loss type."""
    if task != abstract_model_pb2.REGRESSION:
      raise ValueError(
          "A RegressionLoss is only compatible with REGRESSION"
          f" tasks. Received task {abstract_model_pb2.Task.Name(task)}"
      )

  def _to_cc(self):
    return ydf.CCRegressionLoss(
        self.initial_predictions,
        self.loss,
        self.gradient_and_hessian,
        self.may_trigger_gc,
    )


@dataclasses.dataclass(frozen=True)
class BinaryClassificationLoss(AbstractCustomLoss):
  """A user-provided loss function for binary classification problems.

  Note that the labels are binary but 1-based, i.e. the positive class is 2, the
  negative class is 1.

  Loss functions may never reference their arguments outside after returning:
  Bad:
  ```
  mylabels = None
  def initial_predictions(labels, weights):
    nonlocal mylabels
    mylabels = labels  # labels is now referenced outside the function
  ```
  Good:
  ```
  mylabels = None
  def initial_predictions(labels, weights):
    nonlocal mylabels
    mylabels = np.copy(labels)  # mylabels is a copy, not a reference.
  ```

  Attributes:
    initial_predictions: The bias / initial predictions of the GBT model.
      Receives the label values and the weights, outputs the initial prediction
      as a float.
    loss: The loss function controls the early stopping. The loss function
      receives the labels, the current predictions and the current weights and
      must output the loss as a float. Note that the predictions provided to the
      loss functions have not yet had an activation function applied to them.
    gradient_and_hessian: Gradient and hessian of the current predictions. Note
      that only the diagonal of the hessian must be provided. Receives as input
      the labels and the current predictions (without activation). Returns a
      tuple of the gradient and the hessian.
    activation: Activation function to be applied to the model. Binary
      classification models are expected to return a probability after applying
      the activation function.
    may_trigger_gc: If True (default), YDF may trigger Python's garbage
      collection to determine if an Numpy array that is backed by YDF-internal
      data is used after its lifetime has ended. If False, checks for illegal
      memory accesses are disabled. Setting this parameter to False is
      dangerous, since illegal memory accesses will no longer be detected.
  """

  initial_predictions: Callable[
      [npt.NDArray[np.int32], npt.NDArray[np.float32]],
      np.float32,
  ]
  loss: Callable[
      [
          npt.NDArray[np.int32],
          npt.NDArray[np.float32],
          npt.NDArray[np.float32],
      ],
      np.float32,
  ]
  gradient_and_hessian: Callable[
      [npt.NDArray[np.int32], npt.NDArray[np.float32]],
      Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
  ]

  activation: Activation

  def __post_init__(self):
    if self.activation not in [Activation.IDENTITY, Activation.SIGMOID]:
      raise ValueError(
          "Only activation functions IDENTITY and SIGMOID are supported for"
          " BinaryClassificationLoss."
      )

  def check_is_compatible_task(self, task: abstract_model_pb2.Task) -> None:
    """Raises an error if the given task is incompatible with this loss type."""
    if task != abstract_model_pb2.CLASSIFICATION:
      raise ValueError(
          "A BinaryClassificationLoss is only compatible with CLASSIFICATION"
          f" tasks. Received task {abstract_model_pb2.Task.Name(task)}"
      )

  def _to_cc(self):
    return ydf.CCBinaryClassificationLoss(
        self.initial_predictions,
        self.loss,
        self.gradient_and_hessian,
        self.may_trigger_gc,
    )


@dataclasses.dataclass(frozen=True)
class MultiClassificationLoss(AbstractCustomLoss):
  """A user-provided loss function for multi-class problems.

  Note that the labels are but 1-based. Predictions are given in an 2D
  array with one row per example. Initial predictions, gradient and
  hessian are expected for each class, e.g. for a 3-class classification
  problem, output 3 gradients and hessians per class.

  Loss functions may never reference their arguments outside after returning:
  Bad:
  ```
  mylabels = None
  def initial_predictions(labels, weights):
    nonlocal mylabels
    mylabels = labels  # labels is now referenced outside the function
  ```
  Good:
  ```
  mylabels = None
  def initial_predictions(labels, weights):
    nonlocal mylabels
    mylabels = np.copy(labels)  # mylabels is a copy, not a reference.
  ```

  Attributes:
    initial_predictions: The bias / initial predictions of the GBT model.
      Receives the label values and the weights, outputs the initial prediction
      as an array of floats (one initial prediction per class).
    loss: The loss function controls the early stopping. The loss function
      receives the labels, the current predictions and the current weights and
      must output the loss as a float. Note that the predictions provided to the
      loss functions have not yet had an activation function applied to them.
    gradient_and_hessian: Gradient and hessian of the current predictions with
      respect to each class. Note that only the diagonal of the hessian must be
      provided. Receives as input the labels and the current predictions
      (without activation). Returns a tuple of the gradient and the hessian.
      Both gradient and hessian must be arrays of shape (num_classes,
      num_examples).
    activation: Activation function to be applied to the model. Multi-class
      classification models are expected to return a probability distribution
      over the classes after applying the activation function.
    may_trigger_gc: If True (default), YDF may trigger Python's garbage
      collection to determine if an Numpy array that is backed by YDF-internal
      data is used after its lifetime has ended. If False, checks for illegal
      memory accesses are disabled. Setting this parameter to False is
      dangerous, since illegal memory accesses will no longer be detected.
  """

  initial_predictions: Callable[
      [npt.NDArray[np.int32], npt.NDArray[np.float32]],
      npt.NDArray[np.float32],
  ]
  loss: Callable[
      [
          npt.NDArray[np.int32],
          npt.NDArray[np.float32],
          npt.NDArray[np.float32],
      ],
      np.float32,
  ]
  gradient_and_hessian: Callable[
      [npt.NDArray[np.int32], npt.NDArray[np.float32]],
      Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
  ]

  activation: Activation

  def __post_init__(self):
    if (
        self.activation != Activation.IDENTITY
        and self.activation != Activation.SOFTMAX
    ):
      raise ValueError(
          "Only activation functions IDENTITY and SOFTMAX are supported for"
          " MultiClassificationLoss."
      )

  def check_is_compatible_task(self, task: abstract_model_pb2.Task) -> None:
    """Raises an error if the given task is incompatible with this loss type."""
    if task != abstract_model_pb2.CLASSIFICATION:
      raise ValueError(
          "A MultiClassificationLoss is only compatible with CLASSIFICATION"
          f" tasks. Received task {abstract_model_pb2.Task.Name(task)}"
      )

  def _to_cc(self):
    return ydf.CCMultiClassificationLoss(
        self.initial_predictions,
        self.loss,
        self.gradient_and_hessian,
        self.may_trigger_gc,
    )
