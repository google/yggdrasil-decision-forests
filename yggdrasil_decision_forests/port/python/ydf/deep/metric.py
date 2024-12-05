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

"""Utils to compute metrics and losses."""

import abc
import jax
import jax.numpy as jnp
import optax


class Metric(abc.ABC):
  """Abstract class for metrics."""

  @abc.abstractmethod
  def name(self) -> str:
    """Long name of a metric."""
    raise NotImplementedError

  @abc.abstractmethod
  def short_name(self) -> str:
    """Short name of a metric."""
    raise NotImplementedError

  @abc.abstractmethod
  def __call__(self, labels: jax.Array, preds: jax.Array) -> jax.Array:
    """Computes a metric value."""
    raise NotImplementedError


class AccuracyBinaryClassificationMetric(Metric):

  def name(self) -> str:
    return "accuracy binary-class classification"

  def short_name(self) -> str:
    return "accuracy"

  def __call__(self, labels: jax.Array, preds: jax.Array) -> jax.Array:
    assert len(labels.shape) == 1
    assert len(preds.shape) == 1
    z = (preds >= 0.0) == labels
    assert len(z.shape) == 1
    return z.mean()


class LossBinaryClassificationMetric(Metric):

  def name(self) -> str:
    return "cross-entropy loss for binary classification"

  def short_name(self) -> str:
    return "cross-entropy"

  def __call__(self, labels: jax.Array, preds: jax.Array) -> jax.Array:
    assert len(labels.shape) == 1
    assert len(preds.shape) == 1
    z = optax.sigmoid_binary_cross_entropy(preds, labels)
    assert len(z.shape) == 1
    return z.mean()


class MeanSquaredErrorMetric(Metric):

  def name(self) -> str:
    return "mean squared error"

  def short_name(self) -> str:
    return "mse"

  def __call__(self, labels: jax.Array, preds: jax.Array) -> jax.Array:
    assert len(labels.shape) == 1
    assert len(preds.shape) == 1
    z = optax.squared_error(preds, labels)
    assert len(z.shape) == 1
    return z.mean()


class AccuracyMultiClassClassificationMetric(Metric):

  def __init__(self, num_classes: int):
    self._num_classes = num_classes

  def name(self) -> str:
    return "accuracy multi-class classification"

  def short_name(self) -> str:
    return "accuracy"

  def __call__(self, labels: jax.Array, preds: jax.Array) -> jax.Array:
    assert len(labels.shape) == 1
    assert len(preds.shape) == 2
    pred_classes = jnp.argmax(preds, axis=-1)
    assert len(pred_classes.shape) == 1
    z = pred_classes == labels
    assert len(z.shape) == 1
    return z.mean()


class LossMultiClassClassificationMetric(Metric):

  def __init__(self, num_classes: int):
    self._num_classes = num_classes

  def name(self) -> str:
    return "cross-entropy loss for multi-class classification"

  def short_name(self) -> str:
    return "cross-entropy"

  def __call__(self, labels: jax.Array, preds: jax.Array) -> jax.Array:
    assert len(labels.shape) == 1
    assert len(preds.shape) == 2
    z = optax.softmax_cross_entropy_with_integer_labels(preds, labels)
    assert len(z.shape) == 1
    return z.mean()
