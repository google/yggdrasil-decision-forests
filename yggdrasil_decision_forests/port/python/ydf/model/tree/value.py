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

"""The value / prediction of a leaf."""

import abc
import dataclasses
import functools
import math
from typing import Any, Dict, Optional, Sequence
import numpy as np
from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2


# TODO: 310218604 - Use kw_only with default value num_examples = 1.
@dataclasses.dataclass
class AbstractValue(metaclass=abc.ABCMeta):
  """A generic value/prediction/output.

  Attrs:
    num_examples: Number of examples in the node with weight.
  """

  num_examples: float

  @abc.abstractmethod
  def pretty(self) -> str:
    raise NotImplementedError

  def __str__(self):
    return self.pretty()


@dataclasses.dataclass
class RegressionValue(AbstractValue):
  """The regression value of a regressive tree.

  Can also be used in gradient-boosted-trees for classification and ranking.

  Attrs:
    value: Value of the tree. The semantic depends on the tree: For Regression
      Random Forest and Regression GBDT, this value is a regressive value in the
      same unit as the label. For classification and ranking GBDTs, this value
      is a logit.
    standard_deviation: Optional standard deviation attached to the value.
  """

  value: float
  standard_deviation: Optional[float] = None

  def pretty(self) -> str:
    text = f"value={self.value:.5g}"
    if self.standard_deviation is not None:
      text += f" sd={self.standard_deviation:.5g}"
    return text


@dataclasses.dataclass
class ProbabilityValue(AbstractValue):
  """A probability distribution value.

  Used for random Forest / CART classification trees.

  Attrs:
    probability: An array of probabilities of the label classes i.e. the i-th
      value is the probability of the "label_value_idx_to_value(..., i)" class.
      Note that the first value is reserved for the Out-of-vocabulary
  """

  probability: Sequence[float]

  def pretty(self) -> str:
    return f"value={self.probability}"


@dataclasses.dataclass
class UpliftValue(AbstractValue):
  """The uplift value of a classification or regression uplift tree.

  Attrs:
    treatment_effect: An array of the effects on the treatment groups. The i-th
      element of this array is the effect of the "i+1"th treatment compared to
      the control group.
  """

  treatment_effect: Sequence[float]

  def pretty(self) -> str:
    return f"value={self.treatment_effect}"


@dataclasses.dataclass
class AnomalyDetectionValue(AbstractValue):
  """The value of an anomaly detection tree.

  Attrs:
    num_examples_without_weight: Number of examples reaching this node.
  """

  num_examples_without_weight: int

  def pretty(self) -> str:
    return f"count={self.num_examples_without_weight}"


def to_value(proto_node: decision_tree_pb2.Node) -> AbstractValue:
  """Extracts the "value" part of a proto node."""

  if proto_node.HasField("classifier"):
    dist = proto_node.classifier.distribution
    # Note: The first value (out-of-dictionary) is removed.
    probabilities = np.array(dist.counts[1:]) / dist.sum
    return ProbabilityValue(
        probability=probabilities.tolist(), num_examples=dist.sum
    )

  if proto_node.HasField("regressor"):
    dist = proto_node.regressor.distribution
    standard_deviation = None
    if dist.HasField("sum_squares") and dist.count > 0:
      variance = dist.sum_squares / dist.count - dist.sum**2 / dist.count**2
      if variance >= 0:
        standard_deviation = math.sqrt(variance)
    return RegressionValue(
        value=proto_node.regressor.top_value,
        num_examples=dist.count,
        standard_deviation=standard_deviation,
    )

  if proto_node.HasField("uplift"):
    return UpliftValue(
        treatment_effect=proto_node.uplift.treatment_effect[:],
        num_examples=proto_node.uplift.sum_weights,
    )

  if proto_node.HasField("anomaly_detection"):
    return AnomalyDetectionValue(
        num_examples_without_weight=proto_node.anomaly_detection.num_examples_without_weight,
        num_examples=-1.0,  # The number of weighted examples is not tracked.
    )

  raise ValueError("Unsupported value")


@functools.singledispatch
def to_json(value: AbstractValue) -> Dict[str, Any]:
  """Creates a JSON-compatible object of the value.

  Note: While public, this logic is not part of the API. This is why this
  methode's code is not an abstract method in AbstractValue.

  Args:
    value: Input value.

  Returns:
    JSON value.
  """
  raise NotImplementedError("Unsupported value type")


@to_json.register
def _to_json_regression(value: RegressionValue) -> Dict[str, Any]:
  value_as_json = {
      "type": "REGRESSION",
      "value": value.value,
      "num_examples": value.num_examples,
  }
  if value.standard_deviation is not None:
    value_as_json["standard_deviation"] = value.standard_deviation
  return value_as_json


@to_json.register
def _to_json_probability(value: ProbabilityValue) -> Dict[str, Any]:
  return {
      "type": "PROBABILITY",
      "distribution": value.probability,
      "num_examples": value.num_examples,
  }


@to_json.register
def _to_json_uplift(value: UpliftValue) -> Dict[str, Any]:
  return {
      "type": "UPLIFT",
      "treatment_effect": value.treatment_effect,
      "num_examples": value.num_examples,
  }


@to_json.register
def _to_json_uplift(value: AnomalyDetectionValue) -> Dict[str, Any]:
  return {
      "type": "ANOMALY_DETECTION",
      "num_examples_without_weight": value.num_examples_without_weight,
      "num_examples": value.num_examples,
  }


@functools.singledispatch
def set_proto_node(value: AbstractValue, proto_node: decision_tree_pb2.Node):
  """Sets the "value" part in a proto node.

  Note: While public, this logic is not part of the API. This is why this
  methode's code is not an abstract method in AbstractValue.

  Args:
    value: Input value.
    proto_node: Proto node to populate with the input value.
  """
  del value
  del proto_node
  raise NotImplementedError("Unsupported value type")


@set_proto_node.register
def _set_proto_node_from_probability(
    value: ProbabilityValue, proto_node: decision_tree_pb2.Node
):
  dist = proto_node.classifier.distribution
  dist.sum = value.num_examples
  # Add an extra 0 for the out-of-vocabulary item.
  dist.counts[:] = np.array([0.0, *value.probability]) * dist.sum
  proto_node.classifier.top_value = np.argmax(dist.counts)


@set_proto_node.register
def _set_proto_node_from_regression(
    value: RegressionValue, proto_node: decision_tree_pb2.Node
):
  proto_node.regressor.top_value = value.value
  if value.standard_deviation is not None:
    dist = proto_node.regressor.distribution
    dist.count = value.num_examples
    dist.sum = value.value * value.num_examples
    dist.sum_squares = (
        value.standard_deviation**2 * value.num_examples
        + dist.sum**2 / value.num_examples
    )


@set_proto_node.register
def _set_proto_node_from_uplift(
    value: UpliftValue, proto_node: decision_tree_pb2.Node
):
  proto_node.uplift.treatment_effect[:] = value.treatment_effect
  proto_node.uplift.sum_weights = value.num_examples


@set_proto_node.register
def _set_proto_node_from_anomaly_detection(
    value: AnomalyDetectionValue, proto_node: decision_tree_pb2.Node
):
  proto_node.anomaly_detection.num_examples_without_weight = (
      value.num_examples_without_weight
  )
