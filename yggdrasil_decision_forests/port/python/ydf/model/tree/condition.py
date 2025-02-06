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

"""Conditions / splits for non-leaf nodes."""

import abc
import dataclasses
import functools
from typing import Any, Dict, Sequence, Union
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2
from ydf.dataset import dataspec as dataspec_lib

ColumnType = data_spec_pb2.ColumnType


# TODO: b/310218604 - Use kw_only with default value score = 0.
@dataclasses.dataclass
class AbstractCondition(metaclass=abc.ABCMeta):
  """Generic condition.

  Attrs:
    missing: Result of the evaluation of the condition if the input feature is
      missing.
    score: Score of a condition. The semantic depends on the learning algorithm.
  """

  missing: bool
  score: float

  @abc.abstractmethod
  def pretty(self, dataspec: data_spec_pb2.DataSpecification) -> str:
    raise NotImplementedError

  def _tag(self) -> str:
    return f"score={self.score:.5g} missing={self.missing}"


@dataclasses.dataclass
class IsMissingInCondition(AbstractCondition):
  """Condition of the form "attribute is missing".

  Attrs:
    attribute: Attribute (or one of the attributes) tested by the condition.
  """

  attribute: int

  def pretty(self, dataspec: data_spec_pb2.DataSpecification) -> str:
    attribute_name = dataspec.columns[self.attribute].name
    return f"{attribute_name!r} is missing [{self._tag()}]"


@dataclasses.dataclass
class IsTrueCondition(AbstractCondition):
  """Condition of the form "attribute is true".

  Attrs:
    attribute: Attribute tested by the condition.
  """

  attribute: int

  def pretty(self, dataspec: data_spec_pb2.DataSpecification) -> str:
    attribute_name = dataspec.columns[self.attribute].name
    return f"{attribute_name!r} is True [{self._tag()}]"


@dataclasses.dataclass
class NumericalHigherThanCondition(AbstractCondition):
  """Condition of the form "attribute >= threshold".

  Attrs:
    attribute: Attribute tested by the condition.
    threshold: Threshold.
  """

  attribute: int
  threshold: float

  def pretty(self, dataspec: data_spec_pb2.DataSpecification) -> str:
    attribute_name = dataspec.columns[self.attribute].name
    return f"{attribute_name!r} >= {self.threshold:g} [{self._tag()}]"


@dataclasses.dataclass
class DiscretizedNumericalHigherThanCondition(AbstractCondition):
  """Condition of the form "attribute >= bounds[threshold]".

  Attrs:
    attribute: Attribute tested by the condition.
    threshold_idx: Index of threshold in dataspec.
  """

  attribute: int
  threshold_idx: int

  def pretty(self, dataspec: data_spec_pb2.DataSpecification) -> str:
    column_spec = dataspec.columns[self.attribute]
    threshold = column_spec.discretized_numerical.boundaries[
        self.threshold_idx - 1
    ]
    return (
        f"{column_spec.name!r} >="
        f" {threshold:g} [threshold_idx={self.threshold_idx} {self._tag()}]"
    )


@dataclasses.dataclass
class CategoricalIsInCondition(AbstractCondition):
  """Condition of the form "attribute in mask".

  Attrs:
    attribute: Attribute tested by the condition.
    mask: Sorted mask values.
  """

  attribute: int
  mask: Sequence[int]

  def pretty(self, dataspec: data_spec_pb2.DataSpecification) -> str:
    column_spec = dataspec.columns[self.attribute]
    if column_spec.categorical.is_already_integerized:
      mask_repr = list(self.mask)
    else:
      vocab = dataspec_lib.categorical_column_dictionary_to_list(column_spec)
      mask_repr = [vocab[item] for item in self.mask]
    return f"{column_spec.name!r} in {mask_repr} [{self._tag()}]"


@dataclasses.dataclass
class CategoricalSetContainsCondition(AbstractCondition):
  """Condition of the form "attribute intersect mask != empty".

  Attrs:
    attribute: Attribute tested by the condition.
    mask: Sorted mask values.
  """

  attribute: int
  mask: Sequence[int]

  def pretty(self, dataspec: data_spec_pb2.DataSpecification) -> str:
    column_spec = dataspec.columns[self.attribute]
    if column_spec.categorical.is_already_integerized:
      mask_repr = list(self.mask)
    else:
      vocab = dataspec_lib.categorical_column_dictionary_to_list(column_spec)
      mask_repr = [vocab[item] for item in self.mask]
    return f"{column_spec.name!r} intersect {mask_repr} [{self._tag()}]"


@dataclasses.dataclass
class NumericalSparseObliqueCondition(AbstractCondition):
  """Condition of the form "attributes * weights >= threshold".

  Attrs:
    attributes: Attribute tested by the condition.
    weights: Weights for each of the attributes.
    threshold: Threshold value of the condition.
  """

  attributes: Sequence[int]
  weights: Sequence[float]
  threshold: float

  def pretty(self, dataspec: data_spec_pb2.DataSpecification) -> str:
    text = " + ".join(
        f"{dataspec.columns[attribute].name!r} x {weight:g}"
        for attribute, weight in zip(self.attributes, self.weights)
    )
    if not text:
      text = "*nothing*"
    return f"{text} >= {self.threshold:g} [{self._tag()}]"


@dataclasses.dataclass
class NumericalVectorSequenceCloserThanCondition(AbstractCondition):
  """Condition of the type: exits a in Obs; |a - anchor|^2 <= threshold2.

  Attrs:
    attribute: Numerical vector sequence attribute.
    anchor: Anchor to compare to.
    threshold2: Threshold value of the condition.
  """

  attribute: int
  anchor: Sequence[float]
  threshold2: float

  def pretty(self, dataspec: data_spec_pb2.DataSpecification) -> str:
    return (
        f"{dataspec.columns[self.attribute].name!r} contains X with |X -"
        f" {self.anchor}|² <= {self.threshold2}"
    )


@dataclasses.dataclass
class NumericalVectorSequenceProjectedMoreThanCondition(AbstractCondition):
  """Condition of the type: exits a in Obs; <a|anchor> threshold.

  Attrs:
    attribute: Numerical vector sequence attribute.
    anchor: Anchor to compare to.
    threshold: Threshold value of the condition.
  """

  attribute: int
  anchor: Sequence[float]
  threshold: float

  def pretty(self, dataspec: data_spec_pb2.DataSpecification) -> str:
    return (
        f"{dataspec.columns[self.attribute].name!r} contains X with X @"
        f" {self.anchor} >= {self.threshold}"
    )


def to_condition(
    proto_condition: decision_tree_pb2.NodeCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> AbstractCondition:
  """Extracts the "condition" part of a proto node."""

  base_kwargs = {
      "missing": proto_condition.na_value,
      "score": proto_condition.split_score,
  }
  condition_type = proto_condition.condition
  attribute_type = dataspec.columns[proto_condition.attribute].type

  if condition_type.HasField("na_condition"):
    return IsMissingInCondition(
        attribute=proto_condition.attribute, **base_kwargs
    )

  elif condition_type.HasField("true_value_condition"):
    return IsTrueCondition(attribute=proto_condition.attribute, **base_kwargs)

  elif condition_type.HasField("higher_condition"):
    return NumericalHigherThanCondition(
        attribute=proto_condition.attribute,
        threshold=condition_type.higher_condition.threshold,
        **base_kwargs,
    )

  elif condition_type.HasField("contains_bitmap_condition"):
    items = bitmap_to_items(
        dataspec.columns[proto_condition.attribute],
        condition_type.contains_bitmap_condition.elements_bitmap,
    )
    if attribute_type == ColumnType.CATEGORICAL:
      return CategoricalIsInCondition(
          attribute=proto_condition.attribute,
          mask=items,
          **base_kwargs,
      )
    elif attribute_type == ColumnType.CATEGORICAL_SET:
      return CategoricalSetContainsCondition(
          attribute=proto_condition.attribute,
          mask=items,
          **base_kwargs,
      )
    else:
      raise ValueError("Invalid attribute type")

  elif condition_type.HasField("contains_condition"):
    if attribute_type == ColumnType.CATEGORICAL:
      return CategoricalIsInCondition(
          attribute=proto_condition.attribute,
          mask=condition_type.contains_condition.elements,
          **base_kwargs,
      )
    elif attribute_type == ColumnType.CATEGORICAL_SET:
      return CategoricalSetContainsCondition(
          attribute=proto_condition.attribute,
          mask=condition_type.contains_condition.elements,
          **base_kwargs,
      )
    else:
      raise ValueError("Invalid attribute type")

  elif condition_type.HasField("discretized_higher_condition"):
    return DiscretizedNumericalHigherThanCondition(
        attribute=proto_condition.attribute,
        threshold_idx=condition_type.discretized_higher_condition.threshold,
        **base_kwargs,
    )

  elif condition_type.HasField("oblique_condition"):
    return NumericalSparseObliqueCondition(
        attributes=condition_type.oblique_condition.attributes,
        weights=condition_type.oblique_condition.weights,
        threshold=condition_type.oblique_condition.threshold,
        **base_kwargs,
    )

  elif condition_type.HasField("numerical_vector_sequence"):
    if condition_type.numerical_vector_sequence.HasField("closer_than"):
      closer_than = condition_type.numerical_vector_sequence.closer_than
      return NumericalVectorSequenceCloserThanCondition(
          attribute=proto_condition.attribute,
          anchor=closer_than.anchor.grounded,
          threshold2=closer_than.threshold2,
          **base_kwargs,
      )
    elif condition_type.numerical_vector_sequence.HasField(
        "projected_more_than"
    ):
      projected_more_than = (
          condition_type.numerical_vector_sequence.projected_more_than
      )
      return NumericalVectorSequenceProjectedMoreThanCondition(
          attribute=proto_condition.attribute,
          anchor=projected_more_than.anchor.grounded,
          threshold=projected_more_than.threshold,
          **base_kwargs,
      )
    else:
      raise ValueError("Invalid attribute type")

  else:
    raise ValueError(f"Non supported condition type: {proto_condition}")


@functools.singledispatch
def to_json(
    condition: AbstractCondition, dataspec: data_spec_pb2.DataSpecification
) -> Dict[str, Any]:
  """Creates a JSON-compatible object of the condition.

  Note: While public, this logic is not part of the API. This is why this
  methode's code is not an abstract method in AbstractValue.

  Args:
    condition: Input condition.
    dataspec: Dataspec of the model.

  Returns:
    JSON condition.
  """
  raise NotImplementedError("Unsupported value type")


@to_json.register
def _to_json_is_missing(
    condition: IsMissingInCondition, dataspec: data_spec_pb2.DataSpecification
) -> Dict[str, Any]:
  attribute_name = dataspec.columns[condition.attribute].name
  return {"type": "IS_MISSING", "attribute": attribute_name}


@to_json.register
def _to_json_is_true(
    condition: IsTrueCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> Dict[str, Any]:
  attribute_name = dataspec.columns[condition.attribute].name
  return {"type": "IS_TRUE", "attribute": attribute_name}


@to_json.register
def _to_json_higher_than(
    condition: NumericalHigherThanCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> Dict[str, Any]:
  attribute_name = dataspec.columns[condition.attribute].name
  return {
      "type": "NUMERICAL_IS_HIGHER_THAN",
      "attribute": attribute_name,
      "threshold": condition.threshold,
  }


@to_json.register
def _to_json_discretized_higher_than(
    condition: DiscretizedNumericalHigherThanCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> Dict[str, Any]:
  attribute_name = dataspec.columns[condition.attribute].name
  return {
      "type": "DISCRETIZED_NUMERICAL_IS_HIGHER_THAN",
      "attribute": attribute_name,
      "threshold_idx": condition.threshold_idx,
  }


@to_json.register
def _to_json_categorical(
    condition: CategoricalIsInCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> Dict[str, Any]:
  """Returns a JSON-compatible dict for Categorical conditions."""
  attribute_name = dataspec.columns[condition.attribute].name
  column_spec = dataspec.columns[condition.attribute]
  if column_spec.categorical.is_already_integerized:
    mask_repr = list(condition.mask)
  else:
    vocab = dataspec_lib.categorical_column_dictionary_to_list(column_spec)
    mask_repr = [vocab[item] for item in condition.mask]
  return {
      "type": "CATEGORICAL_IS_IN",
      "attribute": attribute_name,
      "mask": mask_repr,
  }


@to_json.register
def _to_json_categorical_set(
    condition: CategoricalSetContainsCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> Dict[str, Any]:
  """Returns a JSON-compatible dict for CategoricalSet conditions."""
  attribute_name = dataspec.columns[condition.attribute].name
  column_spec = dataspec.columns[condition.attribute]
  if column_spec.categorical.is_already_integerized:
    mask_repr = list(condition.mask)
  else:
    vocab = dataspec_lib.categorical_column_dictionary_to_list(column_spec)
    mask_repr = [vocab[item] for item in condition.mask]
  return {
      "type": "CATEGORICAL_SET_CONTAINS",
      "attribute": attribute_name,
      "mask": mask_repr,
  }


@to_json.register
def _to_json_numerical_sparse_oblique(
    condition: NumericalSparseObliqueCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> Dict[str, Any]:
  return {
      "type": "NUMERICAL_SPARSE_OBLIQUE",
      "attributes": [dataspec.columns[f].name for f in condition.attributes],
      "weights": list(condition.weights),
      "threshold": condition.threshold,
  }


@to_json.register
def _to_json_numerical_vector_sequence_closer_than(
    condition: NumericalVectorSequenceCloserThanCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> Dict[str, Any]:
  """Returns a JSON-compatible dict for CloserThan conditions."""
  attribute_name = dataspec.columns[condition.attribute].name
  return {
      "type": "NUMERICAL_VECTOR_SEQUENCE_CLOSER_THAN",
      "attribute": attribute_name,
      "anchor": list(condition.anchor),
      "threshold2": condition.threshold2,
  }


@to_json.register
def _to_json_numerical_vector_sequence_projected_more_than(
    condition: NumericalVectorSequenceProjectedMoreThanCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> Dict[str, Any]:
  """Returns a JSON-compatible dict for ProjectedMoreThan conditions."""
  attribute_name = dataspec.columns[condition.attribute].name
  return {
      "type": "NUMERICAL_VECTOR_SEQUENCE_PROJECTED_MORE_THAN",
      "attribute": attribute_name,
      "anchor": list(condition.anchor),
      "threshold": condition.threshold,
  }


@functools.singledispatch
def to_proto_condition(
    condition: AbstractCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> decision_tree_pb2.NodeCondition:
  """Sets the "condition" part in a proto node.

  Note: While public, this logic is not part of the API. This is why this
  methode's code is not an abstract method in AbstractValue.

  Args:
    condition: Input condition.
    dataspec: Dataspec of the model.

  Returns:
    Proto condition.
  """
  raise NotImplementedError("Unsupported value type")


@to_proto_condition.register
def _to_proto_condition_is_missing(
    condition: IsMissingInCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> decision_tree_pb2.NodeCondition:
  return decision_tree_pb2.NodeCondition(
      na_value=condition.missing,
      split_score=condition.score,
      attribute=condition.attribute,
      condition=decision_tree_pb2.Condition(
          na_condition=decision_tree_pb2.Condition.NA()
      ),
  )


@to_proto_condition.register
def _to_proto_condition_is_true(
    condition: IsTrueCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> decision_tree_pb2.NodeCondition:
  return decision_tree_pb2.NodeCondition(
      na_value=condition.missing,
      split_score=condition.score,
      attribute=condition.attribute,
      condition=decision_tree_pb2.Condition(
          true_value_condition=decision_tree_pb2.Condition.TrueValue()
      ),
  )


@to_proto_condition.register
def _to_proto_condition_is_higher(
    condition: NumericalHigherThanCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> decision_tree_pb2.NodeCondition:
  return decision_tree_pb2.NodeCondition(
      na_value=condition.missing,
      split_score=condition.score,
      attribute=condition.attribute,
      condition=decision_tree_pb2.Condition(
          higher_condition=decision_tree_pb2.Condition.Higher(
              threshold=condition.threshold
          ),
      ),
  )


@to_proto_condition.register
def _to_proto_condition_discretized_is_higher(
    condition: DiscretizedNumericalHigherThanCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> decision_tree_pb2.NodeCondition:
  return decision_tree_pb2.NodeCondition(
      na_value=condition.missing,
      split_score=condition.score,
      attribute=condition.attribute,
      condition=decision_tree_pb2.Condition(
          discretized_higher_condition=decision_tree_pb2.Condition.DiscretizedHigher(
              threshold=condition.threshold_idx
          ),
      ),
  )


@to_proto_condition.register(CategoricalIsInCondition)
@to_proto_condition.register(CategoricalSetContainsCondition)
def _to_proto_condition_is_in(
    condition: Union[CategoricalIsInCondition, CategoricalSetContainsCondition],
    dataspec: data_spec_pb2.DataSpecification,
) -> decision_tree_pb2.NodeCondition:
  """Converts a "is in" condition for a categorical or categorical-set feature.

  This function selects the most compact approach (bitmap or list of items) to
  encode the condition mask.

  Args:
    condition: "is in" input condition.
    dataspec: Dataspec of the model.

  Returns:
    A proto condition.
  """

  proto_condition = decision_tree_pb2.NodeCondition(
      na_value=condition.missing,
      split_score=condition.score,
      attribute=condition.attribute,
  )
  feature_column = dataspec.columns[proto_condition.attribute]
  # Select the most efficient way to represent the mask.
  #
  # A list of indices takes 32bits per active item. A bitmap takes 1 bit per
  # item (active or not).
  if (
      len(condition.mask) * 32 * 8
      > feature_column.categorical.number_of_unique_values
  ):
    # A bitmap is more efficient.
    proto_condition.condition.contains_bitmap_condition.elements_bitmap = (
        items_to_bitmap(feature_column, condition.mask)
    )
  else:
    # A list of indices is more efficient.
    proto_condition.condition.contains_condition.elements[:] = condition.mask
  return proto_condition


@to_proto_condition.register
def _to_proto_condition_oblique(
    condition: NumericalSparseObliqueCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> decision_tree_pb2.NodeCondition:
  return decision_tree_pb2.NodeCondition(
      na_value=condition.missing,
      split_score=condition.score,
      attribute=condition.attributes[0] if condition.attributes else -1,
      condition=decision_tree_pb2.Condition(
          oblique_condition=decision_tree_pb2.Condition.Oblique(
              attributes=condition.attributes,
              weights=condition.weights,
              threshold=condition.threshold,
          ),
      ),
  )


@to_proto_condition.register
def _to_proto_condition_numerical_vector_sequence_closer_than(
    condition: NumericalVectorSequenceCloserThanCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> decision_tree_pb2.NodeCondition:
  return decision_tree_pb2.NodeCondition(
      na_value=condition.missing,
      split_score=condition.score,
      attribute=condition.attribute,
      condition=decision_tree_pb2.Condition(
          numerical_vector_sequence=decision_tree_pb2.Condition.NumericalVectorSequence(
              closer_than=decision_tree_pb2.Condition.NumericalVectorSequence.CloserThan(
                  anchor=decision_tree_pb2.Condition.NumericalVectorSequence.Anchor(
                      grounded=condition.anchor
                  ),
                  threshold2=condition.threshold2,
              )
          ),
      ),
  )


@to_proto_condition.register
def _to_proto_condition_numerical_vector_sequence_projected_more_than(
    condition: NumericalVectorSequenceProjectedMoreThanCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> decision_tree_pb2.NodeCondition:
  return decision_tree_pb2.NodeCondition(
      na_value=condition.missing,
      split_score=condition.score,
      attribute=condition.attribute,
      condition=decision_tree_pb2.Condition(
          numerical_vector_sequence=decision_tree_pb2.Condition.NumericalVectorSequence(
              projected_more_than=decision_tree_pb2.Condition.NumericalVectorSequence.ProjectedMoreThan(
                  anchor=decision_tree_pb2.Condition.NumericalVectorSequence.Anchor(
                      grounded=condition.anchor
                  ),
                  threshold=condition.threshold,
              )
          ),
      ),
  )


def _bitmap_has_item(bitmap: bytes, value: int) -> bool:
  """Checks if the "value"-th bit is set."""

  byte_idx = value // 8
  sub_bit_idx = value & 7
  return (bitmap[byte_idx] & (1 << sub_bit_idx)) != 0


def bitmap_to_items(
    column_spec: data_spec_pb2.Column, bitmap: bytes
) -> Sequence[int]:
  """Returns the list of true bits in a bitmap."""

  return [
      value_idx
      for value_idx in range(column_spec.categorical.number_of_unique_values)
      if _bitmap_has_item(bitmap, value_idx)
  ]


def items_to_bitmap(
    column_spec: data_spec_pb2.Column, items: Sequence[int]
) -> bytes:
  """Returns a bitmap with the "items"-th bits set to true.

  Setting multiple times the same bits is allowed.

  Args:
    column_spec: Column spec of a categorical column.
    items: Bit indexes.
  """

  # Note: num_bytes a rounded-up integer division between
  # p=number_of_unique_values and q=8 i.e. (p+q-1)/q.
  num_bytes = (column_spec.categorical.number_of_unique_values + 7) // 8
  # Allocate a zero-bitmap.
  bitmap = bytearray(num_bytes)

  for item in items:
    if item < 0 or item >= column_spec.categorical.number_of_unique_values:
      raise ValueError(f"Invalid item {item}")
    bitmap[item // 8] |= 1 << item % 8
  return bytes(bitmap)
