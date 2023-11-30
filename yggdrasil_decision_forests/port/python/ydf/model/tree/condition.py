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
from typing import Sequence, Union
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2

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


@dataclasses.dataclass
class IsMissingInCondition(AbstractCondition):
  """Condition of the form "attribute is missing".

  Attrs:
    attribute: Attribute (or one of the attributes) tested by the condition.
  """

  attribute: int


def to_condition(
    proto_condition: decision_tree_pb2.NodeCondition,
    dataspec: data_spec_pb2.DataSpecification,
) -> AbstractCondition:
  """Extracts the "condition" part of a proto node."""

  del dataspec  # dataspec will be used in other cases.

  base_args = {
      "missing": proto_condition.na_value,
      "score": proto_condition.split_score,
  }
  condition_type = proto_condition.condition

  if condition_type.HasField("na_condition"):
    return IsMissingInCondition(
        attribute=proto_condition.attribute, **base_args
    )

  else:
    raise ValueError(f"Non supported condition type: {proto_condition}")


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
