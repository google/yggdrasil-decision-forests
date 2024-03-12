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

"""Node in a tree."""

import abc
import dataclasses
import functools
from typing import Any, Dict, Optional
from yggdrasil_decision_forests.dataset import data_spec_pb2
from ydf.model.tree import condition as condition_lib
from ydf.model.tree import value as value_lib

# Number of spaces printed on the left side of nodes with pretty print.
_PRETTY_MARGIN = 4
# Length / number of characters (e.g. "-") in an edge with pretty print.
_PRETTY_EDGE_LENGTH = 4


class AbstractNode(metaclass=abc.ABCMeta):

  @property
  @abc.abstractmethod
  def is_leaf(self) -> bool:
    """Tells if a node is a leaf."""
    raise NotImplementedError

  @abc.abstractmethod
  def pretty(
      self,
      dataspec: data_spec_pb2.DataSpecification,
      prefix: str,
      is_pos: Optional[bool],
      depth: int,
      max_depth: Optional[int],
  ) -> str:
    raise NotImplementedError


@dataclasses.dataclass
class Leaf(AbstractNode):
  value: value_lib.AbstractValue

  @property
  def is_leaf(self) -> bool:
    return True

  def pretty(
      self,
      dataspec: data_spec_pb2.DataSpecification,
      prefix: str,
      is_pos: Optional[bool],
      depth: int,
      max_depth: Optional[int],
  ) -> str:
    return prefix + _pretty_local_prefix(is_pos) + self.value.pretty() + "\n"


@dataclasses.dataclass
class NonLeaf(AbstractNode):
  value: Optional[value_lib.AbstractValue] = None
  condition: Optional[condition_lib.AbstractCondition] = None
  pos_child: Optional[AbstractNode] = None
  neg_child: Optional[AbstractNode] = None

  @property
  def is_leaf(self) -> bool:
    return False

  def pretty(
      self,
      dataspec: data_spec_pb2.DataSpecification,
      prefix: str,
      is_pos: Optional[bool],
      depth: int,
      max_depth: Optional[int],
  ) -> str:

    # Prefix for the children of this node.
    children_prefix = prefix
    if is_pos is None:
      pass
    elif is_pos:
      children_prefix += " " * _PRETTY_MARGIN + "│" + " " * _PRETTY_EDGE_LENGTH
    elif not is_pos:
      children_prefix += " " * (_PRETTY_MARGIN + 1 + _PRETTY_EDGE_LENGTH)

    # Node's condition.
    condition_prefix = prefix + _pretty_local_prefix(is_pos)
    if self.condition is not None:
      condition_prefix += self.condition.pretty(dataspec)
    else:
      condition_prefix += "No condition"
    condition_prefix += "\n"

    # Children of the node.
    if max_depth is not None and depth >= max_depth:
      return condition_prefix + children_prefix + "...\n"
    else:
      children_text = condition_prefix
      if self.pos_child is not None:
        children_text += self.pos_child.pretty(
            dataspec, children_prefix, True, depth + 1, max_depth
        )
      else:
        children_text += "No positive child\n"
      if self.neg_child is not None:
        children_text += self.neg_child.pretty(
            dataspec, children_prefix, False, depth + 1, max_depth
        )
      else:
        children_text += "No negative child\n"
      return children_text


@functools.singledispatch
def to_json(
    node: AbstractNode,
    depth: int,
    max_depth: Optional[int],
    dataspec: data_spec_pb2.DataSpecification,
) -> Dict[str, Any]:
  """Creates a JSON-compatible object of the node.

  Note: While public, this logic is not part of the API. This is why this
  methode's code is not an abstract method in AbstractValue.

  Args:
    node: Input node.
    depth: Depth of the current node
    max_depth: Maximum depth of the tree in the json
    dataspec: Dataspec associated with this tree.

  Returns:
    JSON node.
  """
  raise NotImplementedError("Unsupported node type")


@to_json.register
def _to_json_leaf(
    node: Leaf,
    depth: int,
    max_depth: Optional[int],
    dataspec: data_spec_pb2.DataSpecification,
) -> Dict[str, Any]:
  return {"value": value_lib.to_json(node.value)}


@to_json.register
def _to_json_non_leaf(
    node: NonLeaf,
    depth: int,
    max_depth: Optional[int],
    dataspec: data_spec_pb2.DataSpecification,
) -> Dict[str, Any]:
  dst = {}
  if node.value is not None:
    dst["value"] = value_lib.to_json(node.value)

  if node.condition is not None:
    dst["condition"] = condition_lib.to_json(node.condition, dataspec)
  if (
      (max_depth is None or (max_depth is not None and depth < max_depth))
      and node.pos_child is not None
      and node.neg_child is not None
  ):
    dst["children"] = [
        to_json(node.pos_child, depth + 1, max_depth, dataspec),
        to_json(node.neg_child, depth + 1, max_depth, dataspec),
    ]
  return dst


def _pretty_local_prefix(is_pos: Optional[bool]) -> str:
  """Prefix added in front of a node with pretty print.

  Args:
    is_pos: True/False if the node is a positive/negative child. None if the
      node is a root.

  Returns:
    The node prefix.
  """

  if is_pos is None:
    # Root node. No prefix.
    return ""
  elif is_pos:
    # Positive nodes are assumed to be printed before negative ones.
    return " " * _PRETTY_MARGIN + "├─(pos)─ "
  else:
    return " " * _PRETTY_MARGIN + "└─(neg)─ "
