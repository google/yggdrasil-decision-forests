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

"""A decision tree."""

import dataclasses
from typing import Iterator, Optional, Sequence
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2
from ydf.model.tree import condition as condition_lib
from ydf.model.tree import node as node_lib
from ydf.model.tree import value as value_lib


@dataclasses.dataclass
class Tree:
  root: node_lib.AbstractNode

  def pretty(
      self,
      dataspec: data_spec_pb2.DataSpecification,
      max_depth: Optional[int] = 6,
  ) -> str:
    """Returns a printable representation of the decision tree.

    Usage example:

    ```python
    model = ydf.load_model("my_model")
    tree = model.get_tree(0)
    print(tree.pretty(model.data_spec()))
    ```

    Args:
      dataspec: Dataspec of the tree.
      max_depth: Maximum printed depth.
    """

    if self.root:
      return self.root.pretty(
          dataspec=dataspec,
          prefix="",
          is_pos=None,
          depth=1,
          max_depth=max_depth,
      )
    else:
      return "No root\n"


def _recusive_build_node(
    node_iterator: Iterator[decision_tree_pb2.Node],
    dataspec: data_spec_pb2.DataSpecification,
) -> node_lib.AbstractNode:
  """Creates recursively a node from a node iterator.

  The nodes should be produced as a depth-first, negative-first, transversal
  order.

  Args:
    node_iterator: Node iterator.
    dataspec: Model dataspec.

  Returns:
    The root node.
  """

  proto_node = next(node_iterator)
  if proto_node.HasField("condition"):
    # If the non-leaf contains a value
    if proto_node.WhichOneof("output") is not None:
      value = value_lib.to_value(proto_node)
    else:
      value = None

    return node_lib.NonLeaf(
        value=value,
        condition=condition_lib.to_condition(proto_node.condition, dataspec),
        neg_child=_recusive_build_node(node_iterator, dataspec),
        pos_child=_recusive_build_node(node_iterator, dataspec),
    )
  else:
    return node_lib.Leaf(value=value_lib.to_value(proto_node))


def proto_nodes_to_tree(
    nodes: Sequence[decision_tree_pb2.Node],
    dataspec: data_spec_pb2.DataSpecification,
) -> Tree:
  """Creates a tree from an depth-first, negative-first list of nodes."""
  return Tree(root=_recusive_build_node(iter(nodes), dataspec))
