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
from typing import Optional
from ydf.model.tree import condition as condition_lib
from ydf.model.tree import value as value_lib


class AbstractNode(metaclass=abc.ABCMeta):

  @property
  @abc.abstractmethod
  def is_leaf(self) -> bool:
    """Tells if a node is a leaf."""
    raise NotImplementedError


@dataclasses.dataclass
class Leaf(AbstractNode):
  value: value_lib.AbstractValue

  @property
  def is_leaf(self) -> bool:
    return True


@dataclasses.dataclass
class NonLeaf(AbstractNode):
  value: Optional[value_lib.AbstractValue] = None
  condition: Optional[condition_lib.AbstractCondition] = None
  pos_child: Optional[AbstractNode] = None
  neg_child: Optional[AbstractNode] = None

  @property
  def is_leaf(self) -> bool:
    return False
