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

from absl.testing import absltest
from ydf.model.tree import condition as condition_lib
from ydf.model.tree import node as node_lib
from ydf.model.tree import value as value_lib

RegressionValue = value_lib.RegressionValue
Leaf = node_lib.Leaf
NonLeaf = node_lib.NonLeaf
IsMissingInCondition = condition_lib.IsMissingInCondition


class NodeTest(absltest.TestCase):

  def test_valid_leaf(self):
    Leaf(value=RegressionValue(value=5, num_examples=1))

  def test_valid_non_leaf(self):
    NonLeaf(
        condition=IsMissingInCondition(missing=True, score=3, attribute=0),
        pos_child=Leaf(value=RegressionValue(value=1, num_examples=1)),
        neg_child=Leaf(value=RegressionValue(value=2, num_examples=1)),
    )


if __name__ == "__main__":
  absltest.main()
