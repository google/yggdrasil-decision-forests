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

import os
import re
from absl.testing import absltest
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2
from ydf.model.tree import condition as condition_lib
from ydf.model.tree import node as node_lib
from ydf.model.tree import tree as tree_lib
from ydf.model.tree import value as value_lib
from ydf.utils import test_utils

RegressionValue = value_lib.RegressionValue
Leaf = node_lib.Leaf
NonLeaf = node_lib.NonLeaf
IsMissingInCondition = condition_lib.IsMissingInCondition
NumericalHigherThanCondition = condition_lib.NumericalHigherThanCondition
Tree = tree_lib.Tree
ProtoNode = decision_tree_pb2.Node
ProtoNodeCondition = decision_tree_pb2.NodeCondition
ProtoCondition = decision_tree_pb2.Condition
ProtoNodeRegressorOutput = decision_tree_pb2.NodeRegressorOutput


class TreeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.dataspec = data_spec_pb2.DataSpecification(
        columns=[
            data_spec_pb2.Column(
                name="f1", type=data_spec_pb2.ColumnType.NUMERICAL
            )
        ]
    )
    self.nodes = [
        ProtoNode(
            condition=ProtoNodeCondition(
                attribute=0,
                condition=ProtoCondition(
                    higher_condition=ProtoCondition.Higher(threshold=2)
                ),
                split_score=3.0,
                na_value=False,
            ),
        ),
        ProtoNode(
            condition=ProtoNodeCondition(
                attribute=0,
                condition=ProtoCondition(
                    higher_condition=ProtoCondition.Higher(threshold=4)
                ),
                split_score=5.0,
                na_value=False,
            ),
            # Non leaf with a value
            regressor=ProtoNodeRegressorOutput(top_value=9.0),
        ),
        ProtoNode(regressor=ProtoNodeRegressorOutput(top_value=6.0)),
        ProtoNode(regressor=ProtoNodeRegressorOutput(top_value=7.0)),
        ProtoNode(regressor=ProtoNodeRegressorOutput(top_value=8.0)),
    ]
    self.tree = Tree(
        root=NonLeaf(
            condition=NumericalHigherThanCondition(
                missing=False, score=3.0, attribute=0, threshold=2.0
            ),
            pos_child=Leaf(value=RegressionValue(num_examples=0.0, value=8.0)),
            neg_child=NonLeaf(
                value=RegressionValue(num_examples=0.0, value=9.0),
                condition=NumericalHigherThanCondition(
                    missing=False, score=5.0, attribute=0, threshold=4.0
                ),
                pos_child=Leaf(
                    value=RegressionValue(num_examples=0.0, value=7.0)
                ),
                neg_child=Leaf(
                    value=RegressionValue(num_examples=0.0, value=6.0)
                ),
            ),
        )
    )

  def test_valid_input(self):
    Tree(
        root=NonLeaf(
            condition=IsMissingInCondition(missing=True, score=3, attribute=0),
            pos_child=Leaf(value=RegressionValue(value=1, num_examples=1)),
            neg_child=Leaf(value=RegressionValue(value=2, num_examples=1)),
        )
    )

  def test_proto_nodes_to_tree_with_valid_input(self):
    tree = tree_lib.proto_nodes_to_tree(self.nodes, self.dataspec)
    self.assertEqual(tree, self.tree)
    test_utils.golden_check_string(
        self,
        tree.pretty(self.dataspec),
        os.path.join(test_utils.pydf_test_data_path(), "toy_tree.txt"),
    )

  def test_tree_to_proto_nodes_with_valid_input(self):
    nodes = tree_lib.tree_to_proto_nodes(self.tree, self.dataspec)
    self.assertEqual(nodes, self.nodes)

  def test_plot_toy_tree(self):
    tree_plot = self.tree.plot(
        self.dataspec, max_depth=None, label_classes=None
    )
    raw_output = tree_plot.html()
    pattern = r"tree_plot_[a-f|\d]{32}"
    reproducible_html = re.sub(pattern, "tree_plot_element", raw_output)
    reproducible_html = reproducible_html.replace(
        """/*
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

""",
        "",
    )
    test_utils.golden_check_string(
        self,
        reproducible_html,
        os.path.join(
            test_utils.pydf_test_data_path(), "golden_toy_tree_plot.txt"
        ),
    )


if __name__ == "__main__":
  absltest.main()
