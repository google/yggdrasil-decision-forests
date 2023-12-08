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
from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2
from ydf.model.tree import value as value_lib
from ydf.utils import test_utils
from yggdrasil_decision_forests.utils import distribution_pb2


class ValueTest(absltest.TestCase):

  def test_to_value_classifier_given_valid_input(self):
    proto_node = decision_tree_pb2.Node(
        classifier=decision_tree_pb2.NodeClassifierOutput(
            distribution=distribution_pb2.IntegerDistributionDouble(
                counts=[0.0, 8.0, 2.0], sum=10.0
            )
        )
    )
    self.assertEqual(
        value_lib.to_value(proto_node),
        value_lib.ProbabilityValue(probability=[0.8, 0.2], num_examples=10),
    )

  def test_to_value_regressor_given_valid_input(self):
    proto_node = decision_tree_pb2.Node(
        regressor=decision_tree_pb2.NodeRegressorOutput(
            top_value=1,
            distribution=distribution_pb2.NormalDistributionDouble(
                sum=10, sum_squares=20, count=10
            ),
        )
    )
    self.assertEqual(
        value_lib.to_value(proto_node),
        value_lib.RegressionValue(
            value=1.0, num_examples=10, standard_deviation=1.0
        ),
    )

  def test_to_value_uplift_given_valid_input(self):
    proto_node = decision_tree_pb2.Node(
        uplift=decision_tree_pb2.NodeUpliftOutput(
            treatment_effect=[0.0, 8.0, 2.0], sum_weights=10
        )
    )
    self.assertEqual(
        value_lib.to_value(proto_node),
        value_lib.UpliftValue(
            treatment_effect=[0.0, 8.0, 2.0], num_examples=10.0
        ),
    )

  def test_classifier_proto_node_is_set_given_valid_input(self):
    proto_node = decision_tree_pb2.Node()
    value_lib.set_proto_node(
        value_lib.ProbabilityValue(probability=[0.8, 0.2], num_examples=10),
        proto_node,
    )
    test_utils.assertProto2Equal(
        self,
        proto_node,
        decision_tree_pb2.Node(
            classifier=decision_tree_pb2.NodeClassifierOutput(
                top_value=1,
                distribution=distribution_pb2.IntegerDistributionDouble(
                    counts=[0.0, 8.0, 2.0], sum=10.0
                ),
            )
        ),
    )

  def test_regressive_proto_node_is_set_given_valid_input(self):
    proto_node = decision_tree_pb2.Node()
    value_lib.set_proto_node(
        value_lib.RegressionValue(
            value=1.0, num_examples=10, standard_deviation=1.0
        ),
        proto_node,
    )
    test_utils.assertProto2Equal(
        self,
        proto_node,
        decision_tree_pb2.Node(
            regressor=decision_tree_pb2.NodeRegressorOutput(
                top_value=1,
                distribution=distribution_pb2.NormalDistributionDouble(
                    sum=10, sum_squares=20, count=10
                ),
            )
        ),
    )

  def test_uplift_proto_node_is_set_given_valid_input(self):
    proto_node = decision_tree_pb2.Node()
    value_lib.set_proto_node(
        value_lib.UpliftValue(
            treatment_effect=[0.0, 8.0, 2.0], num_examples=10.0
        ),
        proto_node,
    )
    test_utils.assertProto2Equal(
        self,
        proto_node,
        decision_tree_pb2.Node(
            uplift=decision_tree_pb2.NodeUpliftOutput(
                treatment_effect=[0.0, 8.0, 2.0], sum_weights=10
            )
        ),
    )

  def test_pretty_classification(self):
    self.assertEqual(
        value_lib.ProbabilityValue(
            probability=[0.8, 0.2], num_examples=10
        ).pretty(),
        "value=[0.8, 0.2]",
    )

  def test_pretty_regression(self):
    self.assertEqual(
        value_lib.RegressionValue(
            value=1.0, num_examples=10, standard_deviation=1.0
        ).pretty(),
        "value=1 sd=1",
    )

  def test_pretty_uplift(self):
    self.assertEqual(
        value_lib.UpliftValue(
            treatment_effect=[0.0, 8.0, 2.0], num_examples=10.0
        ).pretty(),
        "value=[0.0, 8.0, 2.0]",
    )


if __name__ == "__main__":
  absltest.main()
