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

"""Tests for the YDF models."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from yggdrasil_decision_forests.model.random_forest import random_forest_pb2
from ydf.model import generic_model
from ydf.model import model_lib
from ydf.model.decision_forest_model import decision_forest_model
from ydf.utils import test_utils


class DecisionForestModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Loading models needed in many unittests.
    model_dir = os.path.join(test_utils.ydf_test_data_path(), "model")
    # This model is a Random Forest classification model without training logs.
    self.adult_binary_class_rf: decision_forest_model.DecisionForestModel = (
        model_lib.load_model(os.path.join(model_dir, "adult_binary_class_rf"))
    )
    # This model is a GBDT classification model without training logs.
    self.adult_binary_class_gbdt: decision_forest_model.DecisionForestModel = (
        model_lib.load_model(os.path.join(model_dir, "adult_binary_class_gbdt"))
    )
    # This model is a GBDT regression model without training logs.
    self.abalone_regression_gbdt: decision_forest_model.DecisionForestModel = (
        model_lib.load_model(os.path.join(model_dir, "abalone_regression_gbdt"))
    )
    # This model is a RF uplift model
    self.sim_pte_categorical_uplift_rf: (
        decision_forest_model.DecisionForestModel
    ) = model_lib.load_model(
        os.path.join(model_dir, "sim_pte_categorical_uplift_rf")
    )

  def test_num_trees(self):
    self.assertEqual(self.adult_binary_class_rf.num_trees(), 100)

  def test_predict_leaves(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    dataset = pd.read_csv(dataset_path)

    leaves = self.adult_binary_class_gbdt.predict_leaves(dataset)
    self.assertEqual(
        leaves.shape,
        (dataset.shape[0], self.adult_binary_class_gbdt.num_trees()),
    )
    self.assertTrue(np.all(leaves >= 0))

  def test_print_api(self):
    self.adult_binary_class_gbdt.print_tree()
    self.adult_binary_class_gbdt.print_tree(tree_idx=0, max_depth=None)

  def test_plot_api(self):
    self.adult_binary_class_gbdt.plot_tree().html()
    self.adult_binary_class_gbdt.plot_tree(tree_idx=0, max_depth=None).html()

  @parameterized.parameters(x for x in generic_model.NodeFormat)
  def test_node_format(self, node_format: generic_model.NodeFormat):
    """Test that the node format is saved correctly."""
    self.adult_binary_class_rf.set_node_format(node_format=node_format)
    model_save_path = self.create_tempdir().full_path
    self.adult_binary_class_rf.save(
        model_save_path,
        advanced_options=generic_model.ModelIOOptions(file_prefix=""),
    )
    # Read the proto to see if the format is set correctly
    # TODO: Consider exposing the proto directly in ydf.
    random_forest_header = random_forest_pb2.Header()
    random_forest_header_path = os.path.join(
        model_save_path, "random_forest_header.pb"
    )
    self.assertTrue(os.path.exists(random_forest_header_path))
    with open(random_forest_header_path, "rb") as f:
      random_forest_header.ParseFromString(f.read())
    self.assertEqual(random_forest_header.node_format, node_format.name)

  def test_plot_classification(self):
    plot = self.adult_binary_class_gbdt.plot_tree()
    root_as_html = (
        '{"value": {"type": "REGRESSION", "value": -4.158827948685939e-09,'
        ' "num_examples": 0.0}, "condition": {"type": "CATEGORICAL_IS_IN",'
        ' "attribute": "marital_status", "mask": ["Never-married", "Divorced",'
        ' "Widowed", "Separated", "Married-spouse-absent",'
        ' "Married-AF-spouse"]}'
    )
    self.assertIn(root_as_html, plot.html())

  def test_plot_regression(self):
    plot = self.abalone_regression_gbdt.plot_tree()
    root_as_html = (
        '{"value": {"type": "REGRESSION", "value": -4.225819338898873e-08,'
        ' "num_examples": 2663.0, "standard_deviation": 3.227639690862905},'
        ' "condition": {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute":'
        ' "ShellWeight", "threshold": 0.1537500023841858}'
    )
    self.assertIn(root_as_html, plot.html())

  def test_plot_uplift(self):
    plot = self.sim_pte_categorical_uplift_rf.plot_tree()
    root_as_html = (
        '{"value": {"type": "UPLIFT", "treatment_effect":'
        ' [-0.019851017743349075], "num_examples": 1000.0}, "condition":'
        ' {"type": "NUMERICAL_IS_HIGHER_THAN", "attribute": "X16", "threshold":'
        " 1.782151699066162}"
    )
    self.assertIn(root_as_html, plot.html())


if __name__ == "__main__":
  absltest.main()
