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
from ydf.utils import test_utils


class DecisionForestModelTest(parameterized.TestCase):

  def test_num_trees(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_rf"
    )
    model = model_lib.load_model(model_path)
    self.assertEqual(model.num_trees(), 100)

  def test_predict_leaves(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "adult_binary_class_gbdt",
    )
    model = model_lib.load_model(model_path)

    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    dataset = pd.read_csv(dataset_path)

    leaves = model.predict_leaves(dataset)
    self.assertEqual(leaves.shape, (dataset.shape[0], model.num_trees()))
    self.assertTrue(np.all(leaves >= 0))

  @parameterized.parameters(x for x in generic_model.NodeFormat)
  def test_node_format(self, node_format: generic_model.NodeFormat):
    """Test that the node format is saved correctly."""
    model_load_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "adult_binary_class_rf",
    )
    model = model_lib.load_model(model_load_path)
    model.set_node_format(node_format=node_format)
    model_save_path = self.create_tempdir().full_path
    model.save(
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


if __name__ == "__main__":
  absltest.main()
