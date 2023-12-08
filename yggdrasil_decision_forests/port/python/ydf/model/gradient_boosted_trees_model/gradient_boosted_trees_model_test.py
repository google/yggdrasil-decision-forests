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

"""Tests for the Gradient Boosted trees models."""

import os

from absl import logging
from absl.testing import absltest
import numpy as np
import pandas as pd

from ydf.model import model_lib
from ydf.model.tree import condition as condition_lib
from ydf.model.tree import value as value_lib
from ydf.utils import test_utils


class GradientBoostedTreesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    self.adult_binary_class_gbdt = model_lib.load_model(model_path)

  def test_validation_loss(self):
    validation_loss = self.adult_binary_class_gbdt.validation_loss()
    self.assertAlmostEqual(validation_loss, 0.573842942, places=6)

  def test_initial_predictions(self):
    initial_predictions = self.adult_binary_class_gbdt.initial_predictions()
    np.testing.assert_allclose(initial_predictions, [-1.1630996])

  def test_variable_importances(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "synthetic_ranking_gbdt_numerical",
    )
    model = model_lib.load_model(model_path)
    variable_importances = model.variable_importances()
    self.assertEqual(
        variable_importances,
        {
            "NUM_NODES": [
                (355.0, "num_2"),
                (326.0, "num_0"),
                (248.0, "num_1"),
                (193.0, "num_3"),
            ],
            "INV_MEAN_MIN_DEPTH": [
                (0.54955206094026765, "num_0"),
                (0.43300866801748344, "num_2"),
                (0.21987296105251422, "num_1"),
                (0.20886402442940008, "num_3"),
            ],
            "SUM_SCORE": [
                (331.52462868355724, "num_0"),
                (297.70595154801595, "num_2"),
                (103.86176226850876, "num_1"),
                (52.43193327602421, "num_3"),
            ],
            "NUM_AS_ROOT": [
                (35.0, "num_0"),
                (12.0, "num_2"),
                (1.0, "num_3"),
            ],
        },
    )

  def test_predict_distance(self):
    dataset = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
        ),
        nrows=500,
    )

    distances = self.adult_binary_class_gbdt.distance(dataset)
    logging.info("distances:\n%s", distances)
    self.assertEqual(distances.shape, (dataset.shape[0], dataset.shape[0]))

    # Find in "dataset2", the example most similar to "dataset1[0]".
    most_similar_example_idx = np.argmin(distances[0, :])
    logging.info("most_similar_example_idx: %s", most_similar_example_idx)
    logging.info("Seed example:\n%s", dataset.iloc[0])
    logging.info(
        "Most similar example:\n%s", dataset.iloc[most_similar_example_idx]
    )

    # High likelihood that the labels are the same (true in this example).
    self.assertEqual(
        dataset.iloc[most_similar_example_idx]["income"],
        dataset.iloc[0]["income"],
    )

  def test_model_inspector_get_valid_tree(self):
    self.assertEqual(self.adult_binary_class_gbdt.num_trees(), 68)
    self.assertLen(
        list(self.adult_binary_class_gbdt.get_all_trees()),
        self.adult_binary_class_gbdt.num_trees(),
    )

    tree = self.adult_binary_class_gbdt.get_tree(1)
    self.assertFalse(tree.root.is_leaf)
    # Validated with: external/ydf_cc/yggdrasil_decision_forests/cli:show_model
    self.assertEqual(
        tree.root.condition,
        condition_lib.CategoricalIsInCondition(
            missing=False,
            score=3275.003662109375,
            attribute=5,
            mask=[2, 3, 4, 5, 6, 7],
        ),
    )
    self.assertEqual(
        tree.root.value,
        value_lib.RegressionValue(
            value=-0.0006140652694739401, num_examples=0.0
        ),
    )

  def test_model_inspector_get_wrong_tree(self):
    with self.assertRaisesRegex(ValueError, "Invalid tree index"):
      _ = self.adult_binary_class_gbdt.get_tree(-1)
    with self.assertRaisesRegex(ValueError, "Invalid tree index"):
      _ = self.adult_binary_class_gbdt.get_tree(
          self.adult_binary_class_gbdt.num_trees()
      )

  def test_model_inspector_print_tree(self):
    tree = self.adult_binary_class_gbdt.get_tree(1)
    test_utils.golden_check_string(
        self,
        tree.pretty(self.adult_binary_class_gbdt.data_spec()),
        os.path.join(test_utils.pydf_test_data_path(), "adult_gbt_tree_0.txt"),
    )


if __name__ == "__main__":
  absltest.main()
