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

"""Tests for the random forest models."""

import logging
import os

from absl.testing import absltest
import numpy as np
import pandas as pd

from ydf.model import model_lib
from ydf.utils import test_utils


class RandomForestModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_rf"
    )
    self.adult_binary_class_rf = model_lib.load_model(model_path)

  def test_out_of_bag_evaluations(self):
    oob_evaluations = self.adult_binary_class_rf.out_of_bag_evaluations()

    self.assertLen(oob_evaluations, 2)
    self.assertEqual(oob_evaluations[0].number_of_trees, 1)
    self.assertAlmostEqual(oob_evaluations[0].evaluation.loss, 1.80617348178)
    self.assertEqual(oob_evaluations[1].number_of_trees, 100)
    self.assertAlmostEqual(oob_evaluations[1].evaluation.loss, 0.31474323732)

  def test_empty_out_of_bag_evaluations(self):
    # Uplift models do not have OOB evaluations.
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "sim_pte_categorical_uplift_rf",
    )
    model = model_lib.load_model(model_path)

    oob_evaluations = model.out_of_bag_evaluations()

    self.assertEmpty(oob_evaluations)

  def test_predict_distance(self):
    dataset1 = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
        ),
        nrows=500,
    )
    dataset2 = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_train.csv"
        ),
        nrows=800,
    )

    distances = self.adult_binary_class_rf.distance(dataset1, dataset2)
    logging.info("distances:\n%s", distances)
    self.assertEqual(distances.shape, (dataset1.shape[0], dataset2.shape[0]))

    # Find in "dataset2", the example most similar to "dataset1[0]".
    most_similar_example_idx = np.argmin(distances[0, :])
    logging.info("most_similar_example_idx: %s", most_similar_example_idx)
    logging.info("Seed example:\n%s", dataset1.iloc[0])
    logging.info(
        "Most similar example:\n%s", dataset2.iloc[most_similar_example_idx]
    )

    # High likelihood that the labels are the same (true in this example).
    self.assertEqual(
        dataset2.iloc[most_similar_example_idx]["income"],
        dataset1.iloc[0]["income"],
    )

  def test_winner_takes_all_false(self):
    self.assertFalse(self.adult_binary_class_rf.winner_takes_all())

  def test_winner_takes_all_true(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "golden",
        "rf_adult_base",
    )
    model = model_lib.load_model(model_path)

    self.assertTrue(model.winner_takes_all())


if __name__ == "__main__":
  absltest.main()
