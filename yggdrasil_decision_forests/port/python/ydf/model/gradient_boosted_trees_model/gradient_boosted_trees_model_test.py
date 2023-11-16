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
from ydf.utils import test_utils


class GradientBoostedTreesTest(absltest.TestCase):

  def test_validation_loss(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    model = model_lib.load_model(model_path)

    validation_loss = model.validation_loss()
    self.assertAlmostEqual(validation_loss, 0.573842942, places=6)

  def test_initial_predictions(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    model = model_lib.load_model(model_path)

    initial_predictions = model.initial_predictions()
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
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "adult_binary_class_gbdt",
    )
    model = model_lib.load_model(model_path)

    dataset = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
        ),
        nrows=500,
    )

    distances = model.distance(dataset)
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


if __name__ == "__main__":
  absltest.main()
