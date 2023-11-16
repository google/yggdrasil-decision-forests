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

  def test_oob_evaluations(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_rf"
    )
    model = model_lib.load_model(model_path)
    # TODO: Fill this test when OOB evaluations are exposed.
    with self.assertRaises(NotImplementedError):
      model.out_of_bag_evaluation()

  def test_predict_distance(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "adult_binary_class_rf",
    )
    model = model_lib.load_model(model_path)

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

    distances = model.distance(dataset1, dataset2)
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


if __name__ == "__main__":
  absltest.main()
