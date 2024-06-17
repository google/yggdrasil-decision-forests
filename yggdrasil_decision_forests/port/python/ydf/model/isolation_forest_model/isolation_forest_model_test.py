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

"""Tests for the isolation forest models."""

import logging
import os

from absl.testing import absltest
import numpy as np
import pandas as pd

from ydf.model import model_lib
from ydf.utils import test_utils


class IsolationForestModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    def build_path(*args):
      return os.path.join(test_utils.ydf_test_data_path(), *args)

    self.model_gaussians = model_lib.load_model(
        build_path("model", "gaussians_anomaly_if")
    )
    self.dataset_gaussians_train = pd.read_csv(
        build_path("dataset", "gaussians_train.csv")
    )
    self.dataset_gaussians_test = pd.read_csv(
        build_path("dataset", "gaussians_test.csv")
    )

  def test_predict(self):
    predictions = self.model_gaussians.predict(self.dataset_gaussians_test)
    np.testing.assert_allclose(
        predictions[:5],
        [0.419287, 0.441436, 0.507164, 0.425276, 0.386438],
        atol=0.0001,
    )

  def test_distance(self):
    distances = self.model_gaussians.distance(self.dataset_gaussians_test)
    logging.info("distances:\n%s", distances)
    self.assertEqual(
        distances.shape,
        (
            self.dataset_gaussians_test.shape[0],
            self.dataset_gaussians_test.shape[0],
        ),
    )

    # Find the example most similar to "self.dataset_gaussians_test[0]".
    most_similar_example_idx = np.argmin(distances[0, :])
    logging.info("most_similar_example_idx: %s", most_similar_example_idx)
    logging.info("Seed example:\n%s", self.dataset_gaussians_test.iloc[0])
    logging.info(
        "Most similar example:\n%s",
        self.dataset_gaussians_test.iloc[most_similar_example_idx],
    )


if __name__ == "__main__":
  absltest.main()
