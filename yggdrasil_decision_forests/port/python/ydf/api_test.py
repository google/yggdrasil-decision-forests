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

"""API test."""


import math
import os
import tempfile

from absl import flags
from absl import logging
from absl.testing import absltest
import pandas as pd

# TODO: Replace with "import ydf"
import ydf


def data_root_path() -> str:
  return ""


def ydf_test_data_path() -> str:
  return os.path.join(
      data_root_path(), "external/ydf_cc/yggdrasil_decision_forests/test_data"
  )

class ApiTest(absltest.TestCase):

  def test_create_dataset(self):
    pd_ds = pd.DataFrame(
        {
            "c1": [1.0, 1.1, math.nan],
            "c2": [1, 2, 3],
            # "c3": [True, False, True],
            # "c4": ["x", "y", ""],
        }
    )
    ds = ydf.create_vertical_dataset(pd_ds)
    logging.info("My dataset:\n%s", ds)

  def test_load_and_save_model(self):
    model_path = os.path.join(
        ydf_test_data_path(), "model", "adult_binary_class_rf"
    )
    model = ydf.load_model(model_path)
    logging.info(model)
    with tempfile.TemporaryDirectory() as tempdir:
      model.save(tempdir, ydf.ModelIOOptions(file_prefix="model_prefix_"))
      logging.info(os.listdir(tempdir))

  def test_train_random_forest(self):
    pd_ds = pd.DataFrame({
        "c1": [1.0, 1.1, 2.0, 3.5, 4.2] + list(range(10)),
        "label": ["a", "b", "b", "a", "a"] * 3,
    })
    model = ydf.RandomForestLearner(num_trees=3, label="label").train(pd_ds)
    logging.info(model)

  def test_train_gradient_boosted_tree(self):
    pd_ds = pd.DataFrame({
        "c1": [1.0, 1.1, 2.0, 3.5, 4.2] + list(range(10)),
        "label": ["a", "b", "b", "a", "a"] * 3,
    })
    model = ydf.GradientBoostedTreesLearner(num_trees=10, label="label").train(
        pd_ds
    )
    logging.info(model)

  def test_train_cart(self):
    pd_ds = pd.DataFrame({
        "c1": [1.0, 1.1, 2.0, 3.5, 4.2] + list(range(10)),
        "label": ["a", "b", "b", "a", "a"] * 3,
    })
    model = ydf.CartLearner(label="label").train(pd_ds)
    logging.info(model)


if __name__ == "__main__":
  absltest.main()
