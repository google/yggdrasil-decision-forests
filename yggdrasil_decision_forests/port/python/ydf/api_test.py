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

"""Test the API of YDF."""


import math
import os

from absl import logging
from absl.testing import absltest
import pandas as pd

import ydf  # In the world, use "import ydf"
from ydf.utils import test_utils


class ApiTest(absltest.TestCase):

  def test_create_dataset(self):
    pd_ds = pd.DataFrame({
        "c1": [1.0, 1.1, math.nan],
        "c2": [1, 2, 3],
        "c3": [True, False, True],
        "c4": ["x", "y", ""],
    })
    ds = ydf.create_vertical_dataset(pd_ds)
    logging.info("Dataset:\n%s", ds)

  def test_create_dataset_with_column(self):
    pd_ds = pd.DataFrame({
        "c1": [1.0, 1.1, math.nan],
        "c2": [1, 2, 3],
        "c3": [True, False, True],
        "c4": ["x", "y", ""],
    })
    ds = ydf.create_vertical_dataset(
        pd_ds,
        columns=[
            "c1",
            ("c2", ydf.Semantic.NUMERICAL),
            ydf.Column("c3"),
            ydf.Column("c4", ydf.Semantic.CATEGORICAL),
        ],
    )
    logging.info("Dataset:\n%s", ds)

  def test_load_and_save_model(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_rf"
    )
    model = ydf.load_model(model_path)
    logging.info(model)
    tempdir = self.create_tempdir().full_path
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

  def test_evaluate_model(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    model = ydf.load_model(model_path)
    test_ds = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
        )
    )
    evaluation = model.evaluate(test_ds)
    logging.info("Evaluation:\n%s", evaluation)
    self.assertAlmostEqual(evaluation.accuracy, 0.87235, 3)

    tempdir = self.create_tempdir().full_path
    with open(os.path.join(tempdir, "evaluation.html"), "w") as f:
      f.write(evaluation.html())

  def test_analyze_model(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    model = ydf.load_model(model_path)
    test_ds = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
        )
    )
    analysis = model.analyze(test_ds)
    logging.info("Analysis:\n%s", analysis)

    tempdir = self.create_tempdir().full_path
    with open(os.path.join(tempdir, "analysis.html"), "w") as f:
      f.write(analysis.html())

  def test_cross_validation(self):
    pd_ds = pd.DataFrame({
        "c1": [1.0, 1.1, 2.0, 3.5, 4.2] + list(range(10)),
        "label": ["a", "b", "b", "a", "a"] * 3,
    })
    learner = ydf.RandomForestLearner(num_trees=3, label="label")
    evaluation = learner.cross_validation(pd_ds)
    logging.info(evaluation)

  def test_export_to_cc(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    model = ydf.load_model(model_path)
    logging.info(
        "Copy the following in a .h file to run the model in C++:\n%s",
        model.to_cpp(),
    )

  def test_verbose_full(self):
    save_verbose = ydf.verbose(2)
    learner = ydf.RandomForestLearner(label="label")
    _ = learner.train(pd.DataFrame({"feature": [0, 1], "label": [0, 1]}))
    ydf.verbose(save_verbose)

  def test_verbose_default(self):
    learner = ydf.RandomForestLearner(label="label")
    _ = learner.train(pd.DataFrame({"feature": [0, 1], "label": [0, 1]}))

  def test_verbose_none(self):
    save_verbose = ydf.verbose(0)
    learner = ydf.RandomForestLearner(label="label")
    _ = learner.train(pd.DataFrame({"feature": [0, 1], "label": [0, 1]}))
    ydf.verbose(save_verbose)


if __name__ == "__main__":
  absltest.main()
