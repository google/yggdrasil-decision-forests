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

"""Tests for basic model inference."""

import logging
import os
import tempfile
import textwrap

from absl.testing import absltest
import numpy as np
import pandas as pd

from pybind11_abseil import status
from ydf.model import generic_model
from ydf.model import model_lib
from ydf.model import random_forest_model
from ydf.utils import test_utils


class DecisionForestModelTest(absltest.TestCase):

  def test_predict_adult_rf(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_rf"
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    predictions_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "prediction",
        "adult_test_binary_class_rf.csv",
    )
    model = model_lib.load_model(model_path)
    self.assertIsInstance(model, random_forest_model.RandomForestModel)
    self.assertEqual(model.num_trees(), 100)
    self.assertEqual(model.name(), "RANDOM_FOREST")

    test_df = pd.read_csv(dataset_path)
    predictions = model.predict(test_df)
    predictions_df = pd.read_csv(predictions_path)

    expected_predictions = predictions_df[">50K"].to_numpy()
    np.testing.assert_almost_equal(predictions, expected_predictions, decimal=5)

  def test_predict_adult_gbt(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    predictions_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "prediction",
        "adult_test_binary_class_gbdt.csv",
    )
    model = model_lib.load_model(model_path)
    # TODO: Check for GBT once implemented.
    self.assertIsInstance(model, generic_model.GenericModel)
    self.assertEqual(model.name(), "GRADIENT_BOOSTED_TREES")

    test_df = pd.read_csv(dataset_path)
    predictions = model.predict(test_df)
    predictions_df = pd.read_csv(predictions_path)

    expected_predictions = predictions_df[">50K"].to_numpy()
    np.testing.assert_almost_equal(predictions, expected_predictions, decimal=5)

  def test_evaluate_adult_gbt(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )

    model = model_lib.load_model(model_path)
    test_df = pd.read_csv(dataset_path)
    evaluation = model.evaluate(test_df)

    self.assertEqual(
        str(evaluation),
        textwrap.dedent("""\
        accuracy: 0.872351
        confusion matrix:
            label (row) \\ prediction (col)
            +-------+-------+-------+
            |       | <=50K |  >50K |
            +-------+-------+-------+
            | <=50K |  6987 |   822 |
            +-------+-------+-------+
            |  >50K |   425 |  1535 |
            +-------+-------+-------+
        characteristics:
            name: '>50K' vs others
            ROC AUC: 0.927459
            PR AUC: 0.828393
            Num thresholds: 9491
        loss: 0.279777
        num examples: 9769
        num examples (weighted): 9769
        """),
    )

    # with open("/tmp/evaluation.html", "w") as f:
    #   f.write(evaluation._repr_html_())

  def test_analize_adult_gbt(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )

    model = model_lib.load_model(model_path)
    test_df = pd.read_csv(dataset_path)
    analysis = model.analyze(test_df, permutation_variable_importance_rounds=5)

    self.assertEqual(
        str(analysis),
        "A model analysis. Use a notebook cell to display the analysis."
        " Alternatively, export the analysis with"
        ' `analysis.to_html("analysis.html")`.',
    )

    # Note: The analysis computation is not deterministic.
    analysis_html = analysis._repr_html_()
    self.assertIn("Partial Dependence Plot", analysis_html)
    self.assertIn("Conditional Expectation Plot", analysis_html)
    self.assertIn("Permutation Variable Importance", analysis_html)

    # with open("/tmp/analyze.html", "w") as f:
    #   f.write(analysis_html)

  def test_evaluate_bootstrapping_default(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "abalone_regression_gbdt"
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "abalone.csv"
    )
    model = model_lib.load_model(model_path)
    test_df = pd.read_csv(dataset_path)
    evaluation = model.evaluate(test_df)
    self.assertIsNone(evaluation.rmse_ci95_bootstrap)

  def test_evaluate_bootstrapping_bool(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "abalone_regression_gbdt"
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "abalone.csv"
    )
    model = model_lib.load_model(model_path)
    test_df = pd.read_csv(dataset_path)
    evaluation = model.evaluate(test_df, bootstrapping=True)
    self.assertIsNotNone(evaluation.rmse_ci95_bootstrap)
    self.assertAlmostEqual(evaluation.rmse_ci95_bootstrap[0], 1.723, 2)
    self.assertAlmostEqual(evaluation.rmse_ci95_bootstrap[1], 1.866, 2)

  def test_evaluate_bootstrapping_integer(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "abalone_regression_gbdt"
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "abalone.csv"
    )
    model = model_lib.load_model(model_path)
    test_df = pd.read_csv(dataset_path)
    evaluation = model.evaluate(test_df, bootstrapping=599)
    self.assertIsNotNone(evaluation.rmse_ci95_bootstrap)
    self.assertAlmostEqual(evaluation.rmse_ci95_bootstrap[0], 1.723, 1)
    self.assertAlmostEqual(evaluation.rmse_ci95_bootstrap[1], 1.866, 1)

  def test_evaluate_bootstrapping_error(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "abalone_regression_gbdt"
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "abalone.csv"
    )
    model = model_lib.load_model(model_path)
    test_df = pd.read_csv(dataset_path)
    with self.assertRaisesRegex(ValueError, "an integer greater than 100"):
      model.evaluate(test_df, bootstrapping=1)

  def test_prefixed_model_loading_autodetection(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "prefixed_adult_binary_class_gbdt",
    )
    model = model_lib.load_model(model_path)
    self.assertEqual(model.name(), "GRADIENT_BOOSTED_TREES")

  def test_prefixed_model_loading_explicit(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "prefixed_adult_binary_class_gbdt",
    )
    model = model_lib.load_model(
        model_path, generic_model.ModelIOOptions(file_prefix="prefixed_")
    )
    self.assertEqual(model.name(), "GRADIENT_BOOSTED_TREES")

  def test_prefixed_model_loading_fails_when_incorrect(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "prefixed_adult_binary_class_gbdt",
    )
    with self.assertRaises(test_utils.AbslInvalidArgumentError):
      model_lib.load_model(
          model_path, generic_model.ModelIOOptions(file_prefix="wrong_prefix_")
      )

  def test_model_load_and_save(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "prefixed_adult_binary_class_gbdt",
    )
    model = model_lib.load_model(
        model_path, generic_model.ModelIOOptions(file_prefix="prefixed_")
    )
    with tempfile.TemporaryDirectory() as tempdir:
      model.save(tempdir, generic_model.ModelIOOptions(file_prefix="my_prefix"))
      self.assertTrue(os.path.exists(os.path.join(tempdir, "my_prefixdone")))

  def test_model_str(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "adult_binary_class_gbdt",
    )
    model = model_lib.load_model(model_path)
    self.assertEqual(
        str(model),
        """\
Model: GRADIENT_BOOSTED_TREES
Task: CLASSIFICATION
Class: ydf.GradientBoostedTreesModel
Use `model.describe()` for more details
""",
    )

  def test_model_describe(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "adult_binary_class_gbdt",
    )
    model = model_lib.load_model(model_path)
    self.assertIn('Type: "GRADIENT_BOOSTED_TREES"', model.describe())

  def test_model_to_cpp(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "adult_binary_class_gbdt",
    )
    model = model_lib.load_model(model_path)
    cc = model.to_cpp()
    logging.info("cc:\n%s", cc)

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

  def test_benchmark(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    model = model_lib.load_model(model_path)
    test_df = pd.read_csv(dataset_path)
    benchmark_result = model.benchmark(test_df)
    print(benchmark_result)


class RandomForestModelTest(absltest.TestCase):

  def test_oob_evaluations(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_rf"
    )
    model = model_lib.load_model(model_path)
    # TODO: Fill this test when OOB evaluations are exposed.
    with self.assertRaises(NotImplementedError):
      model.out_of_bag_evaluation()


class GradientBoostedTreesTest(absltest.TestCase):

  def test_validation_loss(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    model = model_lib.load_model(model_path)

    validation_loss = model.validation_loss()
    self.assertAlmostEqual(validation_loss, 0.573842942, places=6)


if __name__ == "__main__":
  absltest.main()
