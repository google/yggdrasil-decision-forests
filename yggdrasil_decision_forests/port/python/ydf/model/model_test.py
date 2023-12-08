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

import logging
import os
import tempfile
import textwrap

from absl.testing import absltest
from absl.testing import parameterized
import numpy.testing as npt
import pandas as pd

from ydf.model import generic_model
from ydf.model import model_lib
from ydf.model import model_metadata
from ydf.model.gradient_boosted_trees_model import gradient_boosted_trees_model
from ydf.model.random_forest_model import random_forest_model
from ydf.utils import test_utils


class GenericModelTest(parameterized.TestCase):

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
    self.assertEqual(model.name(), "RANDOM_FOREST")

    test_df = pd.read_csv(dataset_path)
    predictions = model.predict(test_df)
    predictions_df = pd.read_csv(predictions_path)

    expected_predictions = predictions_df[">50K"].to_numpy()
    npt.assert_almost_equal(predictions, expected_predictions, decimal=5)

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
    self.assertIsInstance(
        model, gradient_boosted_trees_model.GradientBoostedTreesModel
    )
    self.assertEqual(model.name(), "GRADIENT_BOOSTED_TREES")

    test_df = pd.read_csv(dataset_path)
    predictions = model.predict(test_df)
    predictions_df = pd.read_csv(predictions_path)

    expected_predictions = predictions_df[">50K"].to_numpy()
    npt.assert_almost_equal(predictions, expected_predictions, decimal=5)

  def test_predict_without_label_column(self):
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

    test_df = pd.read_csv(dataset_path).drop(columns=["income"])
    predictions = model.predict(test_df)
    predictions_df = pd.read_csv(predictions_path)

    expected_predictions = predictions_df[">50K"].to_numpy()
    npt.assert_almost_equal(predictions, expected_predictions, decimal=5)

  def test_predict_fails_with_missing_feature_columns(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_rf"
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    model = model_lib.load_model(model_path)

    test_df = pd.read_csv(dataset_path).drop(columns=["age"])
    with self.assertRaises(ValueError):
      _ = model.predict(test_df)

  def test_evaluate_fails_with_missing_label_columns(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_rf"
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    model = model_lib.load_model(model_path)

    test_df = pd.read_csv(dataset_path).drop(columns=["income"])
    with self.assertRaises(ValueError):
      _ = model.evaluate(test_df)

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
            | <=50K |  6987 |   425 |
            +-------+-------+-------+
            |  >50K |   822 |  1535 |
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

  def test_analyze_adult_gbt(self):
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
        ' `analysis.to_file("analysis.html")`.',
    )

    # Note: The analysis computation is not deterministic.
    analysis_html = analysis._repr_html_()
    self.assertIn("Partial Dependence Plot", analysis_html)
    self.assertIn("Conditional Expectation Plot", analysis_html)
    self.assertIn("Variable Importance", analysis_html)

  def test_explain_prediction_adult_gbt(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )

    model = model_lib.load_model(model_path)
    test_df = pd.read_csv(dataset_path, nrows=1)
    analysis = model.analyze_prediction(test_df)

    self.assertEqual(
        str(analysis),
        "A prediction analysis. Use a notebook cell to display the analysis."
        " Alternatively, export the analysis with"
        ' `analysis.to_file("analysis.html")`.',
    )

    analysis_html = analysis._repr_html_()
    with open("/tmp/analysis.html", "w") as f:
      f.write(analysis_html)
    self.assertIn("Feature Variation", analysis_html)

  def test_explain_prediction_adult_gbt_with_wrong_selection(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    model = model_lib.load_model(model_path)
    test_df = pd.read_csv(dataset_path, nrows=3)
    with self.assertRaises(ValueError):
      _ = model.analyze_prediction(test_df)
    with self.assertRaises(ValueError):
      _ = model.analyze_prediction(test_df.iloc[:0])

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

  def test_model_describe_text(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "adult_binary_class_gbdt",
    )
    model = model_lib.load_model(model_path)
    self.assertIn('Type: "GRADIENT_BOOSTED_TREES"', model.describe("text"))

  def test_model_describe_html(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "adult_binary_class_gbdt",
    )
    model = model_lib.load_model(model_path)
    html_description = model.describe("html")
    self.assertIn("GRADIENT_BOOSTED_TREES", html_description)

  def test_model_to_cpp(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "adult_binary_class_gbdt",
    )
    model = model_lib.load_model(model_path)
    cc = model.to_cpp()
    logging.info("cc:\n%s", cc)

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

  def test_model_metadata(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    model = model_lib.load_model(model_path)
    metadata = model_metadata.ModelMetadata(
        owner="TestOwner",
        created_date=31415,
        uid=271828,
        framework="TestFramework",
    )
    model.set_metadata(metadata)
    self.assertEqual(metadata, model.metadata())

  def test_label_col_idx(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    model = model_lib.load_model(model_path)
    self.assertEqual(model.label_col_idx(), 14)

  def test_label_classes(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "adult_binary_class_gbdt"
    )
    model = model_lib.load_model(model_path)
    label_classes = model.label_classes()
    self.assertEqual(label_classes, ["<=50K", ">50K"])

  def test_model_with_catset(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "sst_binary_class_gbdt"
    )
    model = model_lib.load_model(model_path)
    test_ds_path = "csv:" + os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "sst_binary_test.csv"
    )
    evaluation = model.evaluate(test_ds_path)
    self.assertAlmostEqual(evaluation.accuracy, 0.80011, places=5)


if __name__ == "__main__":
  absltest.main()
