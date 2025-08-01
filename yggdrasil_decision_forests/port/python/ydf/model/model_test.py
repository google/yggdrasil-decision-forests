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

import concurrent.futures
import logging
import os
import pickle
import tempfile
import textwrap
import time

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import numpy.testing as npt
import pandas as pd

from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.dataset import dataset
from ydf.model import analysis as analysis_lib
from ydf.model import generic_model
from ydf.model import model_lib
from ydf.model import model_metadata
from ydf.model.gradient_boosted_trees_model import gradient_boosted_trees_model
from ydf.model.random_forest_model import random_forest_model
from ydf.utils import test_utils


class GenericModelTest(parameterized.TestCase):

  maxDiff = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Loading models needed in many unittests.
    model_dir = os.path.join(test_utils.ydf_test_data_path(), "model")
    cls._model_dir = model_dir

    # This model is a Random Forest classification model without training logs.
    cls.adult_binary_class_rf = model_lib.load_model(
        os.path.join(model_dir, "adult_binary_class_rf")
    )
    # This model is a GBDT classification model without training logs.
    cls.adult_binary_class_gbdt = model_lib.load_model(
        os.path.join(model_dir, "adult_binary_class_gbdt")
    )
    # This model is a GBDT multi-class classification model .
    cls.iris_multi_class_gbdt = model_lib.load_model(
        os.path.join(model_dir, "iris_multi_class_gbdt")
    )
    # This model is a GBDT regression model without training logs.
    cls.abalone_regression_gbdt = model_lib.load_model(
        os.path.join(model_dir, "abalone_regression_gbdt")
    )
    # This model is a GBDT ranking model.
    cls.synthetic_ranking_gbdt = model_lib.load_model(
        os.path.join(model_dir, "synthetic_ranking_gbdt")
    )
    # This model is a RF uplift model.
    cls.sim_pte_categorical_uplift_rf = model_lib.load_model(
        os.path.join(model_dir, "sim_pte_categorical_uplift_rf")
    )

    ds_dir = os.path.join(test_utils.ydf_test_data_path(), "dataset")

    cls.adult_binary_class_gbdt_test_ds = pd.read_csv(
        os.path.join(ds_dir, "adult_test.csv")
    )
    cls.abalone_regression_gbdt_test_ds = pd.read_csv(
        os.path.join(ds_dir, "abalone.csv")
    )
    cls.synthetic_ranking_gbdt_test_ds = pd.read_csv(
        os.path.join(ds_dir, "synthetic_ranking_test.csv")
    )

  def test_rf_instance(self):
    self.assertIsInstance(
        self.adult_binary_class_rf,
        random_forest_model.RandomForestModel,
    )
    self.assertEqual(self.adult_binary_class_rf.name(), "RANDOM_FOREST")

  def test_gbt_instance(self):
    self.assertIsInstance(
        self.adult_binary_class_gbdt,
        gradient_boosted_trees_model.GradientBoostedTreesModel,
    )
    self.assertEqual(
        self.adult_binary_class_gbdt.name(), "GRADIENT_BOOSTED_TREES"
    )

  def test_predict_adult_rf(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    predictions_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "prediction",
        "adult_test_binary_class_rf.csv",
    )

    test_df = pd.read_csv(dataset_path)
    predictions_gt = pd.read_csv(predictions_path)

    # Test probability predictions
    predictions = self.adult_binary_class_rf.predict(test_df)
    expected_predictions = predictions_gt[">50K"].to_numpy()
    npt.assert_almost_equal(predictions, expected_predictions, decimal=5)

    prediction_classes = self.adult_binary_class_rf.predict_class(test_df)
    expected_prediction_classes = np.take(
        ["<=50K", ">50K"], expected_predictions > 0.5
    )
    self.assertEqual(prediction_classes.shape, (len(test_df),))
    self.assertEqual(prediction_classes.dtype.type, np.str_)
    npt.assert_equal(prediction_classes, expected_prediction_classes)

  def test_predict_adult_gbt(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    predictions_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "prediction",
        "adult_test_binary_class_gbdt.csv",
    )

    test_df = pd.read_csv(dataset_path)
    predictions = self.adult_binary_class_gbdt.predict(test_df)
    predictions_df = pd.read_csv(predictions_path)

    expected_predictions = predictions_df[">50K"].to_numpy()
    npt.assert_almost_equal(predictions, expected_predictions, decimal=5)

  def test_predict_iris_gbt(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "iris.csv"
    )
    test_df = pd.read_csv(dataset_path)
    predictions = self.iris_multi_class_gbdt.predict(test_df)
    prediction_classes = self.iris_multi_class_gbdt.predict_class(test_df)

    self.assertEqual(predictions.shape, (len(test_df), 3))
    self.assertEqual(predictions.dtype.type, np.float32)

    self.assertEqual(prediction_classes.shape, (len(test_df),))
    self.assertEqual(prediction_classes.dtype.type, np.str_)

    self.assertAlmostEqual(
        np.mean(
            np.take(
                self.iris_multi_class_gbdt.label_classes(),
                np.argmax(predictions, axis=1),
            )
            == test_df["class"]
        ),
        0.96666,
        delta=0.0001,
    )

    self.assertAlmostEqual(
        np.mean(prediction_classes == test_df["class"]), 0.96666, delta=0.0001
    )

    npt.assert_equal(
        np.take(
            self.iris_multi_class_gbdt.label_classes(),
            np.argmax(predictions, axis=1),
        ),
        prediction_classes,
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "regression",
          "model_name": "abalone_regression_gbdt",
          "test_ds": "abalone.csv",
      },
      {
          "testcase_name": "ranking",
          "model_name": "synthetic_ranking_gbdt",
          "test_ds": "synthetic_ranking_test.csv",
      },
      {
          "testcase_name": "uplift",
          "model_name": "sim_pte_categorical_uplift_rf",
          "test_ds": "sim_pte_test.csv",
      },
  )
  def test_predict_class_not_allowed(self, model_name, test_ds):
    model = getattr(self, model_name)
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", test_ds
    )
    with self.assertRaisesRegex(
        ValueError,
        "predict_class is only supported for classification models",
    ):
      _ = model.predict_class(dataset_path)

  def test_predict_without_label_column(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    predictions_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "prediction",
        "adult_test_binary_class_rf.csv",
    )

    test_df = pd.read_csv(dataset_path).drop(columns=["income"])
    predictions = self.adult_binary_class_rf.predict(test_df)
    predictions_df = pd.read_csv(predictions_path)

    expected_predictions = predictions_df[">50K"].to_numpy()
    npt.assert_almost_equal(predictions, expected_predictions, decimal=5)

  def test_predict_fails_with_missing_feature_columns(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )

    test_df = pd.read_csv(dataset_path).drop(columns=["age"])
    with self.assertRaises(ValueError):
      _ = self.adult_binary_class_rf.predict(test_df)

  def test_evaluate_fails_with_missing_label_columns(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )

    test_df = pd.read_csv(dataset_path).drop(columns=["income"])
    with self.assertRaises(ValueError):
      _ = self.adult_binary_class_rf.evaluate(test_df)

  def test_evaluate_adult_gbt(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )

    test_df = pd.read_csv(dataset_path)
    evaluation = self.adult_binary_class_gbdt.evaluate(test_df)

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
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )

    test_df = pd.read_csv(dataset_path)
    analysis = self.adult_binary_class_gbdt.analyze(
        test_df, permutation_variable_importance_rounds=5
    )

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

  def test_analyze_adult_gbt_sub_features(self):
    _ = self.adult_binary_class_gbdt.analyze(
        self.adult_binary_class_gbdt_test_ds,
        permutation_variable_importance_rounds=5,
        features=["occupation", "age"],
    )

  def test_analyze_adult_gbt_sub_features_wrong_feature(self):
    with self.assertRaisesRegex(
        ValueError, "Unknown column non_existing_feature"
    ):
      _ = self.adult_binary_class_gbdt.analyze(
          self.adult_binary_class_gbdt_test_ds,
          permutation_variable_importance_rounds=5,
          features=["non_existing_feature"],
      )

  def test_analyze_programmatic_data_access_classification(self):
    """Test programmatic access to analysis data."""
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    test_df = pd.read_csv(dataset_path)
    # Large maximum duration reduces test flakiness.
    analysis = self.adult_binary_class_gbdt.analyze(
        test_df, num_bins=4, maximum_duration=600
    )

    # Checked against report.
    self.assertSetEqual(
        set(analysis.variable_importances()),
        set([
            "MEAN_DECREASE_IN_ACCURACY",
            "MEAN_DECREASE_IN_AP_>50K_VS_OTHERS",
            "MEAN_DECREASE_IN_AUC_>50K_VS_OTHERS",
            "MEAN_DECREASE_IN_PRAUC_>50K_VS_OTHERS",
            "[In model] NUM_NODES",
            "[In model] NUM_AS_ROOT",
            "[In model] SUM_SCORE",
            "[In model] INV_MEAN_MIN_DEPTH",
        ]),
    )
    self.assertEqual(
        analysis.variable_importances()["[In model] NUM_AS_ROOT"],
        [
            (31.0, "age"),
            (22.0, "marital_status"),
            (8.0, "capital_gain"),
            (3.0, "occupation"),
            (2.0, "education_num"),
            (1.0, "education"),
            (1.0, "capital_loss"),
        ],
    )
    pdps = analysis.partial_dependence_plots()
    self.assertLen(pdps, 14)
    test_utils.assert_almost_equal(
        pdps[:2],
        [
            analysis_lib.PartialDependencePlot(
                feature_names=["age"],
                feature_values=[np.array([22.25, 32.5, 43.0, 69.25])],
                predictions=np.array([
                    [0.89487603, 0.10512417],
                    [0.77957238, 0.22042732],
                    [0.73085017, 0.26914773],
                    [0.79168959, 0.20831237],
                ]),
            ),
            analysis_lib.PartialDependencePlot(
                feature_names=["workclass"],
                feature_values=[
                    np.array(
                        [
                            "<OOD>",
                            "Private",
                            "Self-emp-not-inc",
                            "Local-gov",
                            "State-gov",
                            "Self-emp-inc",
                            "Federal-gov",
                            "Without-pay",
                        ],
                    )
                ],
                predictions=np.array([
                    [0.75371707, 0.24627779],
                    [0.75371707, 0.24627779],
                    [0.78163852, 0.21835641],
                    [0.77356506, 0.2264321],
                    [0.76878546, 0.23121109],
                    [0.73476332, 0.26523371],
                    [0.72042147, 0.27957545],
                    [0.72042147, 0.27957545],
                ]),
            ),
        ],
    )

  def test_analyze_programmatic_data_access_classification_sub_features(self):
    """Test programmatic access to analysis data."""
    # Large maximum duration reduces test flakiness.
    analysis = self.adult_binary_class_gbdt.analyze(
        self.adult_binary_class_gbdt_test_ds,
        num_bins=4,
        maximum_duration=60,
        features=["occupation", "age"],
    )
    pdps = analysis.partial_dependence_plots()
    self.assertLen(pdps, 2)
    test_utils.assert_almost_equal(
        pdps,
        [
            analysis_lib.PartialDependencePlot(
                feature_names=["occupation"],
                feature_values=[
                    np.array(
                        [
                            "<OOD>",
                            "Prof-specialty",
                            "Exec-managerial",
                            "Craft-repair",
                            "Adm-clerical",
                            "Sales",
                            "Other-service",
                            "Machine-op-inspct",
                            "Transport-moving",
                            "Handlers-cleaners",
                            "Farming-fishing",
                            "Tech-support",
                            "Protective-serv",
                            "Priv-house-serv",
                        ],
                    )
                ],
                predictions=np.array([
                    [0.7290186, 0.27097793],
                    [0.7290186, 0.27097793],
                    [0.70368701, 0.2963083],
                    [0.77382477, 0.22617291],
                    [0.76818352, 0.23181249],
                    [0.76350558, 0.23649114],
                    [0.82140092, 0.1785953],
                    [0.81608086, 0.18391325],
                    [0.81310829, 0.18688661],
                    [0.84014485, 0.15985094],
                    [0.84189474, 0.15810076],
                    [0.7291793, 0.27081758],
                    [0.73594471, 0.26405182],
                    [0.8489624, 0.15103142],
                ]),
            ),
            analysis_lib.PartialDependencePlot(
                feature_names=["age"],
                feature_values=[np.array([22.25, 32.5, 43.0, 69.25])],
                predictions=np.array([
                    [0.89487603, 0.10512417],
                    [0.77957238, 0.22042732],
                    [0.73085017, 0.26914773],
                    [0.79168959, 0.20831237],
                ]),
            ),
        ],
    )

  def test_analyze_programmatic_data_access_regression(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "abalone.csv"
    )
    test_df = pd.read_csv(dataset_path)
    analysis = self.abalone_regression_gbdt.analyze(test_df, num_bins=4)

    # Checked against report.
    self.assertSetEqual(
        set(analysis.variable_importances()),
        set([
            "MEAN_INCREASE_IN_RMSE",
            "[In model] INV_MEAN_MIN_DEPTH",
            "[In model] SUM_SCORE",
            "[In model] NUM_NODES",
            "[In model] NUM_AS_ROOT",
        ]),
    )
    self.assertEqual(
        analysis.variable_importances()["[In model] NUM_AS_ROOT"],
        [
            (20.0, "ShellWeight"),
            (10.0, "Height"),
            (4.0, "Type"),
            (3.0, "Diameter"),
            (1.0, "LongestShell"),
        ],
    )
    pdps = analysis.partial_dependence_plots()
    self.assertLen(pdps, 8)
    test_utils.assert_almost_equal(
        pdps[:1],
        [
            analysis_lib.PartialDependencePlot(
                feature_names=["Type"],
                feature_values=[np.array(["<OOD>", "M", "I", "F"])],
                predictions=np.array([
                    10.20926439,
                    10.22119849,
                    9.8006166,
                    10.19440551,
                ]),
            ),
        ],
    )

  def test_analyze_programmatic_data_access_ranking(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "synthetic_ranking_test.csv"
    )
    test_df = pd.read_csv(dataset_path)
    analysis = self.synthetic_ranking_gbdt.analyze(test_df, num_bins=4)

    # Checked against report.
    self.assertSetEqual(
        set(analysis.variable_importances()),
        set([
            "MEAN_DECREASE_IN_NDCG",
            "[In model] SUM_SCORE",
            "[In model] NUM_NODES",
            "[In model] INV_MEAN_MIN_DEPTH",
            "[In model] NUM_AS_ROOT",
        ]),
    )
    self.assertEqual(
        analysis.variable_importances()["[In model] NUM_AS_ROOT"],
        [(11.0, "cat_str_0"), (2.0, "num_0"), (1.0, "num_2")],
    )
    pdps = analysis.partial_dependence_plots()
    self.assertLen(pdps, 8)
    test_utils.assert_almost_equal(
        pdps[:1],
        [
            analysis_lib.PartialDependencePlot(
                feature_names=["cat_int_0"],
                feature_values=[np.array([4.25, 11.5, 19.5, 26.75])],
                predictions=np.array(
                    [-0.2301757, -0.20255686, -0.2001005, -0.19919406]
                ),
            ),
        ],
    )

  def test_explain_prediction_adult_gbt(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )

    test_df = pd.read_csv(dataset_path, nrows=1)
    analysis = self.adult_binary_class_gbdt.analyze_prediction(test_df)

    self.assertEqual(
        str(analysis),
        "A prediction analysis. Use a notebook cell to display the analysis."
        " Alternatively, export the analysis with"
        ' `analysis.to_file("analysis.html")`.',
    )

    analysis_html = analysis._repr_html_()
    self.assertIn("Feature Variation", analysis_html)

  def test_explain_prediction_adult_gbt_with_sub_features(self):
    analysis = self.adult_binary_class_gbdt.analyze_prediction(
        self.adult_binary_class_gbdt_test_ds[:1],
        features=["occupation", "age"],
    )

    self.assertEqual(
        str(analysis),
        "A prediction analysis. Use a notebook cell to display the analysis."
        " Alternatively, export the analysis with"
        ' `analysis.to_file("analysis.html")`.',
    )

    analysis_html = analysis._repr_html_()
    self.assertIn("Feature Variation", analysis_html)

  def test_explain_prediction_adult_gbt_with_wrong_selection(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    test_df = pd.read_csv(dataset_path, nrows=3)
    with self.assertRaises(ValueError):
      _ = self.adult_binary_class_gbdt.analyze_prediction(test_df)
    with self.assertRaises(ValueError):
      _ = self.adult_binary_class_gbdt.analyze_prediction(test_df.iloc[:0])

  def test_evaluate_bootstrapping_default(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "abalone.csv"
    )
    test_df = pd.read_csv(dataset_path)
    evaluation = self.abalone_regression_gbdt.evaluate(test_df)
    self.assertIsNone(evaluation.rmse_ci95_bootstrap)

  def test_evaluate_bootstrapping_bool(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "abalone.csv"
    )
    test_df = pd.read_csv(dataset_path)
    evaluation = self.abalone_regression_gbdt.evaluate(
        test_df, bootstrapping=True
    )
    self.assertIsNotNone(evaluation.rmse_ci95_bootstrap)
    self.assertAlmostEqual(evaluation.rmse_ci95_bootstrap[0], 1.723, 2)
    self.assertAlmostEqual(evaluation.rmse_ci95_bootstrap[1], 1.866, 2)

  def test_evaluate_bootstrapping_integer(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "abalone.csv"
    )
    test_df = pd.read_csv(dataset_path)
    evaluation = self.abalone_regression_gbdt.evaluate(
        test_df, bootstrapping=599
    )
    self.assertIsNotNone(evaluation.rmse_ci95_bootstrap)
    self.assertAlmostEqual(evaluation.rmse_ci95_bootstrap[0], 1.723, 1)
    self.assertAlmostEqual(evaluation.rmse_ci95_bootstrap[1], 1.866, 1)

  def test_evaluate_bootstrapping_error(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "abalone.csv"
    )
    test_df = pd.read_csv(dataset_path)
    with self.assertRaisesRegex(ValueError, "an integer greater than 100"):
      self.abalone_regression_gbdt.evaluate(test_df, bootstrapping=1)

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
    self.assertEqual(
        str(self.adult_binary_class_gbdt),
        """\
Model: GRADIENT_BOOSTED_TREES
Task: CLASSIFICATION
Class: ydf.GradientBoostedTreesModel
Use `model.describe()` for more details.
""",
    )

  def test_model_describe_text(self):
    text_description = self.adult_binary_class_gbdt.describe("text")
    # Model description
    self.assertIn('Type: "GRADIENT_BOOSTED_TREES"', text_description)
    # Dataspec description
    self.assertIn("DATASPEC:", text_description)
    self.assertIn("Number of records:", text_description)

  def test_model_describe_html(self):
    html_description = self.adult_binary_class_gbdt.describe("html")
    self.assertIn("GRADIENT_BOOSTED_TREES", html_description)

  def test_model_to_cpp(self):
    cc = self.adult_binary_class_gbdt.to_cpp()
    logging.info("cc:\n%s", cc)

  def test_benchmark(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    test_df = pd.read_csv(dataset_path)
    benchmark_result = self.adult_binary_class_gbdt.benchmark(test_df)
    print(benchmark_result)

  def test_model_metadata(self):
    metadata = model_metadata.ModelMetadata(
        owner="TestOwner",
        created_date=31415,
        uid=271828,
        framework="TestFramework",
        custom_fields={"string": "bar", "bytes": b"Caf\351"},
    )
    self.adult_binary_class_gbdt.set_metadata(metadata)
    self.assertEqual(metadata, self.adult_binary_class_gbdt.metadata())

  def test_label_col_idx(self):
    self.assertEqual(self.adult_binary_class_gbdt.label_col_idx(), 14)

  def test_label_classes(self):
    label_classes = self.adult_binary_class_gbdt.label_classes()
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

  def test_multi_thread_predict(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    test_df = pd.read_csv(dataset_path)
    test_ds = dataset.create_vertical_dataset(
        test_df, data_spec=self.adult_binary_class_gbdt.data_spec()
    )
    for num_workers in range(1, 10 + 1):
      with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
        begin = time.time()
        _ = list(
            executor.map(self.adult_binary_class_gbdt.predict, [test_ds] * 10)
        )
        end = time.time()
        logging.info("Runtime for %s workers: %s", num_workers, end - begin)

  def test_self_evaluation_gbt(self):
    # This model is a classification model with full training logs.
    gbt_adult_base_with_na_path = os.path.join(
        test_utils.ydf_test_data_path(), "golden", "gbt_adult_base_with_na"
    )
    gbt_adult_base_with_na = model_lib.load_model(gbt_adult_base_with_na_path)
    self_evaluation = gbt_adult_base_with_na.self_evaluation()
    self.assertAlmostEqual(self_evaluation.accuracy, 0.8498403)

  def test_self_evaluation_rf(self):
    self_evaluation = self.adult_binary_class_rf.self_evaluation()
    self.assertAlmostEqual(self_evaluation.loss, 0.31474323732)

  def test_empty_self_evaluation_rf(self):
    # Uplift models do not have OOB evaluations.
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "sim_pte_categorical_uplift_rf",
    )
    model = model_lib.load_model(model_path)
    self.assertIsNone(model.self_evaluation())

  def test_gbt_list_compatible_engines(self):
    self.assertContainsSubsequence(
        self.adult_binary_class_gbdt.list_compatible_engines(),
        ["GradientBoostedTreesGeneric"],
    )

  def test_rf_list_compatible_engines(self):
    self.assertContainsSubsequence(
        self.adult_binary_class_rf.list_compatible_engines(),
        ["RandomForestGeneric"],
    )

  def test_gbt_force_compatible_engines(self):
    test_df = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
        )
    )
    p1 = self.adult_binary_class_gbdt.predict(test_df)
    self.adult_binary_class_gbdt.force_engine("GradientBoostedTreesGeneric")
    p2 = self.adult_binary_class_gbdt.predict(test_df)
    self.adult_binary_class_gbdt.force_engine(None)
    p3 = self.adult_binary_class_gbdt.predict(test_df)

    np.testing.assert_allclose(
        p1,
        p2,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        p1,
        p3,
        rtol=1e-5,
        atol=1e-5,
    )

  def test_model_serialize(self):
    ds = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
        )
    )
    serialized_model = self.adult_binary_class_gbdt.serialize()
    self.assertIsInstance(serialized_model, bytes)
    deserialized_model = model_lib.deserialize_model(serialized_model)

    self.assertIsInstance(
        deserialized_model,
        gradient_boosted_trees_model.GradientBoostedTreesModel,
    )
    _ = deserialized_model.validation_loss()
    _ = deserialized_model.num_trees()

    original_predictions = self.adult_binary_class_gbdt.predict(ds)
    deserialized_predictions = deserialized_model.predict(ds)
    npt.assert_almost_equal(
        original_predictions, deserialized_predictions, decimal=5
    )

  def test_model_embed(self):
    model = model_lib.load_model(
        os.path.join(self._model_dir, "adult_binary_class_gbdt_v2")
    )
    while model.num_trees() > 3:
      model.remove_tree(model.num_trees() - 1)
    embedded_model_if_else_class = model.to_standalone_cc(algorithm="IF_ELSE")
    embedded_model_routing_proba = model.to_standalone_cc(
        algorithm="ROUTING",
        classification_output="PROBABILITY",
        categorical_from_string=False,
    )
    self.assertIsInstance(embedded_model_if_else_class, str)
    self.assertIsInstance(embedded_model_routing_proba, str)

    test_utils.golden_check_string(
        self,
        embedded_model_if_else_class,
        os.path.join(
            test_utils.ydf_test_data_path(),
            "golden",
            "embed",
            "adult_binary_class_gbdt_v2_class.h.golden",
        ),
    )
    test_utils.golden_check_string(
        self,
        embedded_model_routing_proba,
        os.path.join(
            test_utils.ydf_test_data_path(),
            "golden",
            "embed",
            "adult_binary_class_gbdt_v2_probability_routing.h.golden",
        ),
    )

  def test_model_pickling(self):
    ds = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
        )
    )
    pickled_model = pickle.dumps(self.adult_binary_class_gbdt)
    self.assertIsInstance(pickled_model, bytes)
    unpickled_model = pickle.loads(pickled_model)

    self.assertIsInstance(
        unpickled_model,
        gradient_boosted_trees_model.GradientBoostedTreesModel,
    )
    _ = unpickled_model.validation_loss()
    _ = unpickled_model.num_trees()

    original_predictions = self.adult_binary_class_gbdt.predict(ds)
    unpickled_predictions = unpickled_model.predict(ds)
    npt.assert_almost_equal(
        original_predictions, unpickled_predictions, decimal=5
    )

  def test_build_evaluation_dataspec_classification_default(self):
    model = self.adult_binary_class_gbdt
    original_column_names = [col.name for col in model.data_spec().columns]
    dataspec, label_idx, group_idx, required_columns = (
        model._build_evaluation_dataspec(
            override_task=abstract_model_pb2.Task.CLASSIFICATION,
            override_label=None,
            override_group=None,
            weighted=False,
        )
    )
    self.assertListEqual(
        [col.name for col in dataspec.columns], original_column_names
    )
    self.assertListEqual(required_columns, original_column_names)
    self.assertEqual(label_idx, model._model.label_col_idx())
    self.assertEqual(group_idx, -1)

  def test_build_evaluation_dataspec_classification_weighted_fails(self):
    model = self.adult_binary_class_gbdt
    with self.assertRaisesRegex(
        ValueError,
        "Weighted evaluation is only supported for models trained with"
        " weights.",
    ):
      _ = model._build_evaluation_dataspec(
          override_task=abstract_model_pb2.Task.CLASSIFICATION,
          override_label=None,
          override_group=None,
          weighted=True,
      )

  def test_build_evaluation_dataspec_classification_new_label(self):
    model = self.adult_binary_class_gbdt
    original_column_names = [col.name for col in model.data_spec().columns]
    dataspec, label_idx, group_idx, required_columns = (
        model._build_evaluation_dataspec(
            override_task=abstract_model_pb2.Task.CLASSIFICATION,
            override_label="NEW_LABEL",
            override_group=None,
            weighted=False,
        )
    )
    self.assertListEqual(
        [col.name for col in dataspec.columns],
        original_column_names + ["NEW_LABEL"],
    )
    self.assertListEqual(
        required_columns, original_column_names + ["NEW_LABEL"]
    )
    expected_label_col = len(original_column_names)
    self.assertEqual(label_idx, expected_label_col)
    self.assertEqual(group_idx, -1)
    self.assertEqual(
        dataspec.columns[expected_label_col].categorical,
        dataspec.columns[model._model.label_col_idx()].categorical,
    )

  def test_build_evaluation_dataspec_classification_to_regression(self):
    model = self.adult_binary_class_gbdt
    original_column_names = [col.name for col in model.data_spec().columns]
    dataspec, label_idx, group_idx, required_columns = (
        model._build_evaluation_dataspec(
            override_task=abstract_model_pb2.Task.REGRESSION,
            override_label="NEW_LABEL",
            override_group=None,
            weighted=False,
        )
    )
    self.assertListEqual(
        [col.name for col in dataspec.columns],
        original_column_names + ["NEW_LABEL"],
    )
    self.assertListEqual(
        required_columns, original_column_names + ["NEW_LABEL"]
    )
    expected_label_col = len(original_column_names)
    self.assertEqual(label_idx, expected_label_col)
    self.assertEqual(group_idx, -1)
    self.assertEqual(
        dataspec.columns[expected_label_col].type,
        data_spec_pb2.ColumnType.NUMERICAL,
    )

  def test_build_evaluation_dataspec_classification_to_ranking(self):
    model = self.adult_binary_class_gbdt
    original_column_names = [col.name for col in model.data_spec().columns]
    dataspec, label_idx, group_idx, required_columns = (
        model._build_evaluation_dataspec(
            override_task=abstract_model_pb2.Task.RANKING,
            override_label="NEW_LABEL",
            override_group="NEW_GROUP",
            weighted=False,
        )
    )
    self.assertListEqual(
        [col.name for col in dataspec.columns],
        original_column_names + ["NEW_LABEL", "NEW_GROUP"],
    )
    self.assertListEqual(
        required_columns, original_column_names + ["NEW_LABEL", "NEW_GROUP"]
    )
    expected_label_col = len(original_column_names) + 0
    expected_group_col = len(original_column_names) + 1
    self.assertEqual(label_idx, expected_label_col)
    self.assertEqual(group_idx, expected_group_col)
    self.assertEqual(
        dataspec.columns[expected_label_col].type,
        data_spec_pb2.ColumnType.NUMERICAL,
    )
    self.assertEqual(
        dataspec.columns[expected_group_col].type,
        data_spec_pb2.ColumnType.HASH,
    )

  def test_build_evaluation_dataspec_regression_new_label(self):
    model = self.abalone_regression_gbdt
    original_column_names = [col.name for col in model.data_spec().columns]
    dataspec, label_idx, group_idx, required_columns = (
        model._build_evaluation_dataspec(
            override_task=abstract_model_pb2.Task.REGRESSION,
            override_label="NEW_LABEL",
            override_group=None,
            weighted=False,
        )
    )
    self.assertListEqual(
        [col.name for col in dataspec.columns],
        original_column_names + ["NEW_LABEL"],
    )
    self.assertListEqual(
        required_columns, original_column_names + ["NEW_LABEL"]
    )
    expected_label_col = len(original_column_names)
    self.assertEqual(label_idx, expected_label_col)
    self.assertEqual(group_idx, -1)
    self.assertEqual(
        dataspec.columns[expected_label_col].type,
        data_spec_pb2.ColumnType.NUMERICAL,
    )

  def test_build_evaluation_dataspec_ranking_to_regression(self):
    model = self.synthetic_ranking_gbdt
    original_column_names = [col.name for col in model.data_spec().columns]
    dataspec, label_idx, group_idx, required_columns = (
        model._build_evaluation_dataspec(
            override_task=abstract_model_pb2.Task.REGRESSION,
            override_label=None,
            override_group=None,
            weighted=False,
        )
    )
    self.assertListEqual(
        [col.name for col in dataspec.columns],
        original_column_names,
    )
    self.assertListEqual(required_columns, original_column_names)
    self.assertEqual(label_idx, model._model.label_col_idx())
    self.assertEqual(group_idx, -1)

  def test_eval_binary_classification_as_regression(self):
    model = self.adult_binary_class_gbdt
    ds = self.adult_binary_class_gbdt_test_ds.copy()
    ds["income_regress"] = ds["income"] == ">50K"
    evaluation = model.evaluate(
        ds, task=generic_model.Task.REGRESSION, label="income_regress"
    )
    self.assertAlmostEqual(evaluation.rmse, 0.298, delta=0.001)

  def test_eval_binary_classification_as_regression_with_existing_column(self):
    model = self.adult_binary_class_gbdt
    ds = self.adult_binary_class_gbdt_test_ds.copy()
    evaluation = model.evaluate(
        ds, task=generic_model.Task.REGRESSION, label="age"
    )
    self.assertAlmostEqual(evaluation.rmse, 40.569, delta=0.001)

  def test_eval_binary_classification_as_regression_existing_col_from_file(
      self,
  ):
    model = self.adult_binary_class_gbdt
    evaluation = model.evaluate(
        "csv:"
        + os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
        ),
        task=generic_model.Task.REGRESSION,
        label="age",
    )
    self.assertAlmostEqual(evaluation.rmse, 40.569, delta=0.001)

  def test_eval_binary_classification_as_regression_from_file(self):
    with tempfile.TemporaryDirectory() as tempdir:
      ds_path = os.path.join(tempdir, "data.csv")
      model = self.adult_binary_class_gbdt
      ds = self.adult_binary_class_gbdt_test_ds.copy()

      ds["income_regress"] = (ds["income"] == ">50K").astype(int)
      ds.to_csv(ds_path, index=False)
      evaluation = model.evaluate(
          "csv:" + ds_path,
          task=generic_model.Task.REGRESSION,
          label="income_regress",
      )
      self.assertAlmostEqual(evaluation.rmse, 0.298, delta=0.001)

  def test_eval_binary_classification_as_regression_from_file_replace_col(self):
    with tempfile.TemporaryDirectory() as tempdir:
      ds_path = os.path.join(tempdir, "data.csv")
      model = self.adult_binary_class_gbdt
      ds = self.adult_binary_class_gbdt_test_ds.copy()

      ds["income"] = (ds["income"] == ">50K").astype(int)
      ds.to_csv(ds_path, index=False)
      evaluation = model.evaluate(
          "csv:" + ds_path, task=generic_model.Task.REGRESSION
      )
      self.assertAlmostEqual(evaluation.rmse, 0.298, delta=0.001)

  def test_eval_binary_classification_as_ranking(self):
    model = self.adult_binary_class_gbdt
    ds = self.adult_binary_class_gbdt_test_ds.copy()

    ds["income_ranking"] = ds["income"] == ">50K"
    ds["group"] = ds["race"]
    evaluation = model.evaluate(
        ds,
        task=generic_model.Task.RANKING,
        label="income_ranking",
        group="group",
    )
    self.assertAlmostEqual(evaluation.ndcg, 0.966, delta=0.001)

  def test_eval_binary_classification_as_ranking_replace_column(self):
    model = self.adult_binary_class_gbdt
    ds = self.adult_binary_class_gbdt_test_ds.copy()

    ds["income"] = ds["income"] == ">50K"
    ds["group"] = ds["race"]
    evaluation = model.evaluate(
        ds,
        task=generic_model.Task.RANKING,
        label="income",
        group="group",
    )
    self.assertAlmostEqual(evaluation.ndcg, 0.966, delta=0.001)

  def test_eval_binary_classification_on_other_column(self):
    model = self.adult_binary_class_gbdt
    ds = self.adult_binary_class_gbdt_test_ds.copy()

    ds["inv_income"] = ds["income"].map({">50K": "<=50K", "<=50K": ">50K"})
    evaluation = model.evaluate(ds, label="inv_income")
    self.assertAlmostEqual(evaluation.accuracy, 0.127, delta=0.001)
    self.assertAlmostEqual(
        evaluation.accuracy, 1 - model.evaluate(ds).accuracy, delta=0.001
    )

  def test_eval_ranking_as_regression(self):
    model = self.synthetic_ranking_gbdt
    ds = self.synthetic_ranking_gbdt_test_ds

    evaluation = model.evaluate(ds, task=generic_model.Task.REGRESSION)
    self.assertAlmostEqual(evaluation.rmse, 1.097, delta=0.001)

  def test_eval_regression_as_classification(self):
    model = self.abalone_regression_gbdt
    ds = self.abalone_regression_gbdt_test_ds.copy()
    ds["class_labels"] = (ds["Rings"] >= 6).astype(int)
    evaluation = model.evaluate(
        ds, label="class_labels", task=generic_model.Task.CLASSIFICATION
    )
    self.assertAlmostEqual(evaluation.accuracy, 0.954, delta=0.001)

  def test_eval_override_task_error_wrong_semantic(self):
    model = self.adult_binary_class_gbdt
    ds = self.adult_binary_class_gbdt_test_ds

    with self.assertRaisesRegex(
        ValueError,
        "Cannot convert NUMERICAL column ",
    ):
      model.evaluate(ds, task=generic_model.Task.REGRESSION)

  def test_eval_override_task_error_non_existing_label(self):
    model = self.adult_binary_class_gbdt
    ds = self.adult_binary_class_gbdt_test_ds

    with self.assertRaisesRegex(
        ValueError,
        "Missing required column 'NON_EXISTING_COLUMN'",
    ):
      model.evaluate(ds, label="NON_EXISTING_COLUMN")

  def test_eval_override_task_error_ad_evaluation(self):
    model = self.synthetic_ranking_gbdt
    ds = self.synthetic_ranking_gbdt_test_ds

    with self.assertRaisesRegex(
        ValueError, "Anomaly detection models don't have direct evaluation"
    ):
      model.evaluate(ds, task=generic_model.Task.ANOMALY_DETECTION)

  def test_eval_override_task_error_non_supported_task_override(self):
    model = self.synthetic_ranking_gbdt
    ds = self.synthetic_ranking_gbdt_test_ds.copy()
    ds["class_label"] = ds["GROUP"]

    with self.assertRaisesRegex(
        ValueError,
        "Non supported override of task from RANKING to CLASSIFICATION",
    ):
      model.evaluate(
          ds, label="class_label", task=generic_model.Task.CLASSIFICATION
      )

  @parameterized.named_parameters(
      {
          "testcase_name": "binary_classification",
          "model_name": "adult_binary_class_gbdt",
          "test_ds": "adult_test.csv",
      },
      {
          "testcase_name": "multiclass_classification",
          "model_name": "iris_multi_class_gbdt",
          "test_ds": "iris.csv",
      },
      {
          "testcase_name": "regression",
          "model_name": "abalone_regression_gbdt",
          "test_ds": "abalone.csv",
      },
      {
          "testcase_name": "ranking",
          "model_name": "synthetic_ranking_gbdt",
          "test_ds": "synthetic_ranking_test.csv",
      },
      {
          "testcase_name": "uplift",
          "model_name": "sim_pte_categorical_uplift_rf",
          "test_ds": "sim_pte_test.csv",
      },
  )
  def test_slow_engine_prediction(self, model_name, test_ds):
    model = getattr(self, model_name)
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", test_ds
    )
    fast_predictions = model.predict(dataset_path)
    slow_predictions = model.predict(dataset_path, use_slow_engine=True)
    np.testing.assert_allclose(
        fast_predictions, slow_predictions, atol=1e-6, rtol=1e-6
    )

  def test_evaluate_slow_engine(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )

    test_df = pd.read_csv(dataset_path)
    evaluation_slow = self.adult_binary_class_gbdt.evaluate(
        test_df, use_slow_engine=True
    )
    evaluation_fast = self.adult_binary_class_gbdt.evaluate(test_df)

    self.assertEqual(evaluation_fast, evaluation_slow)

  @parameterized.named_parameters(
      {
          "testcase_name": "ndcg@5",
          "truncation": 5,
          "expected_ndcg": 0.7204528553,
      },
      {
          "testcase_name": "ndcg@2",
          "truncation": 2,
          "expected_ndcg": 0.6304857312,
      },
      {
          "testcase_name": "ndcg@10",
          "truncation": 10,
          "expected_ndcg": 0.8384147895,
      },
  )
  def test_evaluate_ranking_ndcg_truncation(self, truncation, expected_ndcg):
    evaluation = self.synthetic_ranking_gbdt.evaluate(
        self.synthetic_ranking_gbdt_test_ds, ndcg_truncation=truncation
    )
    self.assertAlmostEqual(evaluation.ndcg, expected_ndcg)

  @parameterized.named_parameters(
      {
          "testcase_name": "mrr@5",
          "truncation": 5,
          "expected_mrr": 0.8242574257,
      },
      {
          "testcase_name": "mrr@2",
          "truncation": 2,
          "expected_mrr": 0.7920792079,
      },
      {
          "testcase_name": "mrr@10",
          "truncation": 10,
          "expected_mrr": 0.8259075907,
      },
  )
  def test_evaluate_ranking_mrr_truncation(self, truncation, expected_mrr):
    evaluation = self.synthetic_ranking_gbdt.evaluate(
        self.synthetic_ranking_gbdt_test_ds, mrr_truncation=truncation
    )
    self.assertAlmostEqual(evaluation.mrr, expected_mrr)

  @parameterized.named_parameters(
      {
          "testcase_name": "map@5",
          "truncation": 5,
          "expected_map": 0.793028052,
      },
      {
          "testcase_name": "map@2",
          "truncation": 2,
          "expected_map": 0.792079209,
      },
      {
          "testcase_name": "map@10",
          "truncation": 10,
          "expected_map": 0.7601983518,
      },
  )
  def test_evaluate_ranking_map_truncation(self, truncation, expected_map):
    evaluation = self.synthetic_ranking_gbdt.evaluate(
        self.synthetic_ranking_gbdt_test_ds, map_truncation=truncation
    )
    self.assertAlmostEqual(evaluation.map, expected_map, places=3)

  def test_model_save_pure_serving(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "adult_binary_class_gbdt",
    )
    with open(os.path.join(model_path, "header.pb"), "rb") as f:
      header = abstract_model_pb2.AbstractModel.FromString(f.read())
      self.assertFalse(header.is_pure_model)
    model = model_lib.load_model(model_path)
    with tempfile.TemporaryDirectory() as tempdir:
      model.save(tempdir, pure_serving=True)
      self.assertTrue(os.path.exists(os.path.join(tempdir, "done")))
      self.assertTrue(os.path.exists(os.path.join(tempdir, "header.pb")))
      with open(os.path.join(tempdir, "header.pb"), "rb") as f:
        header = abstract_model_pb2.AbstractModel.FromString(f.read())
        self.assertTrue(header.is_pure_model)
    # The model in memory still has debug information.
    with tempfile.TemporaryDirectory() as tempdir:
      model.save(tempdir)
      self.assertTrue(os.path.exists(os.path.join(tempdir, "done")))
      self.assertTrue(os.path.exists(os.path.join(tempdir, "header.pb")))
      with open(os.path.join(tempdir, "header.pb"), "rb") as f:
        header = abstract_model_pb2.AbstractModel.FromString(f.read())
        self.assertFalse(header.is_pure_model)


if __name__ == "__main__":
  absltest.main()
