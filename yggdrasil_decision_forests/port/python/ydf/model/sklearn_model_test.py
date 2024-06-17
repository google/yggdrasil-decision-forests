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

from typing import Tuple
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from sklearn import datasets
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from ydf.model import export_sklearn
from ydf.model.decision_forest_model import decision_forest_model


def gen_anomaly_detection_dataset(
    n_samples: int = 120,
    n_outliers: int = 40,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
  """Generates a two-gaussians anomaly detection dataset.

  This function is similar to the example in:
  https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html

  Args:
    n_samples: Number of samples to generate in each gaussian.
    n_outliers: Number of outliers to generate.
    seed: Seed to use for random number generation.

  Returns:
    The features and labels for the dataset.
  """
  np.random.seed(seed)
  covariance = np.array([[0.5, -0.1], [0.7, 0.4]])
  cluster_1 = 0.4 * np.random.randn(n_samples, 2) @ covariance + np.array(
      [2, 2]
  )
  cluster_2 = 0.3 * np.random.randn(n_samples, 2) + np.array([-2, -2])
  outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))
  features = np.concatenate([cluster_1, cluster_2, outliers])
  labels = np.concatenate([
      np.zeros((2 * n_samples), dtype=bool),
      np.ones((n_outliers), dtype=bool),
  ])
  return features, labels


class ScikitLearnModelConverterTest(parameterized.TestCase):

  @parameterized.parameters(
      (tree.DecisionTreeRegressor(random_state=42),),
      (tree.ExtraTreeRegressor(random_state=42),),
      (ensemble.RandomForestRegressor(random_state=42),),
      (ensemble.ExtraTreesRegressor(random_state=42),),
      (
          ensemble.GradientBoostingRegressor(
              random_state=42,
          ),
      ),
      (ensemble.GradientBoostingRegressor(random_state=42, init="zero"),),
      (
          ensemble.GradientBoostingRegressor(
              random_state=42,
              init=tree.DecisionTreeRegressor(random_state=42),
          ),
      ),
  )
  def test_import_regression_model(
      self,
      sklearn_model,
  ):
    features, labels = datasets.make_regression(
        n_samples=100,
        n_features=10,
        random_state=42,
    )
    sklearn_model.fit(features, labels)
    sklearn_predictions = sklearn_model.predict(features).astype(np.float32)

    ydf_model = export_sklearn.from_sklearn(sklearn_model)
    assert isinstance(ydf_model, decision_forest_model.DecisionForestModel)
    self.assertSequenceEqual(
        ydf_model.input_feature_names(),
        [
            "features.00_of_10",
            "features.01_of_10",
            "features.02_of_10",
            "features.03_of_10",
            "features.04_of_10",
            "features.05_of_10",
            "features.06_of_10",
            "features.07_of_10",
            "features.08_of_10",
            "features.09_of_10",
        ],
    )
    self.assertEqual(ydf_model.label(), "label")

    ydf_predictions = ydf_model.predict({"features": features})
    np.testing.assert_allclose(sklearn_predictions, ydf_predictions, rtol=1e-4)

  @parameterized.parameters(
      (tree.DecisionTreeClassifier(random_state=42),),
      (tree.ExtraTreeClassifier(random_state=42),),
      (ensemble.RandomForestClassifier(random_state=42),),
      (ensemble.ExtraTreesClassifier(random_state=42),),
  )
  def test_import_classification_model(
      self,
      sklearn_model,
  ):
    features, labels = datasets.make_classification(
        n_samples=100,
        n_features=10,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=42,
    )
    sklearn_model.fit(features, labels)
    sklearn_predictions = sklearn_model.predict_proba(features).astype(
        np.float32
    )

    ydf_model = export_sklearn.from_sklearn(sklearn_model)
    ydf_features = {"features": features}
    ydf_predictions = ydf_model.predict(ydf_features)
    np.testing.assert_allclose(sklearn_predictions, ydf_predictions, rtol=1e-5)

  def test_import_anomaly_detection_model(
      self,
  ):
    train_features, _ = gen_anomaly_detection_dataset(seed=0)
    test_features, test_labels = gen_anomaly_detection_dataset(seed=1)

    # Train isolation forest
    sklearn_model = ensemble.IsolationForest(max_samples=100, random_state=0)
    sklearn_model.fit(train_features)

    # Generate golden predictions
    sklearn_predictions = -sklearn_model.score_samples(test_features)
    # Note: This is different from "sklearn_model.predict" and
    # "sklearn_model.decision_function".

    # Test quality of model
    auc = metrics.roc_auc_score(test_labels, sklearn_predictions)
    self.assertAlmostEqual(auc, 0.99333, delta=0.0001)

    ydf_model = export_sklearn.from_sklearn(sklearn_model)
    self.assertSequenceEqual(
        ydf_model.input_feature_names(),
        [
            "features.0_of_2",
            "features.1_of_2",
        ],
    )
    ydf_features = {"features": test_features}
    ydf_predictions = ydf_model.predict(ydf_features)

    _ = ydf_model.describe("text")
    _ = ydf_model.describe("html")
    _ = ydf_model.analyze_prediction({"features": test_features[:1]})
    _ = ydf_model.analyze(ydf_features)

    # YDF Predictions match SKLearn predictions
    np.testing.assert_allclose(sklearn_predictions, ydf_predictions, rtol=1e-5)

  def test_import_raises_when_unrecognised_model_provided(self):
    features, labels = datasets.make_regression(
        n_samples=100,
        n_features=10,
        random_state=42,
    )
    sklearn_model = linear_model.LinearRegression().fit(features, labels)
    with self.assertRaises(NotImplementedError):
      export_sklearn.from_sklearn(sklearn_model)

  def test_import_raises_when_sklearn_model_is_not_fit(self):
    with self.assertRaisesRegex(
        ValueError,
        "Scikit-Learn model must be fit to data before converting",
    ):
      _ = export_sklearn.from_sklearn(tree.DecisionTreeRegressor())

  def test_import_raises_when_regression_target_is_multivariate(self):
    features, labels = datasets.make_regression(
        n_samples=100,
        n_features=10,
        # This produces a two-dimensional target variable.
        n_targets=2,
        random_state=42,
    )
    sklearn_model = tree.DecisionTreeRegressor().fit(features, labels)
    with self.assertRaisesRegex(
        ValueError,
        "This model type if not supported",
    ):
      _ = export_sklearn.from_sklearn(sklearn_model)

  def test_import_raises_when_classification_target_is_multilabel(
      self,
  ):
    features, labels = datasets.make_multilabel_classification(
        n_samples=100,
        n_features=10,
        # This assigns two class labels per example.
        n_labels=2,
        random_state=42,
    )
    sklearn_model = tree.DecisionTreeClassifier().fit(features, labels)
    with self.assertRaisesRegex(
        ValueError,
        "This model type if not supported",
    ):
      _ = export_sklearn.from_sklearn(sklearn_model)

  def test_convert_raises_when_gbt_initial_estimator_is_not_tree_or_constant(
      self,
  ):
    features, labels = datasets.make_regression(
        n_samples=100,
        n_features=10,
        random_state=42,
    )
    init_estimator = linear_model.LinearRegression()
    sklearn_model = ensemble.GradientBoostingRegressor(init=init_estimator)
    sklearn_model.fit(features, labels)
    with self.assertRaises(ValueError):
      _ = export_sklearn.from_sklearn(sklearn_model)


if __name__ == "__main__":
  absltest.main()
