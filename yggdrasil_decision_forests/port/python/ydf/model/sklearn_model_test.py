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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from sklearn import datasets
from sklearn import ensemble
from sklearn import linear_model
from sklearn import tree
from ydf.model import export_sklearn
from ydf.model.decision_forest_model import decision_forest_model


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
        "Only scalar regression and single-label classification are supported.",
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
        "Only scalar regression and single-label classification are supported.",
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
