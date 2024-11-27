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

"""Testing Metrics."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics

from ydf.metric import evaluate
from ydf.model import generic_model
from ydf.model import model_lib
from ydf.utils import test_utils


class EvaluatePredictionTest(parameterized.TestCase):

  def test_adult(self):
    model_dir = os.path.join(test_utils.ydf_test_data_path(), "model")
    model = model_lib.load_model(
        os.path.join(model_dir, "adult_binary_class_gbdt")
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    ds = pd.read_csv(dataset_path)
    model_evaluation = model.evaluate(ds)
    model_predictions: np.ndarray = model.predict(ds)
    labels = ds["income"].to_numpy()
    label_classes = model.label_classes()
    task = generic_model.Task.CLASSIFICATION
    standalone_evaluation = evaluate.evaluate_predictions(
        model_predictions, labels, task=task, label_classes=label_classes
    )
    self.assertEqual(
        model_evaluation._evaluation_proto.classification,
        standalone_evaluation._evaluation_proto.classification,
    )

  def test_abalone(self):
    model_dir = os.path.join(test_utils.ydf_test_data_path(), "model")
    model = model_lib.load_model(
        os.path.join(model_dir, "abalone_regression_gbdt")
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "abalone.csv"
    )
    ds = pd.read_csv(dataset_path)
    model_evaluation = model.evaluate(ds)
    model_predictions: np.ndarray = model.predict(ds)
    labels = ds["Rings"].to_numpy()
    task = generic_model.Task.REGRESSION
    standalone_evaluation = evaluate.evaluate_predictions(
        model_predictions, labels, task=task
    )
    self.assertEqual(
        model_evaluation._evaluation_proto.regression,
        standalone_evaluation._evaluation_proto.regression,
    )

  def test_iris(self):
    model_dir = os.path.join(test_utils.ydf_test_data_path(), "model")
    model = model_lib.load_model(
        os.path.join(model_dir, "iris_multi_class_gbdt")
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "iris.csv"
    )
    ds = pd.read_csv(dataset_path)
    model_evaluation = model.evaluate(ds)
    model_predictions: np.ndarray = model.predict(ds)
    labels = ds["class"].to_numpy()
    label_classes = model.label_classes()
    task = generic_model.Task.CLASSIFICATION
    standalone_evaluation = evaluate.evaluate_predictions(
        model_predictions, labels, task=task, label_classes=label_classes
    )
    self.assertEqual(
        model_evaluation._evaluation_proto.classification,
        standalone_evaluation._evaluation_proto.classification,
    )

  def test_synthetic_ranking(self):
    model_dir = os.path.join(test_utils.ydf_test_data_path(), "model")
    model = model_lib.load_model(
        os.path.join(model_dir, "synthetic_ranking_gbdt")
    )
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "synthetic_ranking_test.csv"
    )
    ds = pd.read_csv(dataset_path)
    model_evaluation = model.evaluate(ds)
    model_predictions: np.ndarray = model.predict(ds)
    labels = ds["LABEL"].to_numpy()
    ranking_group = ds["GROUP"].to_numpy()
    task = generic_model.Task.RANKING
    standalone_evaluation = evaluate.evaluate_predictions(
        model_predictions, labels, task=task, ranking_groups=ranking_group
    )
    self.assertAlmostEqual(model_evaluation.ndcg, standalone_evaluation.ndcg)
    self.assertAlmostEqual(model_evaluation.mrr, standalone_evaluation.mrr)

  @parameterized.parameters(
      (generic_model.Task.CATEGORICAL_UPLIFT,),
      (generic_model.Task.NUMERICAL_UPLIFT,),
  )
  def test_fails_for_unsupported_uplift_tasks(self, task):
    predictions = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 2, 3])
    with self.assertRaisesRegex(ValueError, "Uplift evaluation not supported."):
      evaluate.evaluate_predictions(predictions, labels, task=task)

  def test_fails_for_unsupported_anomaly_detection_task(self):
    predictions = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 2, 3])
    with self.assertRaisesRegex(
        ValueError,
        "Anomaly detection models must be evaluated as binary classification.",
    ):
      evaluate.evaluate_predictions(
          predictions, labels, task=generic_model.Task.ANOMALY_DETECTION
      )

  @parameterized.parameters(
      (generic_model.Task.CLASSIFICATION,),
      (generic_model.Task.REGRESSION,),
      (generic_model.Task.RANKING,),
  )
  def test_labels_shape(self, task):
    predictions = np.array([0.1, 0.2, 0.3])
    labels = np.array([[1, 2, 3]])
    with self.assertRaisesRegex(ValueError, "The labels must be a 1D array."):
      evaluate.evaluate_predictions(predictions, labels, task=task)

  def test_invalid_labels_type_binary_classification(self):
    predictions = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 2, 3], dtype=np.float32)
    with self.assertRaisesRegex(
        ValueError,
        "Floating-point labels are not supported, use integer or string"
        " labels.",
    ):
      evaluate.evaluate_predictions(
          predictions, labels, task=generic_model.Task.CLASSIFICATION
      )

  @parameterized.parameters(
      (generic_model.Task.RANKING,),
      (generic_model.Task.REGRESSION,),
  )
  def test_invalid_2d_predictions_ranking_regression(self, task):
    predictions = np.array([[0.1, 0.2]])
    labels = np.array([1.0, 2.0])
    with self.assertRaisesRegex(
        ValueError, "Predictions must be a 1D float array"
    ):
      evaluate.evaluate_predictions(predictions, labels, task=task)

  @parameterized.product(
      task=[
          generic_model.Task.CLASSIFICATION,
          generic_model.Task.REGRESSION,
          generic_model.Task.RANKING,
      ],
      dtype=[np.int32, np.str_, np.bool_],
  )
  def test_invalid_predictions_dtype(self, task, dtype):
    predictions = np.array([0.1, 0.2], dtype=dtype)
    labels = np.array([1.0, 2.0])
    with self.assertRaisesRegex(
        ValueError,
        "Predictions must be floating point values",
    ):
      evaluate.evaluate_predictions(predictions, labels, task=task)

  @parameterized.parameters(
      (generic_model.Task.CLASSIFICATION, np.int32),
      (generic_model.Task.REGRESSION, np.float32),
      (generic_model.Task.RANKING, np.float32),
  )
  def test_non_matching_weights_and_labels(self, task, pred_dtype):
    predictions = np.array([0.1, 0.2, 0.3])
    labels = np.array([1.0, 2.0], dtype=pred_dtype)
    with self.assertRaisesRegex(
        ValueError,
        "There must be one prediction per example.",
    ):
      evaluate.evaluate_predictions(predictions, labels, task=task)

  @parameterized.parameters(
      (generic_model.Task.CLASSIFICATION, np.int32),
      (generic_model.Task.REGRESSION, np.float32),
  )
  def test_invalid_ranking_group_for_non_ranking_task(self, task, pred_dtype):
    predictions = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 1, 2], dtype=pred_dtype)
    ranking_groups = np.array([1, 2, 3])
    with self.assertRaisesRegex(
        ValueError,
        "Ranking groups must only be specified for ranking tasks.",
    ):
      evaluate.evaluate_predictions(
          predictions, labels, task=task, ranking_groups=ranking_groups
      )

  def test_weights(self):
    predictions = np.array([1.0, 2.0, 3.0])
    labels = np.array([2.0, 2.0, 5.0])
    weights = np.array([1.0, 2.0, 3.0])
    evaluation = evaluate.evaluate_predictions(
        predictions,
        labels,
        task=generic_model.Task.REGRESSION,
        weights=weights,
    )
    self.assertAlmostEqual(evaluation.rmse, np.sqrt((1 + 3 * 4) / 6))

  def test_integer_label_name(self):
    predictions = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 0, 1])
    label_classes = ["0", "1"]
    evaluation = evaluate.evaluate_predictions(
        predictions,
        labels,
        task=generic_model.Task.CLASSIFICATION,
        label_classes=label_classes,
    )
    self.assertAlmostEqual(evaluation.accuracy, 1 / 3)

  def test_sklearn_model_evaluation(self):
    features, labels = datasets.make_friedman1(
        n_samples=100,
        n_features=10,
        random_state=42,
    )
    model = linear_model.LinearRegression()
    model.fit(features, labels)
    sklearn_predictions = model.predict(features).astype(np.float32)
    evaluation = evaluate.evaluate_predictions(
        sklearn_predictions,
        labels,
        task=generic_model.Task.REGRESSION,
    )
    sklearn_rmse = np.sqrt(
        metrics.mean_squared_error(labels, sklearn_predictions)
    )
    # Make sure the dataset is not trivial.
    self.assertGreaterEqual(sklearn_rmse, 1.0)
    self.assertAlmostEqual(evaluation.rmse, sklearn_rmse)

  @parameterized.product(
      predictions=[
          np.array([0.3, 0.1, 0.2]),
          np.array([[0.7, 0.3], [0.9, 0.1], [0.8, 0.2]]),
      ],
      labels=[np.array([1, 0, 1]), np.array(["b", "a", "b"])],
      label_classes=[None, ["a", "b"]],
  )
  def test_binary_classification(self, predictions, labels, label_classes):
    if label_classes is None and not np.issubdtype(labels.dtype, np.integer):
      self.skipTest("Unsupported")
    evaluation = evaluate.evaluate_predictions(
        predictions,
        labels,
        task=generic_model.Task.CLASSIFICATION,
        label_classes=label_classes,
    )
    self.assertAlmostEqual(evaluation.accuracy, 1 / 3)

  @parameterized.parameters(
      (np.array([1, 0, 1]), None),
      (np.array([1, 0, 2]), ["a", "b", "c"]),
      (np.array(["b", "a", "b"]), ["a", "b"]),
  )
  def test_binary_classification_invalid_shape(self, labels, label_classes):
    predictions = np.array([[0.3], [0.1], [0.2]])
    with self.assertRaisesRegex(
        ValueError,
        r"Classification probabilities should have shape",
    ):
      evaluate.evaluate_predictions(
          predictions,
          labels,
          task=generic_model.Task.CLASSIFICATION,
          label_classes=label_classes,
      )

  @parameterized.parameters(
      (np.array([1, 0, 2]), None),
      (np.array([1, 0, 2]), ["a", "b", "c"]),
      (np.array(["b", "a", "c"]), ["a", "b", "c"]),
  )
  def test_multiclass_classification(self, labels, label_classes):
    predictions = np.array(
        [[0.15, 0.6, 0.25], [0.1, 0.7, 0.2], [0.2, 0.8, 0.0]]
    )
    evaluation = evaluate.evaluate_predictions(
        predictions,
        labels,
        task=generic_model.Task.CLASSIFICATION,
        label_classes=label_classes,
    )
    self.assertAlmostEqual(evaluation.accuracy, 1 / 3)

  def test_classification_string_labels_no_label_classes_fails(self):
    predictions = np.array([0.3, 0.1, 0.2])
    labels = np.array(["b", "a", "b"])
    with self.assertRaisesRegex(
        ValueError,
        "When using string labels, label_classes must be provided",
    ):
      evaluate.evaluate_predictions(
          predictions,
          labels,
          task=generic_model.Task.CLASSIFICATION,
      )

  def test_classification_string_labels_label_classes_ordering(self):
    predictions = np.array([0.3, 0.1, 0.2])
    labels = np.array(["b", "a", "b"])
    evaluation = evaluate.evaluate_predictions(
        predictions,
        labels,
        task=generic_model.Task.CLASSIFICATION,
        label_classes=["a", "b"],
    )
    self.assertAlmostEqual(evaluation.accuracy, 1 / 3)
    # Make sure the order is respected
    evaluation_flipped = evaluate.evaluate_predictions(
        predictions,
        labels,
        task=generic_model.Task.CLASSIFICATION,
        label_classes=["b", "a"],
    )
    self.assertAlmostEqual(evaluation_flipped.accuracy, 2 / 3)

  def test_classification_string_labels_missing_label_class(self):
    predictions = np.array([0.1, 0.2, 0.3])
    labels = np.array(["b", "a", "c"])
    label_classes = ["a", "b"]
    with self.assertRaisesRegex(
        ValueError,
        "Found label not in label_classes: c",
    ):
      evaluate.evaluate_predictions(
          predictions,
          labels,
          task=generic_model.Task.CLASSIFICATION,
          label_classes=label_classes,
      )

  @parameterized.product(
      labels=[
          np.array([1, 0, 2]),
          np.array(["b", "a", "c"]),
      ],
      label_classes=[["a", "b"], ["a", "b", "c", "d"]],
  )
  def test_multiclass_classification_invalid_number_of_label_classes(
      self, labels, label_classes
  ):
    predictions = np.array(
        [[0.15, 0.6, 0.25], [0.1, 0.7, 0.2], [0.2, 0.8, 0.0]]
    )
    with self.assertRaisesRegex(
        ValueError,
        "The number of label classes and the number of prediction dimensions do"
        " not match.",
    ):
      evaluate.evaluate_predictions(
          predictions,
          labels,
          task=generic_model.Task.CLASSIFICATION,
          label_classes=label_classes,
      )

  @parameterized.product(
      predictions=[
          np.array([0.1, 0.2, 0.3]),
          np.array([[0.7, 0.3], [0.9, 0.1], [0.8, 0.2]]),
      ],
      labels=[
          np.array([1, 0, 2]),
          np.array(["b", "a", "c"]),
      ],
      label_classes=[["a"], ["a", "b", "c"]],
  )
  def test_binary_classification_invalid_number_of_label_classes(
      self, predictions, labels, label_classes
  ):
    with self.assertRaisesRegex(
        ValueError,
        "The number of label classes and the number of prediction dimensions do"
        " not match.",
    ):
      evaluate.evaluate_predictions(
          predictions,
          labels,
          task=generic_model.Task.CLASSIFICATION,
          label_classes=label_classes,
      )


if __name__ == "__main__":
  absltest.main()
