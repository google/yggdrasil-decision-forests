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

"""Tests for isolation forest learners."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from sklearn import metrics

from ydf.learner import generic_learner
from ydf.learner import learner_test_utils
from ydf.learner import specialized_learners
from ydf.model.tree import condition as condition_lib
from ydf.model.tree import node as node_lib


class IsolationForestLearnerTest(learner_test_utils.LearnerTest):

  @parameterized.parameters(False, True)
  def test_gaussians_train_and_analyze(self, with_labels: bool):
    if with_labels:
      learner = specialized_learners.IsolationForestLearner(label="label")
    else:
      learner = specialized_learners.IsolationForestLearner(
          features=["features.0_of_2", "features.1_of_2"]
      )
    model = learner.train(self.gaussians.train_pd)
    predictions = model.predict(self.gaussians.test_pd)

    auc = metrics.roc_auc_score(self.gaussians.test_pd["label"], predictions)
    logging.info("auc:%s", auc)
    self.assertGreaterEqual(auc, 0.99)

    _ = model.describe("text")
    _ = model.describe("html")
    _ = model.analyze_prediction(self.gaussians.test_pd.iloc[:1])
    analysis = model.analyze(self.gaussians.test_pd)
    if with_labels:
      self.assertLen(analysis.variable_importances(), 7)
    else:
      self.assertLen(analysis.variable_importances(), 3)

  def test_gaussians_evaluation_default_task(self):
    learner = specialized_learners.IsolationForestLearner(label="label")
    model = learner.train(self.gaussians.train)
    with self.assertRaisesRegex(
        ValueError,
        ".*evaluate the model as a classification model.*",
    ):
      _ = model.evaluate(self.gaussians.test)

  def test_gaussians_evaluation_no_label(self):
    learner = specialized_learners.IsolationForestLearner(
        features=["features.0_of_2", "features.1_of_2"]
    )
    model = learner.train(self.gaussians.train_pd)
    with self.assertRaisesRegex(
        ValueError,
        ".*A model cannot be evaluated without a label..*",
    ):
      _ = model.evaluate(
          self.gaussians.test,
          evaluation_task=generic_learner.Task.CLASSIFICATION,
      )

  def test_gaussians_evaluation_with_label(self):
    learner = specialized_learners.IsolationForestLearner(label="label")
    model = learner.train(self.gaussians.train)
    evaluation = model.evaluate(
        self.gaussians.test,
        evaluation_task=generic_learner.Task.CLASSIFICATION,
    )
    self.assertSameElements(
        evaluation.to_dict().keys(),
        [
            "num_examples",
            "num_examples_weighted",
            "accuracy",
            "characteristic_0:name",
            "characteristic_0:pr_auc",
            "characteristic_0:roc_auc",
            "confusion_matrix",
            "loss",
        ],
    )
    self.assertAlmostEqual(evaluation.accuracy, 0.975, delta=0.015)
    self.assertAlmostEqual(evaluation.loss, 0.52, delta=0.01)

  def test_max_depth_gaussians_subsample_ratio(self):
    learner = specialized_learners.IsolationForestLearner(
        features=["features.0_of_2", "features.1_of_2"],
        subsample_ratio=0.9,
    )
    self.assertEqual(learner.hyperparameters["subsample_ratio"], 0.9)
    model = learner.train(self.gaussians.train_pd)

    max_depth = max([
        learner_test_utils.get_tree_depth(t.root, 0)
        for t in model.get_all_trees()
    ])
    self.assertEqual(max_depth, 8)

  def test_max_depth_gaussians_subsample_count(self):
    learner = specialized_learners.IsolationForestLearner(
        features=["features.0_of_2", "features.1_of_2"],
        subsample_count=128,
    )
    self.assertEqual(learner.hyperparameters["subsample_count"], 128)
    model = learner.train(self.gaussians.train_pd)

    max_depth = max([
        learner_test_utils.get_tree_depth(t.root, 0)
        for t in model.get_all_trees()
    ])
    self.assertEqual(max_depth, 7)

  def test_max_depth_gaussians_max_depth(self):
    learner = specialized_learners.IsolationForestLearner(
        features=["features.0_of_2", "features.1_of_2"],
        subsample_ratio=1.0,
        max_depth=10,
    )
    model = learner.train(self.gaussians.train_pd)

    max_depth = max([
        learner_test_utils.get_tree_depth(t.root, 0)
        for t in model.get_all_trees()
    ])
    self.assertEqual(max_depth, 10)

  def test_illegal_agument_combination_constructor(self):
    with self.assertRaisesRegex(
        ValueError,
        ".*Only one of the following hyperparameters can be set:"
        " (subsample_ratio, subsample_count|subsample_count,"
        " subsample_ratio).*",
    ):
      _ = specialized_learners.IsolationForestLearner(
          features=["f1", "f2"], subsample_count=128, subsample_ratio=0.5
      )

  def test_illegal_agument_combination_explicit_call(self):
    learner = specialized_learners.IsolationForestLearner(
        features=["f1", "f2"], subsample_count=128
    )
    learner.hyperparameters["subsample_ratio"] = 0.5
    with self.assertRaises(ValueError):
      learner.validate_hyperparameters()

  def test_gaussians_oblique(self):
    learner = specialized_learners.IsolationForestLearner(
        features=["features.0_of_2", "features.1_of_2"],
        split_axis="SPARSE_OBLIQUE",
        num_trees=5,
    )
    model = learner.train(self.gaussians.train_pd)
    first_tree = model.get_tree(0)
    first_root = first_tree.root
    self.assertIsInstance(first_root, node_lib.NonLeaf)
    self.assertIsInstance(
        first_root.condition, condition_lib.NumericalSparseObliqueCondition
    )

  def test_weights_not_supported(self):
    ds = {
        "f": np.array([1, 2, 3, 4, 5, 6], dtype=float),
        "w": np.array([1, 2, 3, 4, 5, 6], dtype=float),
        "label": np.array(["a", "a", "b", "b", "b", "c"]),
    }
    with self.assertRaisesRegex(
        ValueError, "Isolation forest does not support weights"
    ):
      _ = specialized_learners.IsolationForestLearner(
          weights="w",
      ).train(ds)

  def test_class_weights_not_supported_with_label(self):
    ds = {
        "f": np.array([1, 2, 3, 4, 5, 6], dtype=float),
        "w": np.array([1, 2, 3, 4, 5, 6], dtype=float),
        "label": np.array(["a", "a", "b", "b", "b", "c"]),
    }
    with self.assertRaisesRegex(
        ValueError, "Isolation forest does not support weights"
    ):
      _ = specialized_learners.IsolationForestLearner(
          label="label",
          class_weights={"a": 1.0, "b": 1.0, "c": 1.0},
      ).train(ds)

  def test_class_weights_not_supported_no_label(self):
    ds = {
        "f": np.array([1, 2, 3, 4, 5, 6], dtype=float),
        "w": np.array([1, 2, 3, 4, 5, 6], dtype=float),
        "label": np.array(["a", "a", "b", "b", "b", "c"]),
    }
    with self.assertRaisesRegex(
        ValueError,
        "Class weights require a label and are not supported for unsupervised"
        " learning",
    ):
      _ = specialized_learners.IsolationForestLearner(
          class_weights={"a": 1.0, "b": 1.0, "c": 1.0},
      ).train(ds)


if __name__ == "__main__":
  absltest.main()
