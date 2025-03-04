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

"""Tests for CART learners."""

import os

from absl.testing import absltest
import numpy as np

from ydf.dataset import dataspec
from ydf.learner import generic_learner
from ydf.learner import learner_test_utils
from ydf.learner import specialized_learners
from ydf.learner import tuner as tuner_lib
from ydf.utils import test_utils

Column = dataspec.Column


class CARTLearnerTest(learner_test_utils.LearnerTest):

  def test_adult(self):
    learner = specialized_learners.CartLearner(label="income")

    model, _, _ = self._check_adult_model(
        learner=learner, minimum_accuracy=0.853
    )
    self.assertGreater(model.self_evaluation().accuracy, 0.84)

  def test_adult_with_validation(self):
    learner = specialized_learners.CartLearner(label="income")

    model, evaluation, _ = self._check_adult_model(
        learner=learner, minimum_accuracy=0.853, valid=self.adult.test
    )
    # Make sure the test dataset is effectively used for validation.
    self.assertEqual(model.self_evaluation().accuracy, evaluation.accuracy)
    self.assertEqual(
        model.self_evaluation().num_examples,
        self.adult.test_pd.shape[0],
    )

  def test_two_center_regression(self):
    learner = specialized_learners.CartLearner(
        label="target", task=generic_learner.Task.REGRESSION
    )
    model = learner.train(self.two_center_regression.train)
    evaluation = model.evaluate(self.two_center_regression.test)
    self.assertAlmostEqual(evaluation.rmse, 116, delta=0.5)

  def test_monotonic_non_compatible_learner(self):
    with self.assertRaisesRegex(
        test_utils.AbslInvalidArgumentError,
        "The learner CART does not support monotonic constraints",
    ):
      _ = specialized_learners.CartLearner(
          label="label", features=[Column("feature", monotonic=+1)]
      )

  def test_tuner_manual(self):
    tuner = tuner_lib.RandomSearchTuner(num_trials=5)
    tuner.choice("min_examples", [1, 2, 5, 10])
    tuner.choice("max_depth", [3, 4, 5, 6])
    learner = specialized_learners.CartLearner(label="income", tuner=tuner)

    model, _, _ = self._check_adult_model(learner, minimum_accuracy=0.83)
    logs = model.hyperparameter_optimizer_logs()
    self.assertIsNotNone(logs)
    self.assertLen(logs.trials, 5)

  def test_tuner_manual_on_validation(self):
    tuner = tuner_lib.RandomSearchTuner(num_trials=5)
    tuner.choice("min_examples", [1, 2, 5, 10])
    tuner.choice("max_depth", [3, 4, 5, 6])
    learner = specialized_learners.CartLearner(label="income", tuner=tuner)

    model, evaluation, _ = self._check_adult_model(
        learner, minimum_accuracy=0.83, valid=self.adult.test
    )
    # Make sure the test dataset is effectively used for validation.
    self.assertEqual(model.self_evaluation().accuracy, evaluation.accuracy)
    self.assertEqual(
        model.self_evaluation().num_examples, self.adult.test_pd.shape[0]
    )
    logs = model.hyperparameter_optimizer_logs()
    self.assertIsNotNone(logs)
    self.assertLen(logs.trials, 5)

  def test_tuner_predefined(self):
    tuner = tuner_lib.RandomSearchTuner(
        num_trials=5, automatic_search_space=True
    )
    learner = specialized_learners.CartLearner(label="income", tuner=tuner)

    model, _, _ = self._check_adult_model(learner, minimum_accuracy=0.83)
    logs = model.hyperparameter_optimizer_logs()
    self.assertIsNotNone(logs)
    self.assertLen(logs.trials, 5)

  def test_oblique_weights_default(self):
    learner = specialized_learners.CartLearner(
        label="label",
        max_depth=2,
        split_axis="SPARSE_OBLIQUE",
    )
    f1 = np.linspace(-1, 1, 50) ** 2
    f2 = np.linspace(1.5, -0.5, 50) ** 2
    label = (0.2 * f1 + 0.7 * f2 >= 0.25).astype(int)
    ds = {"f1": f1, "f2": f2, "label": label}
    model = learner.train(ds)
    root_weights = model.get_tree(0).root.condition.weights
    self.assertTrue(all(x in (-1.0, 1.0) for x in root_weights))

  def test_oblique_weights_binary(self):
    learner = specialized_learners.CartLearner(
        label="label",
        max_depth=2,
        split_axis="SPARSE_OBLIQUE",
        sparse_oblique_weights="BINARY",
    )
    f1 = np.linspace(-1, 1, 50) ** 2
    f2 = np.linspace(1.5, -0.5, 50) ** 2
    label = (0.2 * f1 + 0.7 * f2 >= 0.25).astype(int)
    ds = {"f1": f1, "f2": f2, "label": label}
    model = learner.train(ds)
    root_weights = model.get_tree(0).root.condition.weights
    self.assertTrue(all(x in (-1.0, 1.0) for x in root_weights))

  def test_oblique_weights_continuous(self):
    learner = specialized_learners.CartLearner(
        label="label",
        max_depth=2,
        split_axis="SPARSE_OBLIQUE",
        sparse_oblique_weights="CONTINUOUS",
    )
    f1 = np.linspace(-1, 1, 50) ** 2
    f2 = np.linspace(1.5, -0.5, 50) ** 2
    label = (0.2 * f1 + 0.7 * f2 >= 0.25).astype(int)
    ds = {"f1": f1, "f2": f2, "label": label}
    model = learner.train(ds)
    root_weights = model.get_tree(0).root.condition.weights
    self.assertFalse(all(x in (-1.0, 1.0) for x in root_weights))

  def test_oblique_weights_power_of_two(self):
    learner = specialized_learners.CartLearner(
        label="label",
        max_depth=2,
        split_axis="SPARSE_OBLIQUE",
        sparse_oblique_weights="POWER_OF_TWO",
    )
    f1 = np.linspace(-1, 1, 50) ** 2
    f2 = np.linspace(1.5, -0.5, 50) ** 2
    label = (0.2 * f1 + 0.7 * f2 >= 0.25).astype(int)
    ds = {"f1": f1, "f2": f2, "label": label}
    model = learner.train(ds)
    root_weights = model.get_tree(0).root.condition.weights
    acceptable_weights = [x * 2**y for x in (1.0, -1.0) for y in range(-3, 4)]
    self.assertTrue(all(x in acceptable_weights for x in root_weights))
    learner.hyperparameters[
        "sparse_oblique_weights_power_of_two_min_exponent"
    ] = 4
    learner.hyperparameters[
        "sparse_oblique_weights_power_of_two_max_exponent"
    ] = 7
    model_2 = learner.train(ds)
    root_weights_2 = model_2.get_tree(0).root.condition.weights
    acceptable_weights_2 = [x * 2**y for x in (1.0, -1.0) for y in range(4, 8)]
    self.assertTrue(all(x in acceptable_weights_2 for x in root_weights_2))

  def test_oblique_weights_integer(self):
    learner = specialized_learners.CartLearner(
        label="label",
        max_depth=2,
        split_axis="SPARSE_OBLIQUE",
        sparse_oblique_weights="INTEGER",
    )
    f1 = np.linspace(-1, 1, 200) ** 2
    f2 = np.linspace(1.5, -0.5, 200) ** 2
    label = (0.2 * f1 + 0.7 * f2 >= 0.25).astype(int)
    ds = {"f1": f1, "f2": f2, "label": label}
    model = learner.train(ds)
    root_weights = model.get_tree(0).root.condition.weights
    acceptable_weights = [x * y for x in (1.0, -1.0) for y in range(0, 6)]
    self.assertTrue(all(x in acceptable_weights for x in root_weights))
    learner.hyperparameters["sparse_oblique_weights_integer_minimum"] = 7
    learner.hyperparameters["sparse_oblique_weights_integer_maximum"] = 14
    model_2 = learner.train(ds)
    root_weights_2 = model_2.get_tree(0).root.condition.weights
    acceptable_weights_2 = [x * y for x in (1.0, -1.0) for y in range(7, 15)]
    self.assertTrue(all(x in acceptable_weights_2 for x in root_weights_2))

  def test_no_two_weight_definitions(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot specify both `weights` and `class_weights`."
    ):
      _ = specialized_learners.CartLearner(
          label="label",
          weights="w",
          class_weights={"a": 1.1},
      )

  def test_fails_missing_class_weights(self):
    ds = {
        "f": np.array([1, 2, 3, 4, 5, 6], dtype=float),
        "label": np.array(["a", "a", "b", "b", "b", "c"]),
    }
    with self.assertRaisesRegex(
        ValueError,
        'The categorical weight value "c" in the column "label" does not have a'
        " corresponding weight.",
    ):
      _ = specialized_learners.CartLearner(
          label="label",
          min_examples=1,
          features=["f"],
          max_depth=2,
          # Missing value for class "c"
          class_weights={
              "a": 1.0,
              "b": 1.0,
          },
      ).train(ds)

  def test_fails_negative_class_weights(self):
    ds = {
        "f": np.array([1, 2, 3, 4, 5, 6], dtype=float),
        "label": np.array(["a", "a", "b", "b", "b", "c"]),
    }
    with self.assertRaisesRegex(
        ValueError,
        'The categorical weight value "c" is defined with a negative weight.',
    ):
      _ = specialized_learners.CartLearner(
          label="label",
          min_examples=1,
          features=["f"],
          max_depth=2,
          # Negative value for class "c"
          class_weights={"a": 1.0, "b": 1.0, "c": -1.0},
      ).train(ds)

  def test_class_weights(self):
    ds = {
        "f": np.array([1, 2, 3, 4, 5, 6], dtype=float),
        "label": np.array(["a", "a", "b", "b", "b", "c"]),
    }
    model = specialized_learners.CartLearner(
        label="label",
        min_examples=1,
        features=["f"],
        max_depth=2,
        # Over-weight class c to force a "a, b" vs. "c" split.
        class_weights={"a": 1.0, "b": 1.0, "c": 100.0},
    ).train(ds)
    tree = model.get_tree(0)
    self.assertFalse(tree.root.is_leaf)
    root_condition = tree.root.condition
    self.assertEqual(root_condition.threshold, 5.5)

  def test_zero_class_weights(self):
    ds = {
        "f": np.array([1, 2, 3, 4, 5, 6], dtype=float),
        "label": np.array(["a", "a", "b", "b", "b", "c"]),
    }
    model = specialized_learners.CartLearner(
        label="label",
        min_examples=1,
        features=["f"],
        max_depth=2,
        # Class "b" has 0 weight, so the split becomes "a" vs. "b, c"
        class_weights={"a": 1.0, "b": 0.0, "c": 100.0},
    ).train(ds)
    tree = model.get_tree(0)
    self.assertFalse(tree.root.is_leaf)
    root_condition = tree.root.condition
    self.assertEqual(root_condition.threshold, 2.5)

  def test_non_unicode_feature_values(self):
    text = "feature,label\nCafé,oné\nfoobar,zéro"
    encoded_text = text.encode("windows-1252")
    with self.assertRaises(UnicodeDecodeError):
      encoded_text.decode()
    data_path = self.create_tempfile().full_path
    model_path = self.create_tempdir().full_path
    with open(data_path, "wb") as f:
      f.write(encoded_text)
    model = specialized_learners.CartLearner(
        label="label",
        min_examples=1,
        min_vocab_frequency=1,
    ).train("csv:" + data_path)
    evaluation = model.evaluate("csv:" + data_path)
    self.assertEqual(evaluation.accuracy, 1)
    model.save(model_path)


if __name__ == "__main__":
  absltest.main()
