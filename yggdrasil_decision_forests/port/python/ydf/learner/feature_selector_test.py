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

"""Unit tests of the feature selector."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from ydf.learner import feature_selector as feature_selector_lib
from ydf.learner import specialized_learners
from ydf.utils import test_utils

_NUM_TREES = 10  #  Used for experimentation
_MIN_ACCURACY_ADULT = 0.85


class LearnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.adult = test_utils.load_datasets("adult")


class BackwardSelectionFeatureSelectorTest(LearnerTest):

  def test_auto_get_variable_importance(self):
    feature_selector = feature_selector_lib.BackwardSelectionFeatureSelector()
    _, key = feature_selector._get_variable_importance({
        "a": [],
        "b": [],
        "MEAN_DECREASE_IN_AUC_AAA": [],
        "MEAN_DECREASE_IN_ACCURACY": [],
    })
    self.assertEqual(key, "MEAN_DECREASE_IN_AUC_AAA")

  def test_manual_get_variable_importance(self):
    feature_selector = feature_selector_lib.BackwardSelectionFeatureSelector(
        variable_importance="b"
    )
    _, key = feature_selector._get_variable_importance({
        "a": [],
        "b": [],
        "MEAN_DECREASE_IN_AUC_AAA": [],
        "MEAN_DECREASE_IN_ACCURACY": [],
    })
    self.assertEqual(key, "b")

  def test_missing_variable_importance(self):
    feature_selector = feature_selector_lib.BackwardSelectionFeatureSelector(
        variable_importance="c"
    )
    with self.assertRaisesRegex(ValueError, "does not exist"):
      feature_selector._get_variable_importance({
          "a": [],
          "b": [],
      })

  def test_toy_rf(self):
    n = 2000
    f1 = np.random.uniform(size=n)
    f2 = np.random.uniform(size=n)
    f3 = np.random.uniform(size=n)
    f4 = np.random.uniform(size=n)
    l = (f1 + f2) >= 1
    ds = {"l": l, "f1": f1, "f2": f2, "f3": f3, "f4": f4}
    learner = specialized_learners.RandomForestLearner(
        label="l",
        feature_selector=feature_selector_lib.BackwardSelectionFeatureSelector(),
        compute_oob_variable_importances=True,
    )
    model = learner.train(ds)
    self.assertSameElements(model.input_feature_names(), ["f1", "f2"])

  def test_simple_rf(self):
    learner = specialized_learners.RandomForestLearner(
        label="income",
        feature_selector=feature_selector_lib.BackwardSelectionFeatureSelector(),
        compute_oob_variable_importances=True,
        num_trees=_NUM_TREES,
    )
    model = learner.train(self.adult.train_pd)
    evaluation = model.evaluate(self.adult.test_pd)
    logging.info("Evaluation: %s", evaluation)
    self.assertGreaterEqual(evaluation.accuracy, _MIN_ACCURACY_ADULT)

  def test_simple_gbt(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        feature_selector=feature_selector_lib.BackwardSelectionFeatureSelector(),
        compute_permutation_variable_importance=True,
        num_trees=_NUM_TREES,
    )
    model = learner.train(self.adult.train_pd)
    logs = model.feature_selection_logs()
    logging.info("Feature selection logs:\n%s", logs)

    self.assertIsNotNone(logs)
    self.assertLen(logs.iterations, 14)
    self.assertLen(logs.iterations[0].features, 14)
    self.assertAlmostEqual(logs.iterations[0].score, -0.762, delta=0.1)
    self.assertLen(logs.iterations[-1].features, 1)
    self.assertAlmostEqual(logs.iterations[-1].score, -1, delta=0.1)
    self.assertSetEqual(
        set(logs.iterations[0].metrics.keys()), set(["accuracy", "loss"])
    )

    self.assertSetEqual(
        set(model.input_feature_names()),
        set(logs.iterations[logs.best_iteration_idx].features),
    )

    # At least one feature was removed
    self.assertLess(len(model.input_feature_names()), 14)

    evaluation = model.evaluate(self.adult.test_pd)
    logging.info("Evaluation: %s", evaluation)
    self.assertGreaterEqual(evaluation.accuracy, _MIN_ACCURACY_ADULT)

  def test_simple_cart_fails(self):
    with self.assertRaisesRegex(
        ValueError,
        "No validation dataset was provided to the CART learner.",
    ):
      specialized_learners.CartLearner(
          label="income",
          feature_selector=feature_selector_lib.BackwardSelectionFeatureSelector(),
      ).train(self.adult.train_pd)

  def test_rf_with_validation(self):
    learner = specialized_learners.RandomForestLearner(
        label="income",
        feature_selector=feature_selector_lib.BackwardSelectionFeatureSelector(),
        num_trees=_NUM_TREES,
    )
    model = learner.train(self.adult.train_pd, valid=self.adult.test_pd)
    evaluation = model.evaluate(self.adult.test_pd)
    logging.info("Evaluation: %s", evaluation)
    self.assertGreaterEqual(evaluation.accuracy, _MIN_ACCURACY_ADULT)

  def test_rf_with_validation_and_va_fails(self):
    with self.assertRaisesRegex(
        ValueError,
        "The Random Forest learner is used both with",
    ):
      specialized_learners.RandomForestLearner(
          label="income",
          feature_selector=feature_selector_lib.BackwardSelectionFeatureSelector(),
          compute_oob_variable_importances=True,
          num_trees=_NUM_TREES,
      ).train(self.adult.train_pd, valid=self.adult.test_pd)

  def test_gbt_with_validation(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        feature_selector=feature_selector_lib.BackwardSelectionFeatureSelector(),
        num_trees=_NUM_TREES,
    )
    model = learner.train(self.adult.train_pd, valid=self.adult.test_pd)
    evaluation = model.evaluate(self.adult.test_pd)
    logging.info("Evaluation: %s", evaluation)
    self.assertGreaterEqual(evaluation.accuracy, _MIN_ACCURACY_ADULT)

  def test_gbt_with_validation_and_va(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        feature_selector=feature_selector_lib.BackwardSelectionFeatureSelector(),
        num_trees=_NUM_TREES,
        compute_permutation_variable_importance=True,
    )
    model = learner.train(self.adult.train_pd, valid=self.adult.test_pd)
    evaluation = model.evaluate(self.adult.test_pd)
    logging.info("Evaluation: %s", evaluation)
    self.assertGreaterEqual(evaluation.accuracy, _MIN_ACCURACY_ADULT)

  def test_cart_with_validation(self):
    learner = specialized_learners.CartLearner(
        label="income",
        feature_selector=feature_selector_lib.BackwardSelectionFeatureSelector(),
    )
    model = learner.train(self.adult.train_pd, valid=self.adult.test_pd)
    evaluation = model.evaluate(self.adult.test_pd)
    logging.info("Evaluation: %s", evaluation)
    self.assertGreaterEqual(evaluation.accuracy, _MIN_ACCURACY_ADULT)

  def test_structural_rf_fails(self):
    with self.assertRaisesRegex(
        ValueError,
        "Out-of-bag variable importance computation is not enabled for the"
        " Random Forest learner",
    ):
      specialized_learners.RandomForestLearner(
          label="income",
          feature_selector=feature_selector_lib.BackwardSelectionFeatureSelector(),
          num_trees=_NUM_TREES,
      ).train(self.adult.train_pd)

  def test_structural_gbt_fails(self):
    with self.assertRaisesRegex(
        ValueError,
        "Permutation variable importance computation is not enabled for the"
        " Gradient Boosted Trees learner",
    ):
      specialized_learners.GradientBoostedTreesLearner(
          label="income",
          feature_selector=feature_selector_lib.BackwardSelectionFeatureSelector(),
          num_trees=_NUM_TREES,
      ).train(self.adult.train_pd)

  def test_structural_rf(self):
    learner = specialized_learners.RandomForestLearner(
        label="income",
        feature_selector=feature_selector_lib.BackwardSelectionFeatureSelector(
            allow_structural_variable_importance=True
        ),
        num_trees=_NUM_TREES,
    )
    model = learner.train(self.adult.train_pd)
    evaluation = model.evaluate(self.adult.test_pd)
    logging.info("Evaluation: %s", evaluation)
    self.assertGreaterEqual(evaluation.accuracy, _MIN_ACCURACY_ADULT)

  def test_structural_gbt(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        feature_selector=feature_selector_lib.BackwardSelectionFeatureSelector(
            allow_structural_variable_importance=True
        ),
        num_trees=_NUM_TREES,
    )
    model = learner.train(self.adult.train_pd)
    evaluation = model.evaluate(self.adult.test_pd)
    logging.info("Evaluation: %s", evaluation)
    self.assertGreaterEqual(evaluation.accuracy, _MIN_ACCURACY_ADULT)

  def test_structural_cart(self):
    learner = specialized_learners.CartLearner(
        label="income",
        feature_selector=feature_selector_lib.BackwardSelectionFeatureSelector(
            allow_structural_variable_importance=True
        ),
    )
    model = learner.train(self.adult.train_pd)
    evaluation = model.evaluate(self.adult.test_pd)
    logging.info("Evaluation: %s", evaluation)
    self.assertGreaterEqual(evaluation.accuracy, _MIN_ACCURACY_ADULT)


if __name__ == "__main__":
  absltest.main()
