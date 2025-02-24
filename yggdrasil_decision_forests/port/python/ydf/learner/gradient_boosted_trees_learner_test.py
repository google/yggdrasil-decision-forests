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

"""Tests for Gradient Boosted Trees learners."""

import os

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
import numpy as np
import numpy.testing as npt
import pandas as pd

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.dataset import dataspec
from ydf.learner import generic_learner
from ydf.learner import learner_test_utils
from ydf.learner import specialized_learners
from ydf.learner import tuner as tuner_lib
from ydf.model.decision_forest_model import decision_forest_model
from ydf.utils import test_utils

ProtoMonotonicConstraint = abstract_learner_pb2.MonotonicConstraint
Column = dataspec.Column


class GradientBoostedTreesLearnerTest(learner_test_utils.LearnerTest):

  def test_adult(self):
    learner = specialized_learners.GradientBoostedTreesLearner(label="income")

    self._check_adult_model(learner=learner, minimum_accuracy=0.869)

  def test_ranking(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="LABEL",
        ranking_group="GROUP",
        task=generic_learner.Task.RANKING,
    )

    model = learner.train(self.synthetic_ranking.train)
    evaluation = model.evaluate(self.synthetic_ranking.test)
    self.assertGreaterEqual(evaluation.ndcg, 0.6893)
    self.assertLessEqual(evaluation.ndcg, 0.7457)

  def test_ranking_pd(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="LABEL",
        ranking_group="GROUP",
        task=generic_learner.Task.RANKING,
    )

    model = learner.train(self.synthetic_ranking.train_pd)
    evaluation = model.evaluate(self.synthetic_ranking.test_pd)
    self.assertGreaterEqual(evaluation.ndcg, 0.6893)
    self.assertLessEqual(evaluation.ndcg, 0.7457)

  def test_ranking_path(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="LABEL",
        ranking_group="GROUP",
        task=generic_learner.Task.RANKING,
    )

    model = learner.train(self.synthetic_ranking.train_path)
    evaluation = model.evaluate(self.synthetic_ranking.test_path)
    self.assertGreaterEqual(evaluation.ndcg, 0.6893)
    self.assertLessEqual(evaluation.ndcg, 0.7457)

  @parameterized.named_parameters(
      {
          "testcase_name": "ndcg@2",
          "truncation": 2,
          "expected_ndcg": 0.7196,
          "delta": 0.0358,
      },
      {
          "testcase_name": "ndcg@5",
          "truncation": 5,
          "expected_ndcg": 0.7168,
          "delta": 0.0330,
      },
      {
          "testcase_name": "ndcg@10",
          "truncation": 10,
          "expected_ndcg": 0.7199,
          "delta": 0.0298,
      },
  )
  def test_ranking_ndcg_truncation(self, truncation, expected_ndcg, delta):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="LABEL",
        ranking_group="GROUP",
        task=generic_learner.Task.RANKING,
        ndcg_truncation=truncation,
    )

    model = learner.train(self.synthetic_ranking.train_path)

    evaluation = model.evaluate(
        self.synthetic_ranking.test_pd, ndcg_truncation=5
    )
    self.assertAlmostEqual(evaluation.ndcg, expected_ndcg, delta=delta)

  def test_adult_sparse_oblique(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        num_trees=5,
        split_axis="SPARSE_OBLIQUE",
    )
    model = learner.train(self.adult.train)
    assert isinstance(model, decision_forest_model.DecisionForestModel)
    model.plot_tree().html()
    logging.info("Trained model: %s", model)

  def test_adult_num_threads(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="income", num_threads=12, num_trees=50
    )

    self._check_adult_model(learner=learner, minimum_accuracy=0.869)

  def test_model_type_ranking(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="col_float",
        ranking_group="col_three_string",
        num_trees=1,
        task=generic_learner.Task.RANKING,
    )
    self.assertEqual(
        learner.train(test_utils.toy_dataset()).task(),
        generic_learner.Task.RANKING,
    )

  def test_monotonic_non_compatible_options(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="label", features=[Column("feature", monotonic=+1)]
    )
    ds = pd.DataFrame({"feature": [0, 1], "label": [0, 1]})
    with self.assertRaisesRegex(
        test_utils.AbslInvalidArgumentError,
        "Gradient Boosted Trees does not support monotonic constraints with"
        " use_hessian_gain=false",
    ):
      _ = learner.train(ds)

  def test_monotonic_training(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        num_trees=70,
        use_hessian_gain=True,
        features=[
            Column("age", monotonic=+1),
            Column("hours_per_week", monotonic=-1),
            Column("education_num", monotonic=+1),
        ],
        include_all_columns=True,
    )

    test_utils.assertProto2Equal(
        self,
        learner._get_training_config(),
        abstract_learner_pb2.TrainingConfig(
            learner="GRADIENT_BOOSTED_TREES",
            label="income",
            task=abstract_model_pb2.Task.CLASSIFICATION,
            metadata=abstract_model_pb2.Metadata(framework="Python YDF"),
            monotonic_constraints=[
                ProtoMonotonicConstraint(
                    feature="^age$",
                    direction=ProtoMonotonicConstraint.INCREASING,
                ),
                ProtoMonotonicConstraint(
                    feature="^hours_per_week$",
                    direction=ProtoMonotonicConstraint.DECREASING,
                ),
                ProtoMonotonicConstraint(
                    feature="^education_num$",
                    direction=ProtoMonotonicConstraint.INCREASING,
                ),
            ],
        ),
    )

    model, _, _ = self._check_adult_model(
        learner, minimum_accuracy=0.8565, use_pandas=True
    )

    _ = model.analyze(self.adult.test_pd)

  def test_with_validation_pd(self):
    evaluation = (
        specialized_learners.GradientBoostedTreesLearner(
            label="income", num_trees=50
        )
        .train(self.adult.train, valid=self.adult.test)
        .evaluate(self.adult.test)
    )

    logging.info("evaluation:\n%s", evaluation)
    self.assertAlmostEqual(evaluation.accuracy, 0.87, 1)

  def test_with_validation_missing_columns_fails(self):
    with self.assertRaisesRegex(ValueError, "Missing required column 'age'"):
      invalid_adult_test_pd = self.adult.test_pd.drop(["age"], axis=1)
      specialized_learners.GradientBoostedTreesLearner(
          label="income", num_trees=50
      ).train(self.adult.train_pd, valid=invalid_adult_test_pd)

  def test_with_validation_path(self):
    evaluation = (
        specialized_learners.GradientBoostedTreesLearner(
            label="income", num_trees=50
        )
        .train(self.adult.train_path, valid=self.adult.test_path)
        .evaluate(self.adult.test_path)
    )

    logging.info("evaluation:\n%s", evaluation)
    self.assertAlmostEqual(evaluation.accuracy, 0.87, 1)

  def test_failure_train_path_validation_pd(self):
    with self.assertRaisesRegex(
        ValueError,
        "If the training dataset is a path, the validation dataset must also be"
        " a path.",
    ):
      specialized_learners.GradientBoostedTreesLearner(
          label="income", num_trees=50
      ).train(self.adult.train_path, valid=self.adult.test)

  def test_failure_train_pd_validation_path(self):
    with self.assertRaisesRegex(
        ValueError,
        "The validation dataset may only be a path if the training dataset is"
        " a path.",
    ):
      specialized_learners.GradientBoostedTreesLearner(
          label="income", num_trees=50
      ).train(self.adult.train, valid=self.adult.test_path)

  def test_resume_training(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        num_trees=10,
        resume_training=True,
        working_dir=self.create_tempdir().full_path,
    )
    model_1 = learner.train(self.adult.train)
    assert isinstance(model_1, decision_forest_model.DecisionForestModel)
    self.assertEqual(model_1.num_trees(), 10)
    learner.hyperparameters["num_trees"] = 50
    model_2 = learner.train(self.adult.train)
    assert isinstance(model_2, decision_forest_model.DecisionForestModel)
    self.assertEqual(model_2.num_trees(), 50)

  def test_better_default_template(self):
    ds = test_utils.toy_dataset()
    label = "binary_int_label"
    templates = (
        specialized_learners.GradientBoostedTreesLearner.hyperparameter_templates()
    )
    self.assertIn("better_defaultv1", templates)
    better_defaultv1 = templates["better_defaultv1"]
    learner = specialized_learners.GradientBoostedTreesLearner(
        label=label, **better_defaultv1
    )
    self.assertEqual(
        learner.hyperparameters["growing_strategy"], "BEST_FIRST_GLOBAL"
    )
    _ = learner.train(ds)

  def test_model_with_na_conditions_numerical(self):
    ds = pd.DataFrame({
        "feature": [np.nan] * 10 + [1.234] * 10,
        "label": [0] * 10 + [1] * 10,
    })
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="label",
        allow_na_conditions=True,
        num_trees=1,
    )
    model = learner.train(ds)
    evaluation = model.evaluate(ds)
    self.assertEqual(evaluation.accuracy, 1)

  def test_discretized_numerical(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        num_trees=100,
        shrinkage=0.1,
        max_depth=4,
        discretize_numerical_columns=True,
    )
    model, evaluation, _ = self._check_adult_model(
        learner=learner, use_pandas=True, minimum_accuracy=0.8552
    )
    age_spec = model.data_spec().columns[1]
    self.assertEqual(age_spec.name, "age")
    self.assertEqual(age_spec.type, ds_pb.ColumnType.DISCRETIZED_NUMERICAL)

    self.assertLess(evaluation.accuracy, 0.8746)
    self.assertGreater(evaluation.loss, 0.28042)
    self.assertLess(evaluation.loss, 0.30802)

  @parameterized.parameters(
      (np.array([0, 0, 0, 1, 1]),),
      (np.array([0, 0, 1, 1, 1]),),
      (np.array([1, 1, 0, 0, 0]),),
      (np.array([1, 1, 1, 0, 0]),),
      (np.array([2, 2, 2, 1, 1]),),
      (np.array([2, 2, 1, 1, 1]),),
      (np.array([1, 1, 2, 2, 2]),),
      (np.array([1, 1, 1, 2, 2]),),
      (np.array([20, 20, 20, -10, -10]),),
      (np.array([20, 20, 20, -10, -10]),),
      (np.array([-10, -10, 20, 20, 20]),),
      (np.array([-10, -10, 20, 20, 20]),),
  )
  def test_label_classes_order_int(self, label_data):
    data = {"f": np.arange(5), "label": label_data}
    model_1 = specialized_learners.GradientBoostedTreesLearner(
        label="label",
        min_examples=1,
        num_trees=1,
        validation_ratio=0.0,
    ).train(data)
    npt.assert_equal(model_1.label_classes(), np.unique(label_data).astype(str))

  @parameterized.parameters(
      (np.array(["f", "f", "f", "x", "x"]),),
      (np.array(["f", "f", "x", "x", "x"]),),
      (np.array(["x", "x", "f", "f", "f"]),),
      (np.array(["x", "x", "x", "f", "f"]),),
  )
  def test_label_classes_order_str(self, label_data):
    data = {"f": np.arange(5), "label": label_data}
    model_1 = specialized_learners.GradientBoostedTreesLearner(
        label="label",
        min_examples=1,
        num_trees=1,
        validation_ratio=0.0,
    ).train(data)
    npt.assert_equal(model_1.label_classes(), np.unique(label_data).astype(str))

  def test_adult_poisson(self):
    model = specialized_learners.GradientBoostedTreesLearner(
        label="hours_per_week",
        growing_strategy="BEST_FIRST_GLOBAL",
        task=generic_learner.Task.REGRESSION,
        loss="POISSON",
        split_axis="SPARSE_OBLIQUE",
        validation_ratio=0.2,
        num_trees=10,
    ).train(self.adult.train_pd)
    _ = model.analyze(self.adult.test_pd, sampling=0.1)

  def test_propagate_error(self):
    extra_training_config = text_format.Parse(
        """
[yggdrasil_decision_forests.model.gradient_boosted_trees.proto.gradient_boosted_trees_config] {
  decision_tree {
    internal {
        generate_fake_error_in_splitter: true
    }
  }
}
""",
        abstract_learner_pb2.TrainingConfig(),
    )
    logging.info("extra_training_config: %s", extra_training_config)

    learner = specialized_learners.GradientBoostedTreesLearner(
        label="l", extra_training_config=extra_training_config
    )
    with self.assertRaisesRegex(RuntimeError, "Fake error"):
      learner.train(
          {"l": np.array([0, 1, 0, 1] * 100), "f": np.array([1, 2, 3, 4] * 100)}
      )

  def test_cross_validation_ranking(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="LABEL",
        ranking_group="GROUP",
        task=generic_learner.Task.RANKING,
        num_trees=10,
    )
    evaluation = learner.cross_validation(
        self.synthetic_ranking.train, folds=10, parallel_evaluations=2
    )
    logging.info("evaluation:\n%s", evaluation)
    self.assertGreaterEqual(evaluation.ndcg, 0.729)
    self.assertLessEqual(evaluation.ndcg, 0.761)
    # All the examples are used in the evaluation
    self.assertEqual(
        evaluation.num_examples,
        self.synthetic_ranking.train.data_spec().created_num_rows,
    )

    _ = evaluation._repr_html_()

  def test_tuner_manual(self):
    tuner = tuner_lib.RandomSearchTuner(
        num_trials=5,
        automatic_search_space=True,
        parallel_trials=2,
    )
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        tuner=tuner,
        num_trees=30,
    )

    model, _, _ = self._check_adult_model(learner, minimum_accuracy=0.864)
    logs = model.hyperparameter_optimizer_logs()
    self.assertIsNotNone(logs)
    self.assertLen(logs.trials, 5)

  def test_tuner_predefined(self):
    tuner = tuner_lib.RandomSearchTuner(
        num_trials=5,
        automatic_search_space=True,
        parallel_trials=2,
    )
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        tuner=tuner,
        num_trees=30,
    )

    model, _, _ = self._check_adult_model(learner, minimum_accuracy=0.864)
    logs = model.hyperparameter_optimizer_logs()
    self.assertIsNotNone(logs)
    self.assertLen(logs.trials, 5)

  def test_label_type_error_message(self):
    with self.assertRaisesRegex(
        ValueError,
        "Cannot import column 'l' with semantic=Semantic.CATEGORICAL",
    ):
      _ = specialized_learners.GradientBoostedTreesLearner(
          label="l", task=generic_learner.Task.CLASSIFICATION
      ).train(pd.DataFrame({"l": [1.0, 2.0], "f": [0, 1]}))

    with self.assertRaisesRegex(
        ValueError,
        "Cannot convert NUMERICAL column 'l' of type numpy's array of 'object'"
        " and with content=",
    ):
      _ = specialized_learners.GradientBoostedTreesLearner(
          label="l", task=generic_learner.Task.REGRESSION
      ).train(pd.DataFrame({"l": ["A", "B"], "f": [0, 1]}))


if __name__ == "__main__":
  absltest.main()
