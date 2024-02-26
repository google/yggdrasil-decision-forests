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

"""Tests for model learning."""

import dataclasses
import os
import signal
from typing import Optional, Sequence, Tuple

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import numpy.typing as npty
import pandas as pd

from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.dataset import dataset
from ydf.dataset import dataspec
from ydf.learner import custom_loss
from ydf.learner import generic_learner
from ydf.learner import specialized_learners
from ydf.learner import tuner as tuner_lib
from ydf.metric import metric
from ydf.model import generic_model
from ydf.model import model_lib
from ydf.model.decision_forest_model import decision_forest_model
from ydf.model.gradient_boosted_trees_model import gradient_boosted_trees_model
from ydf.utils import log
from ydf.utils import test_utils

ProtoMonotonicConstraint = abstract_learner_pb2.MonotonicConstraint
Column = dataspec.Column


@dataclasses.dataclass(frozen=True)
class TrainAndTestDataset:
  """Training / test dataset as path, VerticalDataset and DataFrame."""

  train_path: str
  test_path: str
  train_pd: pd.DataFrame
  test_pd: pd.DataFrame
  train: dataset.VerticalDataset
  test: dataset.VerticalDataset


def toy_dataset():
  df = pd.DataFrame({
      "col_three_string": ["A", "A", "B", "B", "C"],
      "col_float": [1, 2.1, 1.3, 5.5, 2.4],
      "col_two_string": ["bar", "foo", "foo", "foo", "foo"],
      "weights": [3, 2, 3.1, 28, 3],
      "binary_int_label": [0, 0, 0, 1, 1],
  })
  return df


def toy_dataset_uplift():
  df = pd.DataFrame({
      "f1": [1, 2, 3, 4] * 10,
      "treatement": ["A", "A", "B", "B"] * 10,
      "effect_binary": [0, 1, 1, 1] * 10,
      "effect_numerical": [0.1, 0.5, 0.6, 0.7] * 10,
  })
  return df


class LearnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dataset_directory = os.path.join(
        test_utils.ydf_test_data_path(), "dataset"
    )

    def load_datasets(
        name: str, column_args: Optional[Sequence[Column]] = None
    ) -> TrainAndTestDataset:
      train_path = os.path.join(self.dataset_directory, f"{name}_train.csv")
      test_path = os.path.join(self.dataset_directory, f"{name}_test.csv")
      train_pd = pd.read_csv(train_path)
      test_pd = pd.read_csv(test_path)
      train_vds = dataset.create_vertical_dataset(
          train_pd, columns=column_args, include_all_columns=True
      )
      test_vds = dataset.create_vertical_dataset(
          test_pd, data_spec=train_vds.data_spec()
      )
      return TrainAndTestDataset(
          train_path, test_path, train_pd, test_pd, train_vds, test_vds
      )

    self.adult = load_datasets("adult")
    self.two_center_regression = load_datasets("two_center_regression")
    self.synthetic_ranking = load_datasets(
        "synthetic_ranking",
        [Column("GROUP", semantic=dataspec.Semantic.HASH)],
    )
    self.sim_pte = load_datasets(
        "sim_pte",
        [
            Column("y", semantic=dataspec.Semantic.CATEGORICAL),
            Column("treat", semantic=dataspec.Semantic.CATEGORICAL),
        ],
    )

  def _check_adult_model(
      self,
      learner: generic_learner.GenericLearner,
      minimum_accuracy: float,
      check_serialization: bool = True,
  ) -> Tuple[generic_model.GenericModel, metric.Evaluation, np.ndarray]:
    """Runs a battery of test on a model compatible with the adult dataset.

    The following tests are run:
      - Train the model.
      - Run and evaluate the model.
      - Serialize the model to a YDF model.
      - Load the serialized model.
      - Make sure predictions of original model and serialized model match.

    Args:
      learner: A learner for on the adult dataset.
      minimum_accuracy: minimum accuracy.
      check_serialization: If true, check the serialization of the model.

    Returns:
      The model, its evaluation and the predictions on the test dataset.
    """
    # Train the model.
    model = learner.train(self.adult.train)

    # Evaluate the trained model.
    evaluation = model.evaluate(self.adult.test)
    self.assertGreaterEqual(evaluation.accuracy, minimum_accuracy)

    predictions = model.predict(self.adult.test)

    if check_serialization:
      ydf_model_path = os.path.join(
          self.create_tempdir().full_path, "ydf_model"
      )
      model.save(ydf_model_path)
      loaded_model = model_lib.load_model(ydf_model_path)
      npt.assert_equal(predictions, loaded_model.predict(self.adult.test))

    return model, evaluation, predictions


class RandomForestLearnerTest(LearnerTest):

  def test_adult_classification(self):
    learner = specialized_learners.RandomForestLearner(label="income")
    model = learner.train(self.adult.train)
    logging.info("Trained model: %s", model)

    # Evaluate the trained model.
    evaluation = model.evaluate(self.adult.test)
    logging.info("Evaluation: %s", evaluation)
    self.assertGreaterEqual(evaluation.accuracy, 0.864)

  def test_two_center_regression(self):
    learner = specialized_learners.RandomForestLearner(
        label="target", task=generic_learner.Task.REGRESSION
    )
    model = learner.train(self.two_center_regression.train)
    logging.info("Trained model: %s", model)

    # Evaluate the trained model.
    evaluation = model.evaluate(self.two_center_regression.test)
    logging.info("Evaluation: %s", evaluation)
    self.assertAlmostEqual(evaluation.rmse, 114.54, places=0)

  def test_sim_pte_uplift(self):
    learner = specialized_learners.RandomForestLearner(
        label="y",
        uplift_treatment="treat",
        task=generic_learner.Task.CATEGORICAL_UPLIFT,
    )
    model = learner.train(self.sim_pte.train)

    evaluation = model.evaluate(self.sim_pte.test)
    self.assertAlmostEqual(evaluation.qini, 0.105709, places=2)

  def test_sim_pte_uplift_pd(self):
    learner = specialized_learners.RandomForestLearner(
        label="y",
        uplift_treatment="treat",
        task=generic_learner.Task.CATEGORICAL_UPLIFT,
    )
    model = learner.train(self.sim_pte.train_pd)

    evaluation = model.evaluate(self.sim_pte.test_pd)
    self.assertAlmostEqual(evaluation.qini, 0.105709, places=2)

  def test_sim_pte_uplift_path(self):
    learner = specialized_learners.RandomForestLearner(
        label="y",
        uplift_treatment="treat",
        task=generic_learner.Task.CATEGORICAL_UPLIFT,
        num_threads=1,
        max_depth=2,
    )
    model = learner.train(self.sim_pte.train_path)
    evaluation = model.evaluate(self.sim_pte.test_path)
    self.assertAlmostEqual(evaluation.qini, 0.105709, places=2)

  def test_adult_classification_pd_and_vds_match(self):
    learner = specialized_learners.RandomForestLearner(
        label="income", num_trees=50
    )
    model_pd = learner.train(self.adult.train_pd)
    model_vds = learner.train(self.adult.train)

    predictions_pd_from_vds = model_vds.predict(self.adult.test_pd)
    predictions_pd_from_pd = model_pd.predict(self.adult.test_pd)
    predictions_vds_from_vds = model_vds.predict(self.adult.test)

    npt.assert_equal(predictions_pd_from_vds, predictions_vds_from_vds)
    npt.assert_equal(predictions_pd_from_pd, predictions_vds_from_vds)

  def test_two_center_regression_pd_and_vds_match(self):
    learner = specialized_learners.RandomForestLearner(
        label="target", task=generic_learner.Task.REGRESSION, num_trees=50
    )
    model_pd = learner.train(self.two_center_regression.train_pd)
    model_vds = learner.train(self.two_center_regression.train)

    predictions_pd_from_vds = model_vds.predict(
        self.two_center_regression.test_pd
    )
    predictions_pd_from_pd = model_pd.predict(
        self.two_center_regression.test_pd
    )
    predictions_vds_from_vds = model_vds.predict(
        self.two_center_regression.test
    )

    npt.assert_equal(predictions_pd_from_vds, predictions_vds_from_vds)
    npt.assert_equal(predictions_pd_from_pd, predictions_vds_from_vds)

  # TODO: Fix this test in OSS.
  @absltest.skip("predictions do not match")
  def test_adult_golden_predictions(self):
    data_spec_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "model",
        "adult_binary_class_rf",
        "data_spec.pb",
    )
    predictions_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "prediction",
        "adult_test_binary_class_rf.csv",
    )
    predictions_df = pd.read_csv(predictions_path)
    data_spec = data_spec_pb2.DataSpecification()
    with open(data_spec_path, "rb") as f:
      data_spec.ParseFromString(f.read())

    learner = specialized_learners.RandomForestLearner(
        label="income",
        num_trees=100,
        winner_take_all=False,
        data_spec=data_spec,
    )
    model = learner.train(self.adult.train_pd)
    predictions = model.predict(self.adult.test_pd)
    expected_predictions = predictions_df[">50K"].to_numpy()
    # This is not particularly exact, but enough for a confidence check.
    np.testing.assert_almost_equal(predictions, expected_predictions, decimal=1)

  def test_model_type_regression(self):
    learner = specialized_learners.RandomForestLearner(
        label="col_float",
        num_trees=1,
        task=generic_learner.Task.REGRESSION,
    )
    self.assertEqual(
        learner.train(toy_dataset()).task(), generic_learner.Task.REGRESSION
    )

  def test_model_type_classification_string_label(self):
    learner = specialized_learners.RandomForestLearner(
        label="col_three_string",
        num_trees=1,
        task=generic_learner.Task.CLASSIFICATION,
    )
    self.assertEqual(
        learner.train(toy_dataset()).task(), generic_learner.Task.CLASSIFICATION
    )

  def test_model_type_classification_int_label(self):
    learner = specialized_learners.RandomForestLearner(
        label="binary_int_label",
        num_trees=1,
        task=generic_learner.Task.CLASSIFICATION,
    )
    self.assertEqual(
        learner.train(toy_dataset()).task(), generic_learner.Task.CLASSIFICATION
    )

  def test_regression_on_categorical_fails(self):
    learner = specialized_learners.RandomForestLearner(
        label="col_three_string",
        num_trees=1,
        task=generic_learner.Task.REGRESSION,
    )
    with self.assertRaises(ValueError):
      _ = learner.train(toy_dataset())

  def test_classification_on_floats_fails(self):
    learner = specialized_learners.RandomForestLearner(
        label="col_float",
        num_trees=1,
        task=generic_learner.Task.CLASSIFICATION,
    )

    with self.assertRaises(ValueError):
      _ = (
          learner.train(toy_dataset()).task(),
          generic_learner.Task.CLASSIFICATION,
      )

  def test_model_type_categorical_uplift(self):
    learner = specialized_learners.RandomForestLearner(
        label="effect_binary",
        uplift_treatment="treatement",
        num_trees=1,
        task=generic_learner.Task.CATEGORICAL_UPLIFT,
    )
    self.assertEqual(
        learner.train(toy_dataset_uplift()).task(),
        generic_learner.Task.CATEGORICAL_UPLIFT,
    )

  def test_model_type_numerical_uplift(self):
    learner = specialized_learners.RandomForestLearner(
        label="effect_numerical",
        uplift_treatment="treatement",
        num_trees=1,
        task=generic_learner.Task.NUMERICAL_UPLIFT,
    )
    self.assertEqual(
        learner.train(toy_dataset_uplift()).task(),
        generic_learner.Task.NUMERICAL_UPLIFT,
    )

  def test_adult_num_threads(self):
    learner = specialized_learners.RandomForestLearner(
        label="income", num_threads=12, num_trees=50
    )

    self._check_adult_model(learner=learner, minimum_accuracy=0.860)

  # TODO: b/310580458 - Fix this test in OSS.
  @absltest.skip("Test sometimes times out")
  def test_interrupt_training(self):
    learner = specialized_learners.RandomForestLearner(
        label="income",
        num_trees=1000000,  # Trains for a very long time
    )

    signal.alarm(3)  # Stop the training in 3 seconds
    with self.assertRaises(ValueError):
      _ = learner.train(self.adult.train)

  def test_cross_validation(self):
    learner = specialized_learners.RandomForestLearner(
        label="income", num_trees=10
    )
    evaluation = learner.cross_validation(
        self.adult.train, folds=10, parallel_evaluations=2
    )
    logging.info("evaluation:\n%s", evaluation)
    self.assertAlmostEqual(evaluation.accuracy, 0.87, 1)
    # All the examples are used in the evaluation
    self.assertEqual(
        evaluation.num_examples, self.adult.train.data_spec().created_num_rows
    )

    with open(self.create_tempfile(), "w") as f:
      f.write(evaluation._repr_html_())

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

  def test_with_validation(self):
    with self.assertRaisesRegex(
        ValueError,
        "The learner 'RandomForestLearner' does not use a validation dataset",
    ):
      _ = specialized_learners.RandomForestLearner(
          label="income", num_trees=5
      ).train(self.adult.train, valid=self.adult.test)

  def test_train_with_path_validation_dataset(self):
    with self.assertRaisesRegex(
        ValueError,
        "The learner 'RandomForestLearner' does not use a validation dataset",
    ):
      _ = specialized_learners.RandomForestLearner(
          label="income", num_trees=5
      ).train(self.adult.train_path, valid=self.adult.test_path)

  def test_compare_pandas_and_path(self):
    learner = specialized_learners.RandomForestLearner(
        label="income", num_trees=50
    )
    model_from_pd = learner.train(self.adult.train)
    predictions_from_pd = model_from_pd.predict(self.adult.test)

    learner_from_path = specialized_learners.RandomForestLearner(
        label="income", data_spec=model_from_pd.data_spec(), num_trees=50
    )
    model_from_path = learner_from_path.train(self.adult.train_path)
    predictions_from_path = model_from_path.predict(self.adult.test)

    npt.assert_equal(predictions_from_pd, predictions_from_path)

  def test_default_hp_dictionary(self):
    learner = specialized_learners.RandomForestLearner(label="l", num_trees=50)
    self.assertDictContainsSubset(
        {
            "num_trees": 50,
            "categorical_algorithm": "CART",
            "categorical_set_split_greedy_sampling": 0.1,
            "compute_oob_performances": True,
            "compute_oob_variable_importances": False,
        },
        learner.hyperparameters,
    )

  def test_multidimensional_training_dataset(self):
    data = {
        "feature": np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
        "label": np.array([0, 1]),
    }
    learner = specialized_learners.RandomForestLearner(label="label")
    model = learner.train(data)

    expected_columns = [
        data_spec_pb2.Column(
            name="feature.0_of_4",
            type=data_spec_pb2.ColumnType.NUMERICAL,
            count_nas=0,
            numerical=data_spec_pb2.NumericalSpec(
                mean=2,
                standard_deviation=2,
                min_value=0,
                max_value=4,
            ),
        ),
        data_spec_pb2.Column(
            name="feature.1_of_4",
            type=data_spec_pb2.ColumnType.NUMERICAL,
            count_nas=0,
            numerical=data_spec_pb2.NumericalSpec(
                mean=3,
                standard_deviation=2,
                min_value=1,
                max_value=5,
            ),
        ),
        data_spec_pb2.Column(
            name="feature.2_of_4",
            type=data_spec_pb2.ColumnType.NUMERICAL,
            count_nas=0,
            numerical=data_spec_pb2.NumericalSpec(
                mean=4,
                standard_deviation=2,
                min_value=2,
                max_value=6,
            ),
        ),
        data_spec_pb2.Column(
            name="feature.3_of_4",
            type=data_spec_pb2.ColumnType.NUMERICAL,
            count_nas=0,
            numerical=data_spec_pb2.NumericalSpec(
                mean=5,
                standard_deviation=2,
                min_value=3,
                max_value=7,
            ),
        ),
    ]
    # Skip the first column that contains the label.
    self.assertEqual(model.data_spec().columns[1:], expected_columns)

    predictions = model.predict(data)
    self.assertEqual(predictions.shape, (2,))

  def test_learn_and_predict_when_label_is_not_last_column(self):
    label = "age"
    learner = specialized_learners.RandomForestLearner(
        label=label, num_trees=10
    )
    # The age column is automatically interpreted as categorical
    model_from_pd = learner.train(self.adult.train_pd)
    evaluation = model_from_pd.evaluate(self.adult.test_pd)
    self.assertGreaterEqual(evaluation.accuracy, 0.05)

  def test_model_metadata_contains_framework(self):
    learner = specialized_learners.RandomForestLearner(
        label="binary_int_label", num_trees=1
    )
    model = learner.train(toy_dataset())
    self.assertEqual(model.metadata().framework, "Python YDF")

  def test_model_metadata_does_not_populate_owner(self):
    learner = specialized_learners.RandomForestLearner(
        label="binary_int_label", num_trees=1
    )
    model = learner.train(toy_dataset())
    self.assertEqual(model.metadata().owner, "")


class CARTLearnerTest(LearnerTest):

  def test_adult(self):
    learner = specialized_learners.CartLearner(label="income")

    self._check_adult_model(learner=learner, minimum_accuracy=0.853)

  def test_two_center_regression(self):
    learner = specialized_learners.CartLearner(
        label="target", task=generic_learner.Task.REGRESSION
    )
    model = learner.train(self.two_center_regression.train)
    evaluation = model.evaluate(self.two_center_regression.test)
    self.assertAlmostEqual(evaluation.rmse, 114.081, places=3)

  def test_monotonic_non_compatible_learner(self):
    learner = specialized_learners.CartLearner(
        label="label", features=[dataspec.Column("feature", monotonic=+1)]
    )
    ds = pd.DataFrame({"feature": [0, 1], "label": [0, 1]})
    with self.assertRaisesRegex(
        test_utils.AbslInvalidArgumentError,
        "The learner CART does not support monotonic constraints",
    ):
      _ = learner.train(ds)


class GradientBoostedTreesLearnerTest(LearnerTest):

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
    self.assertGreaterEqual(evaluation.ndcg, 0.70)
    self.assertLessEqual(evaluation.ndcg, 0.74)

  def test_ranking_pd(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="LABEL",
        ranking_group="GROUP",
        task=generic_learner.Task.RANKING,
    )

    model = learner.train(self.synthetic_ranking.train_pd)
    evaluation = model.evaluate(self.synthetic_ranking.test_pd)
    self.assertGreaterEqual(evaluation.ndcg, 0.70)
    self.assertLessEqual(evaluation.ndcg, 0.74)

  def test_ranking_path(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="LABEL",
        ranking_group="GROUP",
        task=generic_learner.Task.RANKING,
    )

    model = learner.train(self.synthetic_ranking.train_path)
    evaluation = model.evaluate(self.synthetic_ranking.test_path)
    self.assertGreaterEqual(evaluation.ndcg, 0.70)
    self.assertLessEqual(evaluation.ndcg, 0.74)

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
        learner.train(toy_dataset()).task(),
        generic_learner.Task.RANKING,
    )

  def test_monotonic_non_compatible_options(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="label", features=[dataspec.Column("feature", monotonic=+1)]
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
            dataspec.Column("age", monotonic=+1),
            dataspec.Column("hours_per_week", monotonic=-1),
            dataspec.Column("education_num", monotonic=+1),
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

    model, _, _ = self._check_adult_model(learner, minimum_accuracy=0.863)

    _ = model.analyze(self.adult.test)

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

  def test_predict_iris(self):
    dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "iris.csv"
    )
    ds = pd.read_csv(dataset_path)
    model = specialized_learners.RandomForestLearner(label="class").train(ds)

    predictions = model.predict(ds)

    self.assertEqual(predictions.shape, (ds.shape[0], 3))

    row_sums = np.sum(predictions, axis=1)
    # Make sure a multi-dimensional prediction always (mostly) sums to 1.
    npt.assert_array_almost_equal(
        row_sums, np.ones(predictions.shape[0]), decimal=5
    )

  def test_better_default_template(self):
    ds = toy_dataset()
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


class LoggingTest(parameterized.TestCase):

  @parameterized.parameters(0, 1, 2)
  def test_logging(self, verbose):
    save_verbose = log.verbose(verbose)
    learner = specialized_learners.RandomForestLearner(label="label")
    ds = pd.DataFrame({"feature": [0, 1], "label": [0, 1]})
    _ = learner.train(ds)
    log.verbose(save_verbose)


class UtilityTest(LearnerTest):

  def test_feature_name_to_regex(self):
    self.assertEqual(
        generic_learner._feature_name_to_regex("a(z)e"), r"^a\(z\)e$"
    )


class CustomLossTest(LearnerTest):

  @parameterized.parameters(
      (
          custom_loss.RegressionLoss,
          generic_learner.Task.REGRESSION,
          "col_float",
      ),
      (
          custom_loss.BinaryClassificationLoss,
          generic_learner.Task.CLASSIFICATION,
          "binary_int_label",
      ),
      (
          custom_loss.MultiClassificationLoss,
          generic_learner.Task.CLASSIFICATION,
          "col_three_string",
      ),
  )
  def test_loss_raises_exception(self, loss_type, task, label_col):

    def faulty_initial_prediction(*args):
      raise NotImplementedError("Faulty initial prediction")

    faulty_custom_loss = loss_type(
        initial_predictions=faulty_initial_prediction,
        gradient_and_hessian=lambda x, y: (np.ones(len(x)), np.ones(len(x))),
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.Activation.IDENTITY,
    )
    ds = toy_dataset()
    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label=label_col,
        loss=faulty_custom_loss,
        task=task,
    )
    with self.assertRaisesRegex(
        RuntimeError, ".*NotImplementedError: Faulty initial prediction"
    ):
      _ = learner_custom_loss.train(ds)

  def test_avoid_memory_corruption(self):
    ref_to_labels = None

    def faulty_initial_prediction(
        labels: npty.NDArray[np.float32], _: npty.NDArray[np.float32]
    ) -> np.float32:
      nonlocal ref_to_labels
      ref_to_labels = labels
      return np.float32(0)

    faulty_custom_loss = custom_loss.RegressionLoss(
        initial_predictions=faulty_initial_prediction,
        gradient_and_hessian=lambda x, y: (np.ones(len(x)), np.ones(len(x))),
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.Activation.IDENTITY,
    )
    ds = toy_dataset()
    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="col_float",
        loss=faulty_custom_loss,
        task=generic_learner.Task.REGRESSION,
    )
    with self.assertRaisesRegex(
        RuntimeError,
        'Cannot hold a reference to "labels" outside of a custom loss'
        " function.*",
    ):
      _ = learner_custom_loss.train(ds)

  def test_readonly_args(self):
    def faulty_initial_prediction(
        labels: npty.NDArray[np.float32], _: npty.NDArray[np.float32]
    ) -> np.float32:
      labels[0] = 5
      return np.float32(0)

    faulty_custom_loss = custom_loss.RegressionLoss(
        initial_predictions=faulty_initial_prediction,
        gradient_and_hessian=lambda x, y: (np.ones(len(x)), np.ones(len(x))),
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.Activation.IDENTITY,
    )
    ds = toy_dataset()
    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="col_float",
        loss=faulty_custom_loss,
        task=generic_learner.Task.REGRESSION,
    )
    with self.assertRaisesRegex(RuntimeError, ".*read-only.*"):
      _ = learner_custom_loss.train(ds)

  @parameterized.parameters(
      (
          custom_loss.RegressionLoss,
          generic_learner.Task.CLASSIFICATION,
          "binary_int_label",
      ),
      (
          custom_loss.BinaryClassificationLoss,
          generic_learner.Task.REGRESSION,
          "col_float",
      ),
      (
          custom_loss.MultiClassificationLoss,
          generic_learner.Task.REGRESSION,
          "col_float",
      ),
  )
  def test_invalid_tasks(self, loss_type, task_type, label):
    custom_loss_container = loss_type(
        initial_predictions=lambda x, y: np.float32(0),
        gradient_and_hessian=lambda x, y: np.ones([6, len(x)]),
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.Activation.IDENTITY,
    )
    ds = toy_dataset()
    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label=label,
        loss=custom_loss_container,
        task=task_type,
        num_trees=1,
    )
    with self.assertRaisesRegex(
        ValueError, "A .* is only compatible with .* tasks.*"
    ):
      _ = learner_custom_loss.train(ds)

  def test_mse_custom_equal_to_builtin(self):

    def mse_initial_predictions(
        labels: npty.NDArray[np.float32], weights: npty.NDArray[np.float32]
    ) -> np.float32:
      return np.average(labels, weights=weights)

    def mse_gradient(
        labels: npty.NDArray[np.float32], predictions: npty.NDArray[np.float32]
    ) -> Tuple[npty.NDArray[np.float32], npty.NDArray[np.float32]]:
      return (labels - predictions, np.ones(labels.shape))

    def mse_loss(
        labels: npty.NDArray[np.float32],
        predictions: npty.NDArray[np.float32],
        weights: npty.NDArray[np.float32],
    ) -> np.float32:
      numerator = np.sum(np.multiply(weights, np.square(labels - predictions)))
      denominator = np.sum(weights)
      return np.sqrt(numerator / denominator)

    mse_custom_loss = custom_loss.RegressionLoss(
        initial_predictions=mse_initial_predictions,
        gradient_and_hessian=mse_gradient,
        loss=mse_loss,
        activation=custom_loss.Activation.IDENTITY,
    )

    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="target",
        loss=mse_custom_loss,
        task=generic_learner.Task.REGRESSION,
        num_trees=30,
        early_stopping="NONE",
        validation_ratio=0.0,
    )
    model_custom_loss: generic_model.GenericModel = learner_custom_loss.train(
        self.two_center_regression.train
    )

    learner_builtin_loss = specialized_learners.GradientBoostedTreesLearner(
        label="target",
        task=generic_learner.Task.REGRESSION,
        num_trees=30,
        early_stopping="NONE",
        validation_ratio=0.0,
    )
    model_builtin_loss: generic_model.GenericModel = learner_builtin_loss.train(
        self.two_center_regression.train
    )
    npt.assert_allclose(
        model_custom_loss.predict(self.two_center_regression.test),
        model_builtin_loss.predict(self.two_center_regression.test),
        rtol=1e-5,  # Without activation function, the predictions can be large.
        atol=1e-6,
    )

  @parameterized.parameters(
      custom_loss.Activation.IDENTITY,
      custom_loss.Activation.SIGMOID,
  )
  def test_binomial_custom_equal_to_builtin(self, activation):

    def binomial_initial_predictions(
        labels: npty.NDArray[np.int32], weights: npty.NDArray[np.float32]
    ) -> np.float32:
      sum_weights = np.sum(weights)
      sum_weights_positive = np.sum((labels == 2) * weights)
      ratio_positive = sum_weights_positive / sum_weights
      if ratio_positive == 0.0:
        return -np.iinfo(np.float32).max
      elif ratio_positive == 1.0:
        return np.iinfo(np.float32).max
      return np.log(ratio_positive / (1 - ratio_positive))

    def binomial_gradient(
        labels: npty.NDArray[np.int32], predictions: npty.NDArray[np.float32]
    ) -> Tuple[npty.NDArray[np.float32], npty.NDArray[np.float32]]:
      pred_probability = 1.0 / (1.0 + np.exp(-predictions))
      binary_labels = labels == 2
      return (
          binary_labels - pred_probability,
          pred_probability * (1 - pred_probability),
      )

    def binomial_loss(
        labels: npty.NDArray[np.int32],
        predictions: npty.NDArray[np.float32],
        weights: npty.NDArray[np.float32],
    ) -> np.float32:
      binary_labels = labels == 2
      return (
          -2.0
          * np.sum(
              weights
              * (
                  binary_labels * predictions
                  - np.log(1.0 + np.exp(predictions))
              )
          )
          / np.sum(weights)
      )

    binomial_custom_loss = custom_loss.BinaryClassificationLoss(
        initial_predictions=binomial_initial_predictions,
        gradient_and_hessian=binomial_gradient,
        loss=binomial_loss,
        activation=activation,
    )

    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        loss=binomial_custom_loss,
        task=generic_learner.Task.CLASSIFICATION,
        early_stopping="NONE",
        validation_ratio=0.0,
        num_trees=30,
    )
    model_custom_loss: generic_model.GenericModel = learner_custom_loss.train(
        self.adult.train
    )

    learner_builtin_loss = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        task=generic_learner.Task.CLASSIFICATION,
        apply_link_function=(
            activation
            == custom_loss.Activation.SIGMOID
        ),
        early_stopping="NONE",
        validation_ratio=0.0,
        num_trees=30,
    )
    model_builtin_loss: generic_model.GenericModel = learner_builtin_loss.train(
        self.adult.train
    )
    npt.assert_allclose(
        model_custom_loss.predict(self.adult.test),
        model_builtin_loss.predict(self.adult.test),
        rtol=1e-5,  # Without activation function, the predictions can be large.
        atol=1e-6,
    )

  @parameterized.parameters(
      custom_loss.Activation.IDENTITY,
      custom_loss.Activation.SOFTMAX,
  )
  def test_multinomial_custom_equal_to_builtin(self, activation):

    def multinomial_initial_predictions(
        labels: npty.NDArray[np.int32], _: npty.NDArray[np.float32]
    ) -> npty.NDArray[np.float32]:
      dimension = np.max(labels)
      return np.zeros(dimension, dtype=np.float32)

    def multinomial_gradient(
        labels: npty.NDArray[np.int32], predictions: npty.NDArray[np.float32]
    ) -> Tuple[npty.NDArray[np.float32], npty.NDArray[np.float32]]:
      dimension = np.max(labels)
      normalization = 1.0 / np.sum(np.exp(predictions), axis=1)
      normalized_predictions = np.exp(predictions) * normalization[:, None]
      label_indicator = (
          (labels - 1)[:, np.newaxis] == np.arange(dimension)
      ).astype(int)
      gradient = label_indicator - normalized_predictions
      hessian = np.abs(gradient) * (1 - np.abs(gradient))
      return (np.transpose(gradient), np.transpose(hessian))

    def multinomial_loss(
        labels: npty.NDArray[np.int32],
        predictions: npty.NDArray[np.float32],
        weights: npty.NDArray[np.float32],
    ) -> np.float32:
      sum_weights = np.sum(weights)
      dimension = np.max(labels)
      sum_exp_pred = np.sum(np.exp(predictions), axis=1)
      indicator_matrix = (
          (labels - 1)[:, np.newaxis] == np.arange(dimension)
      ).astype(int)
      label_exp_pred = np.exp(np.sum(predictions * indicator_matrix, axis=1))
      return (
          -np.sum(weights * np.log(label_exp_pred / sum_exp_pred)) / sum_weights
      )

    # Path to dataset.
    dataset_directory = os.path.join(test_utils.ydf_test_data_path(), "dataset")
    train_path = os.path.join(dataset_directory, "dna.csv")
    all_ds = pd.read_csv(train_path)
    # Randomly split the dataset into a training (90%) and testing (10%) dataset
    all_ds = all_ds.sample(frac=1)
    split_idx = len(all_ds) * 9 // 10
    train_ds = all_ds.iloc[:split_idx]
    test_ds = all_ds.iloc[split_idx:]

    multinomial_custom_loss = custom_loss.MultiClassificationLoss(
        initial_predictions=multinomial_initial_predictions,
        gradient_and_hessian=multinomial_gradient,
        loss=multinomial_loss,
        activation=activation,
    )

    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="LABEL",
        loss=multinomial_custom_loss,
        task=generic_learner.Task.CLASSIFICATION,
        early_stopping="NONE",
        validation_ratio=0.0,
        num_trees=30,
    )
    model_custom_loss: generic_model.GenericModel = learner_custom_loss.train(
        train_ds
    )

    learner_builtin_loss = specialized_learners.GradientBoostedTreesLearner(
        label="LABEL",
        task=generic_learner.Task.CLASSIFICATION,
        apply_link_function=(
            activation == custom_loss.Activation.SOFTMAX
        ),
        early_stopping="NONE",
        validation_ratio=0.0,
        num_trees=30,
    )
    model_builtin_loss: generic_model.GenericModel = learner_builtin_loss.train(
        train_ds
    )
    npt.assert_allclose(
        model_custom_loss.predict(test_ds),
        model_builtin_loss.predict(test_ds),
        rtol=1e-5,  # Without activation function, the predictions can be large.
        atol=1e-6,
    )

  def test_multiclass_initial_prediction(self):

    def multiclass_initial_prediction(
        labels: npty.NDArray[np.int32], _: npty.NDArray[np.float32]
    ) -> npty.NDArray[np.float32]:
      dimension = np.max(labels)
      return np.arange(1, dimension + 1)

    multiclass_custom_loss = custom_loss.MultiClassificationLoss(
        initial_predictions=multiclass_initial_prediction,
        gradient_and_hessian=lambda x, y: (
            np.ones([3, len(x)]),
            np.ones([3, len(x)]),
        ),
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.Activation.IDENTITY,
    )
    ds = toy_dataset()
    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="col_three_string",
        loss=multiclass_custom_loss,
        task=generic_learner.Task.CLASSIFICATION,
        num_trees=1,
    )
    model: gradient_boosted_trees_model.GradientBoostedTreesModel = (
        learner_custom_loss.train(ds)
    )
    npt.assert_equal(model.initial_predictions(), [1, 2, 3])

  def test_multiclass_wrong_initial_prediction_dimensions(self):

    def multiclass_initial_prediction(
        labels: npty.NDArray[np.int32], _: npty.NDArray[np.float32]
    ) -> npty.NDArray[np.float32]:
      dimension = np.max(labels)
      return np.arange(1, dimension)

    multiclass_custom_loss = custom_loss.MultiClassificationLoss(
        initial_predictions=multiclass_initial_prediction,
        gradient_and_hessian=lambda x, y: (
            np.ones([3, len(x)]),
            np.ones([3, len(x)]),
        ),
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.Activation.IDENTITY,
    )
    ds = toy_dataset()
    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="col_three_string",
        loss=multiclass_custom_loss,
        task=generic_learner.Task.CLASSIFICATION,
        num_trees=1,
    )
    with self.assertRaisesRegex(
        ValueError,
        "The initial_predictions must be a one-dimensional Numpy array of 3"
        " elements.*",
    ):
      _ = learner_custom_loss.train(ds)

  @parameterized.parameters(
      (
          custom_loss.RegressionLoss,
          generic_learner.Task.REGRESSION,
          "col_float",
      ),
      (
          custom_loss.BinaryClassificationLoss,
          generic_learner.Task.CLASSIFICATION,
          "binary_int_label",
      ),
  )
  def test_wrong_gradient_dimensions(self, loss_type, task_type, label):
    faulty_loss = loss_type(
        initial_predictions=lambda x, y: np.float32(0),
        gradient_and_hessian=lambda x, y: (
            np.ones(len(x) - 1),
            np.ones(len(x)),
        ),
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.Activation.IDENTITY,
    )
    ds = toy_dataset()
    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label=label,
        loss=faulty_loss,
        task=task_type,
        num_trees=1,
    )
    with self.assertRaisesRegex(
        ValueError,
        "The gradient must be a one-dimensional Numpy array of 5 elements.*",
    ):
      _ = learner_custom_loss.train(ds)

  @parameterized.parameters(
      (
          custom_loss.RegressionLoss,
          generic_learner.Task.REGRESSION,
          "col_float",
      ),
      (
          custom_loss.BinaryClassificationLoss,
          generic_learner.Task.CLASSIFICATION,
          "binary_int_label",
      ),
  )
  def test_wrong_hessian_dimensions(self, loss_type, task_type, label):
    faulty_loss = loss_type(
        initial_predictions=lambda x, y: np.float32(0),
        gradient_and_hessian=lambda x, y: (
            np.ones(len(x)),
            np.ones(len(x) - 1),
        ),
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.Activation.IDENTITY,
    )
    ds = toy_dataset()
    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label=label,
        loss=faulty_loss,
        task=task_type,
        num_trees=1,
    )
    with self.assertRaisesRegex(
        ValueError,
        "The hessian must be a one-dimensional Numpy array of 5 elements.*",
    ):
      _ = learner_custom_loss.train(ds)

  @parameterized.parameters(
      (
          custom_loss.RegressionLoss,
          generic_learner.Task.REGRESSION,
          "col_float",
      ),
      (
          custom_loss.BinaryClassificationLoss,
          generic_learner.Task.CLASSIFICATION,
          "binary_int_label",
      ),
  )
  def test_stacked_gradient_dimensions(self, loss_type, task_type, label):
    faulty_loss = loss_type(
        initial_predictions=lambda x, y: np.float32(0),
        gradient_and_hessian=lambda x, y: 3 * np.ones([2, 5]),
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.Activation.IDENTITY,
    )
    ds = toy_dataset()
    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label=label,
        loss=faulty_loss,
        task=task_type,
        num_trees=1,
    )
    with self.assertRaisesRegex(
        ValueError,
        ".*gradient_and_hessian function returned a numpy array, expected a"
        " Sequence of two numpy arrays.*",
    ):
      _ = learner_custom_loss.train(ds)

  def test_loss_with_jax_nojit(self):
    def mse_loss(labels, predictions):
      numerator = jnp.sum(jnp.square(jnp.subtract(labels, predictions)))
      denominator = jnp.size(labels)
      res = jax.block_until_ready(jnp.divide(numerator, denominator))
      return res

    def weighted_mse_loss(labels, predictions, _):
      return mse_loss(labels, predictions)

    mse_grad = jax.grad(mse_loss, argnums=1)
    mse_hessian = jax.jacfwd(jax.jacrev(mse_loss), argnums=1)

    def mse_gradient_and_hessian(labels, predictions):
      res = (
          -mse_grad(labels, predictions).block_until_ready(),
          -jnp.diagonal(mse_hessian(labels, predictions)).block_until_ready(),
      )
      return res

    def mse_initial_predictions(labels, weights):
      res = jax.block_until_ready(jnp.average(labels, weights=weights))
      return res

    mse_custom_loss = custom_loss.RegressionLoss(
        initial_predictions=mse_initial_predictions,
        gradient_and_hessian=mse_gradient_and_hessian,
        loss=weighted_mse_loss,
        activation=custom_loss.Activation.IDENTITY,
    )

    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="target",
        loss=mse_custom_loss,
        task=generic_learner.Task.REGRESSION,
        num_trees=30,
        early_stopping="NONE",
        validation_ratio=0.0,
    )
    model_custom_loss: generic_model.GenericModel = learner_custom_loss.train(
        self.two_center_regression.train
    )

    learner_builtin_loss = specialized_learners.GradientBoostedTreesLearner(
        label="target",
        task=generic_learner.Task.REGRESSION,
        num_trees=30,
        early_stopping="NONE",
        validation_ratio=0.0,
    )
    model_builtin_loss: generic_model.GenericModel = learner_builtin_loss.train(
        self.two_center_regression.train
    )
    npt.assert_allclose(
        model_custom_loss.predict(self.two_center_regression.test),
        model_builtin_loss.predict(self.two_center_regression.test),
        rtol=1e-5,  # Without activation function, the predictions can be large.
        atol=1e-6,
    )

  def test_loss_with_jax_jit(self):
    @jax.jit
    def mse_loss(labels, predictions):
      numerator = jnp.sum(jnp.square(jnp.subtract(labels, predictions)))
      denominator = jnp.size(labels)
      res = jax.block_until_ready(jnp.divide(numerator, denominator))
      return res

    @jax.jit
    def weighted_mse_loss(labels, predictions, _):
      return mse_loss(labels, predictions)

    mse_grad = jax.jit(jax.grad(mse_loss, argnums=1))
    mse_hessian = jax.jit(jax.jacfwd(jax.jacrev(mse_loss), argnums=1))

    def mse_gradient_and_hessian(labels, predictions):
      return (
          -mse_grad(labels, predictions).block_until_ready(),
          -jnp.diagonal(mse_hessian(labels, predictions)).block_until_ready(),
      )

    @jax.jit
    def mse_initial_predictions(labels, weights):
      res = jax.block_until_ready(jnp.average(labels, weights=weights))
      return res

    mse_custom_loss = custom_loss.RegressionLoss(
        initial_predictions=mse_initial_predictions,
        gradient_and_hessian=mse_gradient_and_hessian,
        loss=weighted_mse_loss,
        activation=custom_loss.Activation.IDENTITY,
    )

    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="target",
        loss=mse_custom_loss,
        task=generic_learner.Task.REGRESSION,
        num_trees=30,
        early_stopping="NONE",
        validation_ratio=0.0,
    )
    model_custom_loss: generic_model.GenericModel = learner_custom_loss.train(
        self.two_center_regression.train
    )

    learner_builtin_loss = specialized_learners.GradientBoostedTreesLearner(
        label="target",
        task=generic_learner.Task.REGRESSION,
        num_trees=30,
        early_stopping="NONE",
        validation_ratio=0.0,
    )
    model_builtin_loss: generic_model.GenericModel = learner_builtin_loss.train(
        self.two_center_regression.train
    )
    npt.assert_allclose(
        model_custom_loss.predict(self.two_center_regression.test),
        model_builtin_loss.predict(self.two_center_regression.test),
        rtol=1e-5,  # Without activation function, the predictions can be large.
        atol=1e-6,
    )


if __name__ == "__main__":
  absltest.main()
