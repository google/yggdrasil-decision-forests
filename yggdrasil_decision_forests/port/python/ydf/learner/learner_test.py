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

import collections
import os
import signal
from typing import Tuple

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
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
from ydf.model.decision_forest_model import decision_forest_model
from ydf.model.gradient_boosted_trees_model import gradient_boosted_trees_model
from ydf.utils import log
from ydf.utils import test_utils

ProtoMonotonicConstraint = abstract_learner_pb2.MonotonicConstraint
Column = dataspec.Column

# TODO: Convert to dataclass.
DatasetForTesting = collections.namedtuple(
    "Dataset",
    [
        "train",
        "test",
        "label",
    ],
)


def toy_dataset():
  df = pd.DataFrame({
      "col1": ["A", "A", "B", "B", "C"],
      "col2": [1, 2.1, 1.3, 5.5, 2.4],
      "col3": ["bar", "foo", "foo", "foo", "foo"],
      "weights": [3, 2, 3.1, 28, 3],
      "label": [0, 0, 0, 1, 1],
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


def adult_dataset() -> DatasetForTesting:
  """Adult/census binary classification dataset."""

  # Path to dataset.
  dataset_directory = os.path.join(test_utils.ydf_test_data_path(), "dataset")
  train_path = os.path.join(dataset_directory, "adult_train.csv")
  test_path = os.path.join(dataset_directory, "adult_test.csv")

  train = pd.read_csv(train_path)
  test = pd.read_csv(test_path)
  label = "income"

  return DatasetForTesting(train, test, label)


class LearnerTest(absltest.TestCase):

  def _check_adult_model(
      self,
      learner: generic_learner.GenericLearner,
      ds: DatasetForTesting,
      minimum_accuracy: float,
  ) -> Tuple[generic_model.GenericModel, metric.Evaluation, np.ndarray]:
    """Runs a battery of test on a model compatible with the adult dataset.

    The following tests are run:
      - Train the model.
      - Run and evaluate the model.
      - Serialize the model to a YDF model.
      - Run the model is a separate binary (without dependencies to the training
        custom OPs).
      - Move the serialized model to another random location.
      - Load the serialized model.
      - Evaluate and run the loaded model.

    Args:
      learner: A learner for on the adult dataset.
      ds: A dataset compatible with the learner.
      minimum_accuracy: minimum accuracy.

    Returns:
      The model, its evaluation and the predictions on the test dataset.
    """
    # Train the model.
    model = learner.train(ds.train)
    logging.info("Trained model:")

    # Evaluate the trained model.
    evaluation = model.evaluate(ds.test)
    logging.info("Evaluation: %s", evaluation)
    self.assertGreaterEqual(evaluation.accuracy, minimum_accuracy)

    predictions = model.predict(ds.test)
    logging.info("Predictions: %s", predictions)

    return model, evaluation, predictions


class RandomForestLearnerTest(LearnerTest):

  def test_toy_with_inference(self):
    learner = specialized_learners.RandomForestLearner(
        label="label",
        num_trees=1,
        min_vocab_frequency=1,
        min_examples=1,
        bootstrap_training_dataset=False,
        features=["col1", "col2"],
        weights="weights",
        task=generic_learner.Task.REGRESSION,
    )
    model = learner.train(toy_dataset())
    predictions = model.predict(toy_dataset())
    np.testing.assert_equal(predictions, toy_dataset()["label"].to_numpy())

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

    ds = adult_dataset()

    learner = specialized_learners.RandomForestLearner(
        label=ds.label,
        num_trees=100,
        winner_take_all=False,
        data_spec=data_spec,
    )
    _, _, predictions = self._check_adult_model(learner, ds, 0.864)

    expected_predictions = predictions_df[">50K"].to_numpy()
    # This is not particularly exact, but enough for a confidence check.
    np.testing.assert_almost_equal(predictions, expected_predictions, decimal=1)

  def test_train_with_vds(self):
    pd_dataset = adult_dataset()
    vds_train = dataset.create_vertical_dataset(pd_dataset.train)
    vds_test = dataset.create_vertical_dataset(
        pd_dataset.test, data_spec=vds_train.data_spec()
    )
    vds_dataset = DatasetForTesting(vds_train, vds_test, pd_dataset.label)

    learner = specialized_learners.RandomForestLearner(label=pd_dataset.label)

    self._check_adult_model(learner, ds=vds_dataset, minimum_accuracy=0.864)

  # TODO: Add a test for ranking and uplifting.

  def test_toy_regression(self):
    learner = specialized_learners.RandomForestLearner(
        label="col2",
        num_trees=1,
        task=generic_learner.Task.REGRESSION,
    )
    self.assertEqual(
        learner.train(toy_dataset()).task(), generic_learner.Task.REGRESSION
    )

  def test_toy_regression_on_categorical(self):
    learner = specialized_learners.RandomForestLearner(
        label="col1",
        num_trees=1,
        task=generic_learner.Task.REGRESSION,
    )
    with self.assertRaises(ValueError):
      _ = learner.train(toy_dataset())

  def test_toy_classification(self):
    learner = specialized_learners.RandomForestLearner(
        label="col1",
        num_trees=1,
        task=generic_learner.Task.CLASSIFICATION,
    )
    self.assertEqual(
        learner.train(toy_dataset()).task(), generic_learner.Task.CLASSIFICATION
    )

  def test_toy_classification_on_ints(self):
    learner = specialized_learners.RandomForestLearner(
        label="label",
        num_trees=1,
        task=generic_learner.Task.CLASSIFICATION,
    )
    self.assertEqual(
        learner.train(toy_dataset()).task(), generic_learner.Task.CLASSIFICATION
    )

  def test_toy_classification_on_floats(self):
    learner = specialized_learners.RandomForestLearner(
        label="col2",
        num_trees=1,
        task=generic_learner.Task.CLASSIFICATION,
    )

    with self.assertRaises(ValueError):
      _ = (
          learner.train(toy_dataset()).task(),
          generic_learner.Task.CLASSIFICATION,
      )

  def test_toy_categorical_uplift(self):
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

  def test_toy_numerical_uplift(self):
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

  # TODO: b/310580458 - Fix this test in OSS.
  @absltest.skip("Test sometimes times out")
  def test_interrupt_training(self):
    ds = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_train.csv"
        )
    )
    learner = specialized_learners.RandomForestLearner(
        label="income",
        num_trees=1000000,  # Trains for a very long time
    )

    signal.alarm(3)  # Stop the training in 3 seconds
    with self.assertRaises(ValueError):
      _ = learner.train(ds)

  def test_cross_validation(self):
    ds = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_train.csv"
        )
    )
    learner = specialized_learners.RandomForestLearner(
        label="income", num_trees=10
    )
    evaluation = learner.cross_validation(ds, folds=10, parallel_evaluations=2)
    logging.info("evaluation:\n%s", evaluation)
    self.assertAlmostEqual(evaluation.accuracy, 0.87, 1)
    # All the examples are used in the evaluation
    self.assertEqual(evaluation.num_examples, ds.shape[0])

    with open(self.create_tempfile(), "w") as f:
      f.write(evaluation._repr_html_())

  def test_tuner_manual(self):
    pd_dataset = adult_dataset()
    vds_train = dataset.create_vertical_dataset(pd_dataset.train)
    vds_test = dataset.create_vertical_dataset(
        pd_dataset.test, data_spec=vds_train.data_spec()
    )
    vds_dataset = DatasetForTesting(vds_train, vds_test, pd_dataset.label)

    tuner = tuner_lib.RandomSearchTuner(
        num_trials=5,
        automatic_search_space=True,
        parallel_trials=2,
    )
    learner = specialized_learners.GradientBoostedTreesLearner(
        label=pd_dataset.label,
        tuner=tuner,
        num_trees=30,
    )

    model, _, _ = self._check_adult_model(
        learner, ds=vds_dataset, minimum_accuracy=0.864
    )
    logs = model.hyperparameter_optimizer_logs()
    self.assertIsNotNone(logs)
    self.assertLen(logs.trials, 5)

  def test_tuner_predefined(self):
    pd_dataset = adult_dataset()
    vds_train = dataset.create_vertical_dataset(pd_dataset.train)
    vds_test = dataset.create_vertical_dataset(
        pd_dataset.test, data_spec=vds_train.data_spec()
    )
    vds_dataset = DatasetForTesting(vds_train, vds_test, pd_dataset.label)

    tuner = tuner_lib.RandomSearchTuner(
        num_trials=5,
        automatic_search_space=True,
        parallel_trials=2,
    )
    learner = specialized_learners.GradientBoostedTreesLearner(
        label=pd_dataset.label,
        tuner=tuner,
        num_trees=30,
    )

    model, _, _ = self._check_adult_model(
        learner, ds=vds_dataset, minimum_accuracy=0.864
    )
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
    train_ds = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_train.csv"
        )
    )
    test_ds = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
        )
    )

    with self.assertRaisesRegex(
        ValueError,
        "The learner 'RandomForestLearner' does not use a validation dataset",
    ):
      _ = specialized_learners.RandomForestLearner(
          label="income", num_trees=50
      ).train(train_ds, valid=test_ds)

  def test_compare_pandas_and_path(self):
    dataset_directory = os.path.join(test_utils.ydf_test_data_path(), "dataset")
    train_path = os.path.join(dataset_directory, "adult_train.csv")
    test_path = os.path.join(dataset_directory, "adult_test.csv")
    label = "income"

    pd_train = pd.read_csv(train_path)
    pd_test = pd.read_csv(test_path)

    learner = specialized_learners.RandomForestLearner(label=label)
    model_from_pd = learner.train(pd_train)
    accuracy_from_pd = model_from_pd.evaluate(pd_test).accuracy

    learner_from_path = specialized_learners.RandomForestLearner(
        label=label, data_spec=model_from_pd.data_spec()
    )
    model_from_path = learner_from_path.train(train_path)
    accuracy_from_path = model_from_path.evaluate(pd_test).accuracy

    self.assertAlmostEqual(accuracy_from_path, accuracy_from_pd)

  def test_train_with_path_validation_dataset(self):
    dataset_directory = os.path.join(test_utils.ydf_test_data_path(), "dataset")
    train_path = os.path.join(dataset_directory, "adult_train.csv")
    test_path = os.path.join(dataset_directory, "adult_test.csv")
    label = "income"
    learner = specialized_learners.RandomForestLearner(label=label)
    model = learner.train(train_path, valid=test_path)
    evaluation = model.evaluate(test_path)
    self.assertGreaterEqual(evaluation.accuracy, 0.86)

  def test_hp_dictionary(self):
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
    dataset_directory = os.path.join(test_utils.ydf_test_data_path(), "dataset")
    train_path = os.path.join(dataset_directory, "adult_train.csv")
    test_path = os.path.join(dataset_directory, "adult_test.csv")
    label = "age"

    pd_train = pd.read_csv(train_path)
    pd_test = pd.read_csv(test_path)

    learner = specialized_learners.RandomForestLearner(
        label=label, num_trees=10
    )
    model_from_pd = learner.train(pd_train)

    logging.info(model_from_pd.predict(pd_test))

  def test_model_metadata_contains_framework(self):
    learner = specialized_learners.RandomForestLearner(
        label="label", num_trees=2
    )
    model = learner.train(toy_dataset())
    self.assertEqual(model.metadata().framework, "Python YDF")

  def test_model_metadata_does_not_populate_owner(self):
    learner = specialized_learners.RandomForestLearner(
        label="label", num_trees=2
    )
    model = learner.train(toy_dataset())
    self.assertEqual(model.metadata().owner, "")


class CARTLearnerTest(LearnerTest):

  def test_default_settings(self):
    ds = adult_dataset()
    learner = specialized_learners.CartLearner(label=ds.label)

    self._check_adult_model(learner=learner, ds=ds, minimum_accuracy=0.853)

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

  def test_default_settings(self):
    ds = adult_dataset()
    learner = specialized_learners.GradientBoostedTreesLearner(label=ds.label)

    self._check_adult_model(learner=learner, ds=ds, minimum_accuracy=0.869)

  def test_with_num_threads(self):
    ds = adult_dataset()
    learner = specialized_learners.GradientBoostedTreesLearner(
        label=ds.label, num_threads=12
    )

    self._check_adult_model(learner=learner, ds=ds, minimum_accuracy=0.869)

  def test_toy_ranking(self):
    learner = specialized_learners.GradientBoostedTreesLearner(
        label="col2",
        ranking_group="col1",
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
    vds_dataset = adult_dataset()

    learner = specialized_learners.GradientBoostedTreesLearner(
        label=vds_dataset.label,
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

    model, _, _ = self._check_adult_model(
        learner, ds=vds_dataset, minimum_accuracy=0.863
    )

    _ = model.analyze(vds_dataset.test)

  def test_with_validation(self):
    train_ds = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_train.csv"
        )
    )
    test_ds = pd.read_csv(
        os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_test.csv"
        )
    )
    evaluation = (
        specialized_learners.GradientBoostedTreesLearner(
            label="income", num_trees=50
        )
        .train(train_ds, valid=test_ds)
        .evaluate(test_ds)
    )

    logging.info("evaluation:\n%s", evaluation)
    self.assertAlmostEqual(evaluation.accuracy, 0.87, 1)

  def test_resume_training(self):
    ds = adult_dataset()
    learner = specialized_learners.GradientBoostedTreesLearner(
        label=ds.label,
        num_trees=10,
        resume_training=True,
        working_dir=self.create_tempdir().full_path,
    )
    model_1 = learner.train(ds.train)
    assert isinstance(model_1, decision_forest_model.DecisionForestModel)
    self.assertEqual(model_1.num_trees(), 10)
    learner.hyperparameters["num_trees"] = 50
    model_2 = learner.train(ds.train)
    assert isinstance(model_2, decision_forest_model.DecisionForestModel)
    self.assertEqual(model_2.num_trees(), 50)

  def test_ranking(self):
    dataset_directory = os.path.join(test_utils.ydf_test_data_path(), "dataset")
    train_path = os.path.join(dataset_directory, "synthetic_ranking_train.csv")
    test_path = os.path.join(dataset_directory, "synthetic_ranking_test.csv")
    train_ds = pd.read_csv(train_path)
    test_ds = pd.read_csv(test_path)
    label = "LABEL"
    ranking_group = "GROUP"

    learner = specialized_learners.GradientBoostedTreesLearner(
        label=label,
        ranking_group=ranking_group,
        task=generic_learner.Task.RANKING,
    )

    model = learner.train(train_ds)
    evaluation = model.evaluate(test_ds)
    self.assertAlmostEqual(evaluation.ndcg, 0.71, places=1)

  def test_ranking_path(self):
    dataset_directory = os.path.join(test_utils.ydf_test_data_path(), "dataset")
    train_path = os.path.join(dataset_directory, "synthetic_ranking_train.csv")
    test_path = os.path.join(dataset_directory, "synthetic_ranking_test.csv")
    label = "LABEL"
    ranking_group = "GROUP"

    learner = specialized_learners.GradientBoostedTreesLearner(
        label=label,
        ranking_group=ranking_group,
        task=generic_learner.Task.RANKING,
    )

    model = learner.train(train_path)
    evaluation = model.evaluate(test_path)
    self.assertAlmostEqual(evaluation.ndcg, 0.71, places=1)

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
    label = "label"
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


class CustomLossTest(parameterized.TestCase):

  @parameterized.parameters(
      (custom_loss.RegressionLoss, generic_learner.Task.REGRESSION, "col2"),
      (
          custom_loss.BinaryClassificationLoss,
          generic_learner.Task.CLASSIFICATION,
          "label",
      ),
      (
          custom_loss.MultiClassificationLoss,
          generic_learner.Task.CLASSIFICATION,
          "col1",
      ),
  )
  def test_loss_raises_exception(self, loss_type, task, label_col):
    def faulty_inital_prediction(*args):
      raise NotImplementedError("Faulty initial prediction")

    faulty_custom_loss = loss_type(
        initial_predictions=faulty_inital_prediction,
        gradient_and_hessian=lambda x, y: (np.ones(len(x)), np.ones(len(x))),
        loss=lambda x, y, z: np.float32(0),
        activation=loss_type.Activation.IDENTITY,
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

    def faulty_inital_prediction(
        labels: npty.NDArray[np.float32], _: npty.NDArray[np.float32]
    ) -> np.float32:
      nonlocal ref_to_labels
      ref_to_labels = labels
      return np.float32(0)

    faulty_custom_loss = custom_loss.RegressionLoss(
        initial_predictions=faulty_inital_prediction,
        gradient_and_hessian=lambda x, y: (np.ones(len(x)), np.ones(len(x))),
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.RegressionLoss.Activation.IDENTITY,
    )
    ds = toy_dataset()
    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="col2",
        loss=faulty_custom_loss,
        task=generic_learner.Task.REGRESSION,
    )
    with self.assertRaisesRegex(
        RuntimeError,
        'Cannot hold a reference to "labels" outside of a custom loss'
        " function.*",
    ):
      _ = learner_custom_loss.train(ds)

  @parameterized.parameters(
      (
          custom_loss.RegressionLoss,
          generic_learner.Task.CLASSIFICATION,
          "label",
      ),
      (
          custom_loss.BinaryClassificationLoss,
          generic_learner.Task.REGRESSION,
          "col2",
      ),
      (
          custom_loss.MultiClassificationLoss,
          generic_learner.Task.REGRESSION,
          "col2",
      ),
  )
  def test_invalid_tasks(self, loss_type, task_type, label):
    custom_loss_container = loss_type(
        initial_predictions=lambda x, y: np.float32(0),
        gradient_and_hessian=lambda x, y: np.ones([6, len(x)]),
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.MultiClassificationLoss.Activation.IDENTITY,
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
    def mse_inital_predictions(
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

    # Path to dataset.
    dataset_directory = os.path.join(test_utils.ydf_test_data_path(), "dataset")
    train_path = os.path.join(
        dataset_directory, "two_center_regression_train.csv"
    )
    test_path = os.path.join(
        dataset_directory, "two_center_regression_test.csv"
    )
    train_ds = pd.read_csv(train_path)
    test_ds = pd.read_csv(test_path)
    mse_custom_loss = custom_loss.RegressionLoss(
        initial_predictions=mse_inital_predictions,
        gradient_and_hessian=mse_gradient,
        loss=mse_loss,
        activation=custom_loss.RegressionLoss.Activation.IDENTITY,
    )

    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="target",
        loss=mse_custom_loss,
        task=generic_learner.Task.REGRESSION,
        num_trees=30,
    )
    model_custom_loss: generic_model.GenericModel = learner_custom_loss.train(
        train_ds
    )

    learner_builtin_loss = specialized_learners.GradientBoostedTreesLearner(
        label="target",
        task=generic_learner.Task.REGRESSION,
        num_trees=30,
    )
    model_builtin_loss: generic_model.GenericModel = learner_builtin_loss.train(
        train_ds
    )
    npt.assert_allclose(
        model_custom_loss.predict(test_ds),
        model_builtin_loss.predict(test_ds),
    )

  @parameterized.parameters(
      custom_loss.BinaryClassificationLoss.Activation.IDENTITY,
      custom_loss.BinaryClassificationLoss.Activation.SIGMOID,
  )
  def test_binomial_custom_equal_to_builtin(self, activation):
    def binomial_inital_predictions(
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

    # Path to dataset.
    dataset_directory = os.path.join(test_utils.ydf_test_data_path(), "dataset")
    train_path = os.path.join(dataset_directory, "adult_train.csv")
    test_path = os.path.join(dataset_directory, "adult_test.csv")
    train_ds = pd.read_csv(train_path)
    test_ds = pd.read_csv(test_path)
    binomial_custom_loss = custom_loss.BinaryClassificationLoss(
        initial_predictions=binomial_inital_predictions,
        gradient_and_hessian=binomial_gradient,
        loss=binomial_loss,
        activation=activation,
    )

    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        loss=binomial_custom_loss,
        task=generic_learner.Task.CLASSIFICATION,
        num_trees=30,
    )
    model_custom_loss: generic_model.GenericModel = learner_custom_loss.train(
        train_ds
    )

    learner_builtin_loss = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        task=generic_learner.Task.CLASSIFICATION,
        apply_link_function=(
            activation
            == custom_loss.BinaryClassificationLoss.Activation.SIGMOID
        ),
        num_trees=30,
    )
    model_builtin_loss: generic_model.GenericModel = learner_builtin_loss.train(
        train_ds
    )
    npt.assert_allclose(
        model_custom_loss.predict(test_ds),
        model_builtin_loss.predict(test_ds),
        atol=1e-6,
    )

  @parameterized.parameters(
      custom_loss.MultiClassificationLoss.Activation.IDENTITY,
      custom_loss.MultiClassificationLoss.Activation.SOFTMAX,
  )
  def test_multinomial_custom_equal_to_builtin(self, activation):
    def multinomial_inital_predictions(
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
        initial_predictions=multinomial_inital_predictions,
        gradient_and_hessian=multinomial_gradient,
        loss=multinomial_loss,
        activation=activation,
    )

    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="LABEL",
        loss=multinomial_custom_loss,
        task=generic_learner.Task.CLASSIFICATION,
        num_trees=30,
    )
    model_custom_loss: generic_model.GenericModel = learner_custom_loss.train(
        train_ds
    )

    learner_builtin_loss = specialized_learners.GradientBoostedTreesLearner(
        label="LABEL",
        task=generic_learner.Task.CLASSIFICATION,
        apply_link_function=(
            activation == custom_loss.MultiClassificationLoss.Activation.SOFTMAX
        ),
        num_trees=30,
    )
    model_builtin_loss: generic_model.GenericModel = learner_builtin_loss.train(
        train_ds
    )
    npt.assert_allclose(
        model_custom_loss.predict(test_ds),
        model_builtin_loss.predict(test_ds),
        atol=1e-6,
    )

  def test_multiclass_initial_prediction(self):
    def multiclass_inital_prediction(
        labels: npty.NDArray[np.int32], _: npty.NDArray[np.float32]
    ) -> npty.NDArray[np.float32]:
      dimension = np.max(labels)
      return np.arange(1, dimension + 1)

    multiclass_custom_loss = custom_loss.MultiClassificationLoss(
        initial_predictions=multiclass_inital_prediction,
        gradient_and_hessian=lambda x, y: (
            np.ones([3, len(x)]),
            np.ones([3, len(x)]),
        ),
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.MultiClassificationLoss.Activation.IDENTITY,
    )
    ds = toy_dataset()
    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="col1",
        loss=multiclass_custom_loss,
        task=generic_learner.Task.CLASSIFICATION,
        num_trees=1,
    )
    model: gradient_boosted_trees_model.GradientBoostedTreesModel = (
        learner_custom_loss.train(ds)
    )
    npt.assert_equal(model.initial_predictions(), [1, 2, 3])

  def test_multiclass_wrong_initial_prediction_dimensions(self):
    def multiclass_inital_prediction(
        labels: npty.NDArray[np.int32], _: npty.NDArray[np.float32]
    ) -> npty.NDArray[np.float32]:
      dimension = np.max(labels)
      return np.arange(1, dimension)

    multiclass_custom_loss = custom_loss.MultiClassificationLoss(
        initial_predictions=multiclass_inital_prediction,
        gradient_and_hessian=lambda x, y: (
            np.ones([3, len(x)]),
            np.ones([3, len(x)]),
        ),
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.MultiClassificationLoss.Activation.IDENTITY,
    )
    ds = toy_dataset()
    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="col1",
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
          "col2",
      ),
      (
          custom_loss.BinaryClassificationLoss,
          generic_learner.Task.CLASSIFICATION,
          "label",
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
        activation=loss_type.Activation.IDENTITY,
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
          "col2",
      ),
      (
          custom_loss.BinaryClassificationLoss,
          generic_learner.Task.CLASSIFICATION,
          "label",
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
        activation=loss_type.Activation.IDENTITY,
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
          "col2",
      ),
      (
          custom_loss.BinaryClassificationLoss,
          generic_learner.Task.CLASSIFICATION,
          "label",
      ),
  )
  def test_stacked_gradient_dimensions(self, loss_type, task_type, label):
    faulty_loss = loss_type(
        initial_predictions=lambda x, y: np.float32(0),
        gradient_and_hessian=lambda x, y: 3 * np.ones([2, 5]),
        loss=lambda x, y, z: np.float32(0),
        activation=loss_type.Activation.IDENTITY,
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


if __name__ == "__main__":
  absltest.main()
