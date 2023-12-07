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
import pandas as pd

from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.dataset import dataset
from ydf.dataset import dataspec
from ydf.learner import generic_learner
from ydf.learner import specialized_learners
from ydf.learner import tuner as tuner_lib
from ydf.metric import metric
from ydf.model import generic_model
from ydf.model.decision_forest_model import decision_forest_model
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

    signal.alarm(3)
    model = learner.train(ds)
    self.assertEqual(model.task(), generic_learner.Task.CLASSIFICATION)
    # Test that the model is functionnal
    _ = model.evaluate(ds)

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
        num_trials=5, automatic_search_space=True
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
        num_trials=5, automatic_search_space=True
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
        learner, ds=vds_dataset, minimum_accuracy=0.864
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


if __name__ == "__main__":
  absltest.main()
