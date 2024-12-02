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

import os
import signal
from typing import Any, Dict, Optional, Tuple

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import fastavro
import numpy as np
import numpy.testing as npt
import pandas as pd
import polars as pl
from sklearn import metrics

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.dataset import dataspec
from ydf.learner import generic_learner
from ydf.learner import specialized_learners
from ydf.learner import tuner as tuner_lib
from ydf.metric import metric
from ydf.model import generic_model
from ydf.model import model_lib
from ydf.model.decision_forest_model import decision_forest_model
from ydf.model.tree import condition as condition_lib
from ydf.model.tree import node as node_lib
from ydf.utils import log
from ydf.utils import test_utils

ProtoMonotonicConstraint = abstract_learner_pb2.MonotonicConstraint
Column = dataspec.Column


def get_tree_depth(
    current_node: Any,
    depth: int,
):
  if current_node.is_leaf:
    return depth
  return max(
      get_tree_depth(current_node.neg_child, depth + 1),
      get_tree_depth(current_node.pos_child, depth + 1),
  )


class LearnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dataset_directory = os.path.join(
        test_utils.ydf_test_data_path(), "dataset"
    )

    self.adult = test_utils.load_datasets("adult")
    self.two_center_regression = test_utils.load_datasets(
        "two_center_regression"
    )
    self.synthetic_ranking = test_utils.load_datasets(
        "synthetic_ranking",
        [Column("GROUP", semantic=dataspec.Semantic.HASH)],
    )
    self.sim_pte = test_utils.load_datasets(
        "sim_pte",
        [
            Column("y", semantic=dataspec.Semantic.CATEGORICAL),
            Column("treat", semantic=dataspec.Semantic.CATEGORICAL),
        ],
    )
    self.gaussians = test_utils.load_datasets(
        "gaussians",
        column_args=[
            Column("label", semantic=dataspec.Semantic.CATEGORICAL),
            Column("features.0_of_2", semantic=dataspec.Semantic.NUMERICAL),
            Column("features.1_of_2", semantic=dataspec.Semantic.NUMERICAL),
        ],
    )

  def _check_adult_model(
      self,
      learner: generic_learner.GenericLearner,
      minimum_accuracy: float,
      check_serialization: bool = True,
      use_pandas: bool = False,
      valid: Optional[Any] = None,
  ) -> Tuple[generic_model.GenericModel, metric.Evaluation, np.ndarray]:
    """Runs a battery of test on a model compatible with the adult dataset.

    The following tests are run:
      - Train the model.
      - Run and evaluate the model.
      - Serialize the model to a YDF model.
      - Load the serialized model.
      - Make sure predictions of original model and serialized model match.

    Args:
      learner: A learner on the adult dataset.
      minimum_accuracy: Minimum accuracy.
      check_serialization: If true, check the serialization of the model.
      use_pandas: If true, load the dataset from Pandas
      valid: Optional validation dataset.

    Returns:
      The model, its evaluation and the predictions on the test dataset.
    """
    if use_pandas:
      train_ds = self.adult.train_pd
      test_ds = self.adult.test_pd
    else:
      train_ds = self.adult.train
      test_ds = self.adult.test

    # Train the model.
    model = learner.train(train_ds, valid=valid)

    # Evaluate the trained model.
    evaluation = model.evaluate(test_ds)
    self.assertGreaterEqual(evaluation.accuracy, minimum_accuracy)

    predictions = model.predict(test_ds)

    if check_serialization:
      ydf_model_path = os.path.join(
          self.create_tempdir().full_path, "ydf_model"
      )
      model.save(ydf_model_path)
      loaded_model = model_lib.load_model(ydf_model_path)
      npt.assert_equal(predictions, loaded_model.predict(test_ds))

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

  def test_adult_classification_on_tfrecord_dataset(self):
    learner = specialized_learners.RandomForestLearner(label="income")
    model = learner.train(
        "tfrecord:"
        + os.path.join(
            test_utils.ydf_test_data_path(),
            "dataset",
            "adult_train.recordio.gz",
        )
    )
    logging.info("Trained model: %s", model)

    # Evaluate the trained model.
    evaluation = model.evaluate(
        "tfrecord:"
        + os.path.join(
            test_utils.ydf_test_data_path(), "dataset", "adult_test.recordio.gz"
        )
    )
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
    data_spec = ds_pb.DataSpecification()
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
        learner.train(test_utils.toy_dataset()).task(),
        generic_learner.Task.REGRESSION,
    )

  def test_model_type_regression_unordered_set_indices(self):
    ds = pd.DataFrame({
        "col_cat_set": [
            ["A", "A"],
            ["A", "B"],
            ["A", "B"],
            ["C", "B"],
            ["C", "B"],
        ],
        "binary_int_label": [0, 0, 1, 1, 1],
    })
    learner = specialized_learners.RandomForestLearner(
        label="binary_int_label",
        num_trees=1,
        task=generic_learner.Task.REGRESSION,
        min_vocab_frequency=1,
        num_candidate_attributes_ratio=1.0,
        bootstrap_training_dataset=False,
        min_examples=1,
        categorical_set_split_greedy_sampling=1.0,
    )
    _ = learner.train(ds)

  def test_model_type_classification_string_label(self):
    learner = specialized_learners.RandomForestLearner(
        label="col_three_string",
        num_trees=1,
        task=generic_learner.Task.CLASSIFICATION,
    )
    self.assertEqual(
        learner.train(test_utils.toy_dataset()).task(),
        generic_learner.Task.CLASSIFICATION,
    )

  def test_model_type_classification_int_label(self):
    learner = specialized_learners.RandomForestLearner(
        label="binary_int_label",
        num_trees=1,
        task=generic_learner.Task.CLASSIFICATION,
    )
    self.assertEqual(
        learner.train(test_utils.toy_dataset()).task(),
        generic_learner.Task.CLASSIFICATION,
    )

  def test_regression_on_categorical_fails(self):
    learner = specialized_learners.RandomForestLearner(
        label="col_three_string",
        num_trees=1,
        task=generic_learner.Task.REGRESSION,
    )
    with self.assertRaises(ValueError):
      _ = learner.train(test_utils.toy_dataset())

  def test_classification_on_floats_fails(self):
    learner = specialized_learners.RandomForestLearner(
        label="col_float",
        num_trees=1,
        task=generic_learner.Task.CLASSIFICATION,
    )

    with self.assertRaises(ValueError):
      _ = (
          learner.train(test_utils.toy_dataset()).task(),
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
        learner.train(test_utils.toy_dataset_uplift()).task(),
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
        learner.train(test_utils.toy_dataset_uplift()).task(),
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

  def test_cross_validation_classification(self):
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

  def test_cross_validation_regression(self):
    learner = specialized_learners.RandomForestLearner(
        label="target", num_trees=10, task=generic_learner.Task.REGRESSION
    )
    evaluation = learner.cross_validation(
        self.two_center_regression.train, folds=10, parallel_evaluations=2
    )
    logging.info("evaluation:\n%s", evaluation)
    self.assertAlmostEqual(evaluation.rmse, 116, delta=1)
    # All the examples are used in the evaluation
    self.assertEqual(
        evaluation.num_examples,
        self.two_center_regression.train.data_spec().created_num_rows,
    )

    with open(self.create_tempfile(), "w") as f:
      f.write(evaluation._repr_html_())

  def test_cross_validation_uplift(self):
    learner = specialized_learners.RandomForestLearner(
        label="y",
        uplift_treatment="treat",
        num_trees=10,
        task=generic_learner.Task.CATEGORICAL_UPLIFT,
    )
    evaluation = learner.cross_validation(
        self.sim_pte.train, folds=10, parallel_evaluations=2
    )
    logging.info("evaluation:\n%s", evaluation)
    self.assertAlmostEqual(evaluation.qini, 0.06893, delta=1)
    # All the examples are used in the evaluation
    self.assertEqual(
        evaluation.num_examples, self.sim_pte.train.data_spec().created_num_rows
    )

    with open(self.create_tempfile(), "w") as f:
      f.write(evaluation._repr_html_())

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
    self.assertGreaterEqual(evaluation.ndcg, 0.70)
    self.assertLessEqual(evaluation.ndcg, 0.75)
    # All the examples are used in the evaluation
    self.assertEqual(
        evaluation.num_examples,
        self.synthetic_ranking.train.data_spec().created_num_rows,
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
    self.assertLessEqual(
        {
            "num_trees": 50,
            "categorical_algorithm": "CART",
            "categorical_set_split_greedy_sampling": 0.1,
            "compute_oob_performances": True,
            "compute_oob_variable_importances": False,
        }.items(),
        learner.hyperparameters.items(),
    )

  def test_unicode_dataset(self):
    data = {
        "\u0080feature": np.array(
            [["\u0080a", "\u0080b"], ["\u0080c", "\u0080d"]]
        ),
        "label": np.array([0, 1]),
    }
    learner = specialized_learners.RandomForestLearner(
        label="label", min_vocab_frequency=1
    )
    model = learner.train(data)

    expected_columns = [
        ds_pb.Column(
            name="\u0080feature.0_of_2",
            type=ds_pb.ColumnType.CATEGORICAL,
            dtype=ds_pb.DType.DTYPE_BYTES,
            count_nas=0,
            categorical=ds_pb.CategoricalSpec(
                items={
                    "<OOD>": ds_pb.CategoricalSpec.VocabValue(index=0, count=0),
                    "\u0080a": ds_pb.CategoricalSpec.VocabValue(
                        index=1, count=1
                    ),
                    "\u0080c": ds_pb.CategoricalSpec.VocabValue(
                        index=2, count=1
                    ),
                },
                number_of_unique_values=3,
            ),
            is_unstacked=True,
        ),
        ds_pb.Column(
            name="\u0080feature.1_of_2",
            type=ds_pb.ColumnType.CATEGORICAL,
            dtype=ds_pb.DType.DTYPE_BYTES,
            count_nas=0,
            categorical=ds_pb.CategoricalSpec(
                items={
                    "<OOD>": ds_pb.CategoricalSpec.VocabValue(index=0, count=0),
                    "\u0080b": ds_pb.CategoricalSpec.VocabValue(
                        index=1, count=1
                    ),
                    "\u0080d": ds_pb.CategoricalSpec.VocabValue(
                        index=2, count=1
                    ),
                },
                number_of_unique_values=3,
            ),
            is_unstacked=True,
        ),
    ]
    # Skip the first column that contains the label.
    self.assertEqual(model.data_spec().columns[1:], expected_columns)

    predictions = model.predict(data)
    self.assertEqual(predictions.shape, (2,))

  def test_multidimensional_training_dataset(self):
    data = {
        "feature": np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
        "label": np.array([0, 1]),
    }
    learner = specialized_learners.RandomForestLearner(label="label")
    model = learner.train(data)

    expected_columns = [
        ds_pb.Column(
            name="feature.0_of_4",
            type=ds_pb.ColumnType.NUMERICAL,
            dtype=ds_pb.DType.DTYPE_INT64,
            count_nas=0,
            numerical=ds_pb.NumericalSpec(
                mean=2,
                standard_deviation=2,
                min_value=0,
                max_value=4,
            ),
            is_unstacked=True,
        ),
        ds_pb.Column(
            name="feature.1_of_4",
            type=ds_pb.ColumnType.NUMERICAL,
            dtype=ds_pb.DType.DTYPE_INT64,
            count_nas=0,
            numerical=ds_pb.NumericalSpec(
                mean=3,
                standard_deviation=2,
                min_value=1,
                max_value=5,
            ),
            is_unstacked=True,
        ),
        ds_pb.Column(
            name="feature.2_of_4",
            type=ds_pb.ColumnType.NUMERICAL,
            dtype=ds_pb.DType.DTYPE_INT64,
            count_nas=0,
            numerical=ds_pb.NumericalSpec(
                mean=4,
                standard_deviation=2,
                min_value=2,
                max_value=6,
            ),
            is_unstacked=True,
        ),
        ds_pb.Column(
            name="feature.3_of_4",
            type=ds_pb.ColumnType.NUMERICAL,
            dtype=ds_pb.DType.DTYPE_INT64,
            count_nas=0,
            numerical=ds_pb.NumericalSpec(
                mean=5,
                standard_deviation=2,
                min_value=3,
                max_value=7,
            ),
            is_unstacked=True,
        ),
    ]
    # Skip the first column that contains the label.
    self.assertEqual(model.data_spec().columns[1:], expected_columns)

    predictions = model.predict(data)
    self.assertEqual(predictions.shape, (2,))

  def test_multidimensional_features_with_feature_arg(self):
    ds = {
        "f1": np.random.uniform(size=(100, 5)),
        "f2": np.random.uniform(size=(100, 5)),
        "label": np.random.randint(0, 2, size=100),
    }
    learner = specialized_learners.RandomForestLearner(
        label="label",
        features=["f1"],
        num_trees=3,
    )
    model = learner.train(ds)
    self.assertEqual(
        model.input_feature_names(),
        ["f1.0_of_5", "f1.1_of_5", "f1.2_of_5", "f1.3_of_5", "f1.4_of_5"],
    )

  def test_multidimensional_labels(self):
    ds = {
        "feature": np.array([[0, 1], [1, 0]]),
        "label": np.array([[0, 1], [1, 0]]),
    }
    learner = specialized_learners.RandomForestLearner(label="label")
    with self.assertRaisesRegex(
        test_utils.AbslInvalidArgumentError,
        "The column 'label' is multi-dimensional \\(shape=\\(2, 2\\)\\) while"
        " the model requires this column to be single-dimensional",
    ):
      _ = learner.train(ds)

  def test_multidimensional_weights(self):
    ds = {
        "feature": np.array([[0, 1], [1, 0]]),
        "label": np.array([0, 1]),
        "weight": np.array([[0, 1], [1, 0]]),
    }
    learner = specialized_learners.RandomForestLearner(
        label="label", weights="weight"
    )
    with self.assertRaisesRegex(
        test_utils.AbslInvalidArgumentError,
        "The column 'weight' is multi-dimensional \\(shape=\\(2, 2\\)\\) while"
        " the model requires this column to be single-dimensional",
    ):
      _ = learner.train(ds)

  def test_weighted_training_and_evaluation(self):

    def gen_ds(seed, n=10000):
      np.random.seed(seed)
      f1 = np.random.uniform(size=n)
      f2 = np.random.uniform(size=n)
      f3 = np.random.uniform(size=n)
      weights = np.random.uniform(size=n)
      return {
          "f1": f1,
          "f2": f2,
          "f3": f3,
          "label": (
              # Make the examples with high weights harder to predict.
              f1 + f2 * 0.5 + f3 * 0.5 + np.random.uniform(size=n) * weights
              >= 1.5
          ),
          "weights": weights,
      }

    model = specialized_learners.RandomForestLearner(
        label="label",
        weights="weights",
        num_trees=300,
        winner_take_all=False,
    ).train(gen_ds(0))

    test_ds = gen_ds(1)

    self_evaluation = model.self_evaluation()
    non_weighted_evaluation = model.evaluate(test_ds, weighted=False)
    weighted_evaluation = model.evaluate(test_ds, weighted=True)

    self.assertIsNotNone(self_evaluation)
    self.assertAlmostEqual(self_evaluation.accuracy, 0.824501, delta=0.005)
    self.assertAlmostEqual(
        non_weighted_evaluation.accuracy, 0.8417, delta=0.005
    )
    self.assertAlmostEqual(weighted_evaluation.accuracy, 0.8172290, delta=0.005)
    predictions = model.predict(test_ds)

    manual_non_weighted_evaluation = np.mean(
        (predictions >= 0.5) == test_ds["label"]
    )
    manual_weighted_evaluation = np.sum(
        ((predictions >= 0.5) == test_ds["label"]) * test_ds["weights"]
    ) / np.sum(test_ds["weights"])
    self.assertAlmostEqual(
        manual_non_weighted_evaluation, non_weighted_evaluation.accuracy
    )
    self.assertAlmostEqual(
        manual_weighted_evaluation, weighted_evaluation.accuracy
    )

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
    model = learner.train(test_utils.toy_dataset())
    self.assertEqual(model.metadata().framework, "Python YDF")

  def test_model_metadata_does_not_populate_owner(self):
    learner = specialized_learners.RandomForestLearner(
        label="binary_int_label", num_trees=1
    )
    model = learner.train(test_utils.toy_dataset())
    self.assertEqual(model.metadata().owner, "")

  def test_adult_sparse_oblique(self):
    learner = specialized_learners.RandomForestLearner(
        label="income",
        num_trees=4,
        split_axis="SPARSE_OBLIQUE",
        sparse_oblique_weights="CONTINUOUS",
    )
    model = learner.train(self.adult.train)
    assert isinstance(model, decision_forest_model.DecisionForestModel)
    model.plot_tree().html()
    logging.info("Trained model: %s", model)

  def test_adult_mhld_oblique(self):
    learner = specialized_learners.RandomForestLearner(
        label="income", num_trees=4, split_axis="MHLD_OBLIQUE"
    )
    model = learner.train(self.adult.train)
    logging.info("Trained model: %s", model)

  def test_warning_categorical_numerical(self):
    ds = {
        "label": np.array([0, 1, 0, 1] * 5, dtype=np.int64),
        "f": np.array([0, 1, "", ""] * 5, dtype=np.object_),
    }
    learner = specialized_learners.RandomForestLearner(
        label="label", num_trees=4
    )
    _ = learner.train(ds)

  def test_label_is_dataset(self):
    with self.assertRaisesRegex(ValueError, "should be a string"):
      _ = specialized_learners.RandomForestLearner(
          label=np.array([1, 0])
      )  # pytype: disable=wrong-arg-types

  def test_wrong_shape_multidim_model(self):
    model = specialized_learners.RandomForestLearner(
        label="label", num_trees=5
    ).train({
        "feature": np.array([[0, 1], [3, 4]]),
        "label": np.array([0, 1]),
    })
    with self.assertRaisesRegex(
        ValueError,
        r"Column 'feature' is expected to be multi-dimensional with shape 2 but"
        r" it is single-dimensional. If you use Numpy arrays, the column is"
        r" expected to be an array of shape \[num_examples, 2\].",
    ):
      _ = model.predict({
          "feature": np.array([0, 1]),
      })
    with self.assertRaisesRegex(
        ValueError,
        r"Unexpected shape for multi-dimensional column 'feature'. Column has"
        r" shape 1 but is expected to have shape 2.",
    ):
      _ = model.predict({
          "feature": np.array([[0], [1]]),
      })
    with self.assertRaisesRegex(
        ValueError,
        r"Unexpected shape for multi-dimensional column 'feature'. Column has"
        r" shape 3 but is expected to have shape 2.",
    ):
      _ = model.predict({
          "feature": np.array([[0, 1, 2], [3, 4, 5]]),
      })

  def test_wrong_shape_singledim_model(self):
    model = specialized_learners.RandomForestLearner(
        label="label", num_trees=5
    ).train({
        "feature": np.array([0, 1]),
        "label": np.array([0, 1]),
    })
    with self.assertRaisesRegex(
        ValueError,
        r"Column 'feature' is expected to single-dimensional but it is"
        r" multi-dimensional with shape 2.",
    ):
      _ = model.predict({
          "feature": np.array([[0, 1]]),
      })
    with self.assertRaisesRegex(
        ValueError,
        r"Column 'feature' is expected to single-dimensional but it is"
        r" multi-dimensional with shape 1.",
    ):
      _ = model.predict({
          "feature": np.array([[0], [1]]),
      })

  def test_analyze_ensure_maximum_duration(self):
    # Create an analysis that would take a lot of time if not limited in time.
    def create_dataset(n: int) -> Dict[str, np.ndarray]:
      return {
          "feature": np.random.uniform(size=(n, 100)),
          "label": np.random.uniform(size=(n,)),
      }

    model = specialized_learners.RandomForestLearner(
        label="label", task=generic_learner.Task.REGRESSION
    ).train(create_dataset(1_000))
    _ = model.analyze(create_dataset(100_000))

  def test_boolean_feature(self):
    data = {
        "f1": np.array(
            [True, True, True, True, True, False, False, False, False, False]
        ),
        "label": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
    }
    model = specialized_learners.RandomForestLearner(
        label="label",
        features=[("f1", dataspec.Semantic.BOOLEAN)],
        num_trees=1,
        bootstrap_training_dataset=False,
    ).train(data)
    npt.assert_equal(model.predict(data), data["label"])

  def test_fail_gracefully_for_hash_columns(self):
    data = {
        "f1": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "label": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
    }
    with self.assertRaisesRegex(
        test_utils.AbslInvalidArgumentError,
        "Column f1 has type HASH, which is not supported for decision tree"
        " training.",
    ):
      _ = specialized_learners.RandomForestLearner(
          label="label", features=[("f1", dataspec.Semantic.HASH)], num_trees=2
      ).train(data)


class CARTLearnerTest(LearnerTest):

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
    self.assertAlmostEqual(evaluation.rmse, 114.081, places=3)

  def test_monotonic_non_compatible_learner(self):
    with self.assertRaisesRegex(
        test_utils.AbslInvalidArgumentError,
        "The learner CART does not support monotonic constraints",
    ):
      _ = specialized_learners.CartLearner(
          label="label", features=[dataspec.Column("feature", monotonic=+1)]
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

  @parameterized.named_parameters(
      {
          "testcase_name": "ndcg@2",
          "truncation": 2,
          "expected_ndcg": 0.723,
          "delta": 0.025,
      },
      {
          "testcase_name": "ndcg@5",
          "truncation": 5,
          "expected_ndcg": 0.716,
          "delta": 0.024,
      },
      {
          "testcase_name": "ndcg@10",
          "truncation": 10,
          "expected_ndcg": 0.716,
          "delta": 0.03,
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

    model, _, _ = self._check_adult_model(
        learner, minimum_accuracy=0.863, use_pandas=True
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


class LoggingTest(parameterized.TestCase):

  @parameterized.parameters(0, 1, 2, False, True)
  def test_logging_function(self, verbose):
    save_verbose = log.verbose(verbose)
    learner = specialized_learners.RandomForestLearner(label="label")
    ds = pd.DataFrame({"feature": [0, 1], "label": [0, 1]})
    _ = learner.train(ds)
    log.verbose(save_verbose)

  @parameterized.parameters(0, 1, 2, False, True)
  def test_logging_arg(self, verbose):
    learner = specialized_learners.RandomForestLearner(label="label")
    ds = pd.DataFrame({"feature": [0, 1], "label": [0, 1]})
    _ = learner.train(ds, verbose=verbose)


class IsolationForestLearnerTest(LearnerTest):

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
    _ = model.analyze(self.gaussians.test_pd)

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

    max_depth = max([get_tree_depth(t.root, 0) for t in model.get_all_trees()])
    self.assertEqual(max_depth, 8)

  def test_max_depth_gaussians_subsample_count(self):
    learner = specialized_learners.IsolationForestLearner(
        features=["features.0_of_2", "features.1_of_2"],
        subsample_count=128,
    )
    self.assertEqual(learner.hyperparameters["subsample_count"], 128)
    model = learner.train(self.gaussians.train_pd)

    max_depth = max([get_tree_depth(t.root, 0) for t in model.get_all_trees()])
    self.assertEqual(max_depth, 7)

  def test_max_depth_gaussians_max_depth(self):
    learner = specialized_learners.IsolationForestLearner(
        features=["features.0_of_2", "features.1_of_2"],
        subsample_ratio=1.0,
        max_depth=10,
    )
    model = learner.train(self.gaussians.train_pd)

    max_depth = max([get_tree_depth(t.root, 0) for t in model.get_all_trees()])
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


class DatasetFormatsTest(parameterized.TestCase):

  def features(self):
    return [
        "f1",
        "f2",
        "i1",
        "i2",
        "c1",
        "multi_c1",
        ("cs1", dataspec.Semantic.CATEGORICAL_SET),
        "multi_f1",
    ]

  def create_polars_dataset(self, n: int = 1000) -> pl.DataFrame:
    return pl.DataFrame({
        # Single-dim features
        "f1": np.random.random(size=n),
        "f2": np.random.random(size=n),
        "i1": np.random.randint(100, size=n),
        "i2": np.random.randint(100, size=n),
        "c1": np.random.choice(["x", "y", "z"], size=n, p=[0.6, 0.3, 0.1]),
        "multi_c1": np.array(
            [["a", "x", "z"], ["b", "x", "w"], ["a", "y", "w"], ["b", "y", "z"]]
            * (n // 4)
        ),
        # Cat-set features
        # ================
        # Note: Polars as a bug when serializing empty lists of string to Avro
        # files (only write one of the two required "optional" bit).
        # TODO: Replace [""] by [] once the bug if fixed is added.
        "cs1": [["<SOMETHING>"], ["a", "b", "c"], ["b", "c"], ["a"]] * (n // 4),
        # Multi-dim features
        # ==================
        # Note: It seems support for this type of feature was temporarly dropped
        # in Polars 1.9 i.e. the data packing was improved but the avro
        # serialization was not implemented. This code would fail with recent
        # version of polars with: not yet implemented: write
        # FixedSizeList(Field { name: "item", dtype: Float64, is_nullable: true,
        # metadata: {} }, 5) to avro.
        "multi_f1": np.random.random(size=(n, 3)),
        # # Labels
        "label_class_binary1": np.random.choice([False, True], size=n),
        "label_class_binary2": np.random.choice([0, 1], size=n),
        "label_class_binary3": np.random.choice(["l1", "l2"], size=n),
        "label_class_multi1": np.random.choice(["l1", "l2", "l3"], size=n),
        "label_class_multi2": np.random.choice([0, 1, 2], size=n),
        "label_regress1": np.random.random(size=n),
    })

  def test_avro_from_raw_fastavro(self):
    tmp_dir = self.create_tempdir().full_path
    ds_path = os.path.join(tmp_dir, "dataset.avro")
    schema = fastavro.parse_schema({
        "name": "ToyDataset",
        "doc": "A toy dataset.",
        "type": "record",
        "fields": [
            {"name": "f1", "type": "float"},
            {"name": "f2", "type": ["null", "float"]},
            {"name": "i1", "type": "int"},
            {"name": "c1", "type": "string"},
            {
                "name": "multi_f1",
                "type": {"type": "array", "items": "float"},
            },
            {
                "name": "multi_c1",
                "type": {"type": "array", "items": "string"},
            },
            {
                "name": "cs1",
                "type": {"type": "array", "items": "string"},
            },
            {
                "name": "cs2",
                "type": [
                    "null",
                    {"type": "array", "items": ["null", "string"]},
                ],
            },
            {"name": "l", "type": "float"},
        ],
    })
    records = []
    for _ in range(100):
      record = {
          "f1": np.random.rand(),
          "i1": np.random.randint(100),
          "c1": np.random.choice(["x", "y", "z"]),
          "multi_f1": [np.random.rand() for _ in range(3)],
          "multi_c1": [np.random.choice(["x", "y", "z"]) for _ in range(3)],
          "cs1": [
              np.random.choice(["x", "y", "z"])
              for _ in range(np.random.randint(3))
          ],
          "l": np.random.rand(),
      }
      if np.random.rand() < 0.8:
        record["f2"] = np.random.rand()
      if np.random.rand() < 0.8:
        record["cs2"] = [
            np.random.choice(["x", "y", None])
            for _ in range(np.random.randint(3))
        ]
      records.append(record)
    with open(ds_path, "wb") as out:
      fastavro.writer(out, schema, records, codec="deflate")
    learner = specialized_learners.RandomForestLearner(
        label="l",
        num_trees=3,
        features=[
            ("cs1", dataspec.Semantic.CATEGORICAL_SET),
            ("cs2", dataspec.Semantic.CATEGORICAL_SET),
        ],
        include_all_columns=True,
        task=generic_learner.Task.REGRESSION,
    )
    model = learner.train("avro:" + ds_path)
    self.assertEqual(model.num_trees(), 3)
    logging.info("model.input_features():\n%s", model.input_features())
    InputFeature = generic_model.InputFeature
    Semantic = dataspec.Semantic
    self.assertEqual(
        model.input_features(),
        [
            InputFeature(name="f1", semantic=Semantic.NUMERICAL, column_idx=0),
            InputFeature(name="f2", semantic=Semantic.NUMERICAL, column_idx=1),
            InputFeature(name="i1", semantic=Semantic.NUMERICAL, column_idx=2),
            InputFeature(
                name="c1", semantic=Semantic.CATEGORICAL, column_idx=3
            ),
            InputFeature(
                name="cs1", semantic=Semantic.CATEGORICAL_SET, column_idx=4
            ),
            InputFeature(
                name="cs2", semantic=Semantic.CATEGORICAL_SET, column_idx=5
            ),
            InputFeature(
                name="multi_f1.0_of_3",
                semantic=Semantic.NUMERICAL,
                column_idx=7,
            ),
            InputFeature(
                name="multi_f1.1_of_3",
                semantic=Semantic.NUMERICAL,
                column_idx=8,
            ),
            InputFeature(
                name="multi_f1.2_of_3",
                semantic=Semantic.NUMERICAL,
                column_idx=9,
            ),
            InputFeature(
                name="multi_c1.0_of_3",
                semantic=Semantic.CATEGORICAL,
                column_idx=10,
            ),
            InputFeature(
                name="multi_c1.1_of_3",
                semantic=Semantic.CATEGORICAL,
                column_idx=11,
            ),
            InputFeature(
                name="multi_c1.2_of_3",
                semantic=Semantic.CATEGORICAL,
                column_idx=12,
            ),
        ],
    )

  def test_avro_from_fastavro_with_pandas(self):
    tmp_dir = self.create_tempdir().full_path
    ds_path = os.path.join(tmp_dir, "dataset.avro")
    schema = fastavro.parse_schema({
        "name": "ToyDataset",
        "doc": "A toy dataset.",
        "type": "record",
        "fields": [
            {"name": "f1", "type": "float"},
            {"name": "f2", "type": "float"},
            {"name": "f3", "type": ["null", "float"]},
            {"name": "l", "type": "float"},
        ],
    })
    ds = pd.DataFrame({
        "f1": np.random.rand(100),
        "f2": np.random.rand(100),
        "f3": np.random.rand(100),
        "l": np.random.rand(100),
    })
    with open(ds_path, "wb") as out:
      fastavro.writer(out, schema, ds.to_dict("records"), codec="deflate")
    learner = specialized_learners.RandomForestLearner(
        label="l",
        num_trees=3,
        task=generic_learner.Task.REGRESSION,
    )
    model = learner.train("avro:" + ds_path)
    self.assertEqual(model.num_trees(), 3)


class UtilityTest(LearnerTest):

  def test_feature_name_to_regex(self):
    self.assertEqual(
        generic_learner._feature_name_to_regex("a(z)e"), r"^a\(z\)e$"
    )


if __name__ == "__main__":
  absltest.main()
