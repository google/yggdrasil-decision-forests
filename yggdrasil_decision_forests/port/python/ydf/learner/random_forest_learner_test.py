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

"""Tests for random forest learners."""

import os
import random
import signal
from typing import Dict

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import numpy.testing as npt
import pandas as pd

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from yggdrasil_decision_forests.learner.gradient_boosted_trees import gradient_boosted_trees_pb2 as _
from ydf.dataset import dataspec
from ydf.learner import generic_learner
from ydf.learner import learner_test_utils
from ydf.learner import specialized_learners
from ydf.model.decision_forest_model import decision_forest_model
from ydf.utils import test_utils


class RandomForestLearnerTest(learner_test_utils.LearnerTest):

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

    _ = evaluation._repr_html_()

  def test_cross_validation_regression(self):
    learner = specialized_learners.RandomForestLearner(
        label="target", num_trees=10, task=generic_learner.Task.REGRESSION
    )
    evaluation = learner.cross_validation(
        self.two_center_regression.train, folds=10, parallel_evaluations=2
    )
    logging.info("evaluation:\n%s", evaluation)
    self.assertAlmostEqual(evaluation.rmse, 118, delta=3)
    # All the examples are used in the evaluation
    self.assertEqual(
        evaluation.num_examples,
        self.two_center_regression.train.data_spec().created_num_rows,
    )

    _ = evaluation._repr_html_()

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

    _ = evaluation._repr_html_()

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

  @parameterized.parameters(
      {"weight_column_in_test": False}, {"weight_column_in_test": True}
  )
  def test_weighted_training_and_evaluation(self, weight_column_in_test):

    def gen_ds(seed: int, n: int, include_weight_column: bool):
      np.random.seed(seed)
      f1 = np.random.uniform(size=n)
      f2 = np.random.uniform(size=n)
      f3 = np.random.uniform(size=n)
      weights = np.random.uniform(size=n)
      ds = {
          "f1": f1,
          "f2": f2,
          "f3": f3,
          "label": (
              # Make the examples with high weights harder to predict.
              f1 + f2 * 0.5 + f3 * 0.5 + np.random.uniform(size=n) * weights
              >= 1.5
          ),
      }
      if include_weight_column:
        ds["weights"] = weights
      return ds

    model = specialized_learners.RandomForestLearner(
        label="label",
        weights="weights",
        num_trees=300,
        winner_take_all=False,
    ).train(gen_ds(0, 10000, include_weight_column=True))

    test_ds = gen_ds(1, 10000, include_weight_column=weight_column_in_test)

    self_evaluation = model.self_evaluation()
    non_weighted_evaluation = model.evaluate(test_ds, weighted=False)

    self.assertIsNotNone(self_evaluation)
    self.assertAlmostEqual(self_evaluation.accuracy, 0.824501, delta=0.005)
    self.assertAlmostEqual(
        non_weighted_evaluation.accuracy, 0.8417, delta=0.005
    )

    predictions = model.predict(test_ds)
    manual_non_weighted_evaluation = np.mean(
        (predictions >= 0.5) == test_ds["label"]
    )
    self.assertAlmostEqual(
        manual_non_weighted_evaluation, non_weighted_evaluation.accuracy
    )
    # Weighted evaluation only if the test dataset contains weights.
    if weight_column_in_test:
      weighted_evaluation = model.evaluate(test_ds, weighted=True)
      self.assertAlmostEqual(
          weighted_evaluation.accuracy, 0.8172290, delta=0.005
      )

      manual_weighted_evaluation = np.sum(
          ((predictions >= 0.5) == test_ds["label"]) * test_ds["weights"]
      ) / np.sum(test_ds["weights"])
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
    _ = model.analyze(create_dataset(100_000), maximum_duration=5)

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

  def test_numerical_vector_sequence(self):
    def make_ds():
      features = []
      labels = []
      num_examples = 1_000
      num_dims = 2
      distance_limit = 0.2
      for _ in range(num_examples):
        num_vectors = random.randint(0, 5)
        vectors = np.random.uniform(
            size=(num_vectors, num_dims), low=0.0, high=1.0
        )
        label = np.any(
            np.sum((vectors - np.array([[0.5, 0.5]])) ** 2, axis=1)
            < distance_limit * distance_limit
        )
        features.append(vectors)
        labels.append(label)
      return {"label": np.array(labels), "features": features}

    train_ds = make_ds()
    test_ds = make_ds()
    model = specialized_learners.RandomForestLearner(
        label="label",
        num_trees=50,
        numerical_vector_sequence_num_examples=2000,
    ).train(train_ds)
    evaluation = model.evaluate(test_ds)
    logging.info("Evaluation: %s", evaluation)
    self.assertGreaterEqual(evaluation.accuracy, 0.95)

    _ = model.analyze(test_ds)
    _ = model.get_tree(0)
    model.print_tree()

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


if __name__ == "__main__":
  absltest.main()
