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

"""Tests for the custom losses."""

import os
from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import numpy.typing as npty
import pandas as pd

from ydf.dataset import dataspec
from ydf.learner import custom_loss
from ydf.learner import generic_learner
from ydf.learner import specialized_learners
from ydf.model import generic_model
from ydf.model.gradient_boosted_trees_model import gradient_boosted_trees_model
from ydf.utils import test_utils

Column = dataspec.Column


class CustomLossTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dataset_directory = os.path.join(
        test_utils.ydf_test_data_path(), "dataset"
    )

    self.adult = test_utils.load_datasets("adult")
    self.two_center_regression = test_utils.load_datasets(
        "two_center_regression"
    )

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

    def faulty_initial_prediction(*_):
      raise NotImplementedError("Faulty initial prediction")

    faulty_custom_loss = loss_type(
        initial_predictions=faulty_initial_prediction,
        gradient_and_hessian=lambda x, y: (np.ones(len(x)), np.ones(len(x))),
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.Activation.IDENTITY,
    )
    ds = test_utils.toy_dataset()
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

    def faulty_gradient_and_hessian(labels, predictions):
      nonlocal ref_to_labels
      ref_to_labels = labels
      return (np.ones(len(labels)), np.ones(len(predictions)))

    faulty_custom_loss = custom_loss.RegressionLoss(
        initial_predictions=lambda x, y: np.float32(0),
        gradient_and_hessian=faulty_gradient_and_hessian,
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.Activation.IDENTITY,
    )
    ds = test_utils.toy_dataset()
    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="col_float",
        loss=faulty_custom_loss,
        task=generic_learner.Task.REGRESSION,
        num_trees=5,
    )
    with self.assertRaisesRegex(
        RuntimeError,
        'Cannot hold a reference to "labels" outside of a custom loss'
        " function.*",
    ):
      _ = learner_custom_loss.train(ds)

  def test_honor_trigger_gc(self):
    ref_to_labels = None

    def faulty_gradient_and_hessian(labels, predictions):
      nonlocal ref_to_labels
      ref_to_labels = labels
      return (np.ones(len(labels)), np.ones(len(predictions)))

    faulty_custom_loss = custom_loss.RegressionLoss(
        initial_predictions=lambda x, y: np.float32(0),
        gradient_and_hessian=faulty_gradient_and_hessian,
        loss=lambda x, y, z: np.float32(0),
        activation=custom_loss.Activation.IDENTITY,
        may_trigger_gc=False,
    )
    ds = test_utils.toy_dataset()
    learner_custom_loss = specialized_learners.GradientBoostedTreesLearner(
        label="col_float",
        loss=faulty_custom_loss,
        task=generic_learner.Task.REGRESSION,
        num_trees=5,
    )
    model = learner_custom_loss.train(ds)
    self.assertEqual(model.num_trees(), 5)

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
    ds = test_utils.toy_dataset()
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
    ds = test_utils.toy_dataset()
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
      return (predictions - labels, -np.ones(labels.shape))

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
          pred_probability - binary_labels,
          pred_probability * (pred_probability - 1),
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
        apply_link_function=(activation == custom_loss.Activation.SIGMOID),
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
      gradient = normalized_predictions - label_indicator
      hessian = np.abs(gradient) * (np.abs(gradient) - 1)
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
        apply_link_function=(activation == custom_loss.Activation.SOFTMAX),
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
    ds = test_utils.toy_dataset()
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
    ds = test_utils.toy_dataset()
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
    ds = test_utils.toy_dataset()
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
    ds = test_utils.toy_dataset()
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
    ds = test_utils.toy_dataset()
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
          mse_grad(labels, predictions).block_until_ready(),
          jnp.diagonal(mse_hessian(labels, predictions)).block_until_ready(),
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
          mse_grad(labels, predictions).block_until_ready(),
          jnp.diagonal(mse_hessian(labels, predictions)).block_until_ready(),
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
