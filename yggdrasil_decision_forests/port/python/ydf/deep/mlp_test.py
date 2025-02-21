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

"""Test of the MLP learner. Also tests internal components of generic_jax."""

import os
import re

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from ydf.dataset import dataspec as dataspec_lib
from ydf.dataset.io import dataset_io as dataset_io_lib
from ydf.deep import mlp
from ydf.deep import model_lib
from ydf.learner import specialized_learners
from ydf.model import generic_model as generic_model_lib
from ydf.utils import test_utils


def dataset_path(filename: str) -> str:
  return os.path.join(test_utils.ydf_test_data_path(), "dataset", filename)


class MLPTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.adult = test_utils.load_datasets("adult")
    self.iris = pd.read_csv(dataset_path("iris.csv"))
    self.abalone = pd.read_csv(dataset_path("abalone.csv"))

  def test_adult_dataspec_is_same_as_cc_learner(self):
    learner_args = {
        "label": "income",
        "max_vocab_count": 20,
        "min_vocab_frequency": 5,
    }
    cart_learner = specialized_learners.CartLearner(**learner_args)
    mlp_learner = mlp.MultiLayerPerceptronLearner(**learner_args)
    cart_dataspec = cart_learner.train(self.adult.train_pd).data_spec()
    # Note: Training of an MLP take time. This way is faster.
    mlp_dataspec = mlp_learner._infer_dataspec(
        dataset_io_lib.build_batched_example_generator(self.adult.train_pd)
    )

    # Handle rounding errors in an open-source way.
    str_mlp_dataspec = str(mlp_dataspec)
    str_cart_dataspec = str(cart_dataspec)

    # Floating point rounding differences
    def clean(x):
      x = re.sub(
          "standard_deviation: 7509\\.[0-9]+",
          "standard_deviation: 7509",
          x,
      )
      x = re.sub(
          "standard_deviation: 106423\\.[0-9]+",
          "standard_deviation: 106423",
          x,
      )
      return x

    str_mlp_dataspec = clean(str_mlp_dataspec)
    str_cart_dataspec = clean(str_cart_dataspec)

    # The MLP model computes quantiles.
    str_mlp_dataspec = re.sub(
        r"\s+discretized_numerical {\n(\s+boundaries: \S+\n)*\s+}",
        "",
        str_mlp_dataspec,
    )

    self.assertEqual(str_mlp_dataspec, str_cart_dataspec)

  def test_binary_class_adult(self):
    learner = mlp.MultiLayerPerceptronLearner(label="income")
    self.assertDictEqual(
        learner.hyperparameters,
        {
            "batch_size": 512,
            "drop_out": 0.05,
            "layer_size": 200,
            "learning_rate": 0.01,
            "learning_rate_policy": "cosine_decay",
            "num_epochs": 1000,
            "num_layers": 8,
            "num_steps": None,
            "random_seed": 1234,
            "early_stopping_epoch_patience": 10,
            "early_stopping_revert_params": True,
            "maximum_training_duration_seconds": -1,
        },
    )
    model = learner.train(self.adult.train_pd, valid=self.adult.test_pd)
    predictions = model.predict(self.adult.test_pd)
    self.assertEqual(predictions.shape, (len(self.adult.test_pd),))
    evaluation = model.evaluate(self.adult.test_pd)
    logging.info("Evaluation:\n%s", evaluation)
    self.assertGreaterEqual(evaluation.accuracy, 0.845)
    self.assertLessEqual(evaluation.loss, 0.33)
    analysis = model.analyze(self.adult.test_pd, maximum_duration=5)
    _ = analysis.html()

  def test_regression_abalone(self):
    train_ds, test_ds = test_utils.split_ds(self.abalone)
    learner = mlp.MultiLayerPerceptronLearner(
        label="Rings", task=generic_model_lib.Task.REGRESSION, num_epochs=30
    )
    model = learner.train(train_ds, valid=test_ds, verbose=2)
    predictions = model.predict(test_ds)
    self.assertEqual(predictions.shape, (len(test_ds),))
    evaluation = model.evaluate(test_ds)
    logging.info("Evaluation:\n%s", evaluation)
    analysis = model.analyze(test_ds, maximum_duration=5)
    _ = analysis.html()

  def test_multiclass_iris(self):
    train_ds, test_ds = test_utils.split_ds(self.iris)
    learner = mlp.MultiLayerPerceptronLearner(label="class", num_epochs=30)
    model = learner.train(train_ds, verbose=2, valid=test_ds)
    predictions = model.predict(test_ds)
    self.assertEqual(predictions.shape, (len(test_ds), 3))
    evaluation = model.evaluate(test_ds)
    logging.info("Evaluation:\n%s", evaluation)
    analysis = model.analyze(test_ds, maximum_duration=5)
    _ = analysis.html()

  def test_interrupt_training(self):
    learner = mlp.MultiLayerPerceptronLearner(
        label="class", num_epochs=1000000, maximum_training_duration_seconds=10
    )
    _ = learner.train(self.iris, verbose=2)

  def test_bad_dataset(self):
    def make_ds(seed: int):
      rng = np.random.default_rng(seed)
      n = 10_000
      data = {
          "numerical_simple": rng.uniform(0, 1, size=n),
          "numerical_with_nan": np.where(
              rng.uniform(0, 1, size=n) < 0.5, rng.uniform(0, 1, size=n), np.nan
          ),
          "numerical_only_nan": np.full(fill_value=np.nan, shape=n),
          "numerical_constant": np.full(fill_value=2.0, shape=n),
          "categorical_only_nan": np.full(fill_value=np.nan, shape=n),
          "categorical_constant": np.full(fill_value="X", shape=n),
      }
      data["l"] = (
          data["numerical_simple"]
          + np.nan_to_num(data["numerical_with_nan"]) * 2
      ) >= 1
      return data

    train_ds = make_ds(1)
    test_ds = make_ds(2)
    learner = mlp.MultiLayerPerceptronLearner(
        label="l",
        num_epochs=30,
        features=[
            ("numerical_simple", dataspec_lib.Semantic.NUMERICAL),
            ("numerical_with_nan", dataspec_lib.Semantic.NUMERICAL),
            ("numerical_only_nan", dataspec_lib.Semantic.NUMERICAL),
            ("numerical_constant", dataspec_lib.Semantic.NUMERICAL),
            ("categorical_only_nan", dataspec_lib.Semantic.CATEGORICAL),
            ("categorical_constant", dataspec_lib.Semantic.CATEGORICAL),
        ],
    )
    model = learner.train(train_ds, verbose=2, valid=test_ds)
    _ = model.predict(test_ds)
    evaluation = model.evaluate(test_ds)
    logging.info("Evaluation:\n%s", evaluation)

  def test_save_and_load(self):
    train_ds, test_ds = test_utils.split_ds(self.abalone)
    learner = mlp.MultiLayerPerceptronLearner(
        label="Rings",
        task=generic_model_lib.Task.REGRESSION,
        num_epochs=5,
        num_layers=2,
        layer_size=100,
        drop_out=0.1,
    )
    model = learner.train(train_ds, valid=test_ds, verbose=2)
    predictions = model.predict(test_ds)
    ydf_model_path = os.path.join(self.create_tempdir().full_path, "ydf_model")
    model.save(ydf_model_path)
    loaded_model = model_lib.load_model(ydf_model_path)
    loaded_model_predictions = loaded_model.predict(test_ds)
    np.testing.assert_almost_equal(predictions, loaded_model_predictions)

  def test_weights_not_supported(self):
    with self.assertRaisesRegex(
        ValueError,
        "Training with sample weights or class weights is not yet supported for"
        " deep models.",
    ):
      _ = mlp.MultiLayerPerceptronLearner(label="Rings", weights="weights")

  def test_class_weights_not_supported(self):
    with self.assertRaisesRegex(
        ValueError,
        "Training with sample weights or class weights is not yet supported for"
        " deep models.",
    ):
      _ = mlp.MultiLayerPerceptronLearner(
          label="Rings", class_weights={"a": 1.0, "b": 2.0}
      )


if __name__ == "__main__":
  absltest.main()
