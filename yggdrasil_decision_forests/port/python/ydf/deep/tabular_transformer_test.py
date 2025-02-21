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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ydf.deep import layer as layer_lib
from ydf.deep import model_lib
from ydf.deep import tabular_transformer
from ydf.model import generic_model as generic_model_lib
from ydf.utils import test_utils


def dataset_path(filename: str) -> str:
  return os.path.join(test_utils.ydf_test_data_path(), "dataset", filename)


FeatureType = layer_lib.FeatureType
Feature = layer_lib.Feature


class FTTransformerTokenizerTest(parameterized.TestCase):

  def test_basic(self):
    m = tabular_transformer.FTTransformerTokenizer(
        tabular_transformer.FTTransformerTokenizer.Config(token_dim=3)
    )
    x = [
        (Feature("n1", FeatureType.NUMERICAL), jnp.array([1, 2])),
        (Feature("n2", FeatureType.NUMERICAL), jnp.array([[1, 2], [3, 4]])),
        (Feature("b1", FeatureType.BOOLEAN), jnp.array([True, False])),
        (
            Feature("c2", FeatureType.CATEGORICAL, num_categorical_values=3),
            jnp.array([0, 2]),
        ),
    ]
    state = m.init(jax.random.PRNGKey(0), x)
    self.assertDictEqual(
        jax.tree.map(lambda x: x.shape, state),
        {
            "params": {
                "bias_kernel": (6, 3),
                "embedding_c2": {"embedding": (3, 3)},
                "numerical_kernel": (4, 3),
            }
        },
    )
    y = m.apply(state, x)
    # 2 examples, 6 tokens, dim 3
    self.assertEqual(y.shape, (2, 6, 3))

    state = {
        "params": {
            "bias_kernel": jnp.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0],
            ]),
            "embedding_c2": {
                "embedding": jnp.array([
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                ])
            },
            "numerical_kernel": jnp.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]),
        }
    }

    y = m.apply(state, x)
    np.testing.assert_array_equal(
        y,
        jnp.array(
            [
                [
                    [1.0, 2.0, 3.0],
                    [5.0, 7.0, 9.0],
                    [11.0, 13.0, 15.0],
                    [24.0, 27.0, 30.0],
                    [23.0, 25.0, 27.0],
                    [16.1, 17.2, 18.3],
                ],
                [
                    [1.0, 2.0, 3.0],
                    [6.0, 9.0, 12.0],
                    [19.0, 23.0, 27.0],
                    [38.0, 43.0, 48.0],
                    [13.0, 14.0, 15.0],
                    [16.7, 17.8, 18.9],
                ],
            ],
            dtype=jnp.float32,
        ),
    )


class TabularTransformerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.adult = test_utils.load_datasets("adult")
    self.iris = pd.read_csv(dataset_path("iris.csv"))
    self.abalone = pd.read_csv(dataset_path("abalone.csv"))

  def test_simple_binary_class_adult(self):
    learner = tabular_transformer.TabularTransformerLearner(
        label="income", allow_cpu=True, num_epochs=10  # Faster
    )
    self.assertDictEqual(
        learner.hyperparameters,
        {
            "batch_size": 256,
            "num_layers": 3,
            "drop_out": 0.05,
            "num_heads": 4,
            "qkv_features": 16,
            "token_dim": 50,
            "learning_rate": 0.001,
            "learning_rate_policy": "cosine_decay",
            "num_epochs": 10,
            "num_steps": None,
            "random_seed": 1234,
            "early_stopping_epoch_patience": 10,
            "early_stopping_revert_params": True,
            "maximum_training_duration_seconds": -1.0,
        },
    )
    model = learner.train(self.adult.train_pd, valid=self.adult.test_pd)
    predictions = model.predict(self.adult.test_pd)
    self.assertEqual(predictions.shape, (len(self.adult.test_pd),))
    evaluation = model.evaluate(self.adult.test_pd)
    logging.info("Evaluation:\n%s", evaluation)
    self.assertGreaterEqual(evaluation.accuracy, 0.85)
    self.assertLessEqual(evaluation.loss, 0.33)

  def test_simple_regression_abalone(self):
    train_ds, test_ds = test_utils.split_ds(self.abalone)
    learner = tabular_transformer.TabularTransformerLearner(
        label="Rings",
        allow_cpu=True,
        task=generic_model_lib.Task.REGRESSION,
        num_epochs=30,
    )
    model = learner.train(train_ds, valid=test_ds, verbose=2)
    predictions = model.predict(test_ds)
    self.assertEqual(predictions.shape, (len(test_ds),))
    evaluation = model.evaluate(test_ds)
    logging.info("Evaluation:\n%s", evaluation)

  def test_simple_multiclass_iris(self):
    train_ds, test_ds = test_utils.split_ds(self.iris)
    learner = tabular_transformer.TabularTransformerLearner(
        label="class", allow_cpu=True, num_epochs=30
    )
    model = learner.train(train_ds, verbose=2, valid=test_ds)
    predictions = model.predict(test_ds)
    self.assertEqual(predictions.shape, (len(test_ds), 3))
    evaluation = model.evaluate(test_ds)
    logging.info("Evaluation:\n%s", evaluation)

  def test_save_and_load(self):
    train_ds, test_ds = test_utils.split_ds(self.abalone)
    learner = tabular_transformer.TabularTransformerLearner(
        label="Rings",
        allow_cpu=True,
        task=generic_model_lib.Task.REGRESSION,
        num_epochs=5,
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
      _ = tabular_transformer.TabularTransformerLearner(
          label="Rings",
          weights="weights",
          allow_cpu=True,
      )

  def test_class_weights_not_supported(self):
    with self.assertRaisesRegex(
        ValueError,
        "Training with sample weights or class weights is not yet supported for"
        " deep models.",
    ):
      _ = tabular_transformer.TabularTransformerLearner(
          label="Rings",
          class_weights={"a": 1.0, "b": 2.0},
          allow_cpu=True,
      )


if __name__ == "__main__":
  absltest.main()
