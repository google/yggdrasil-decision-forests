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

from absl import flags
from absl import logging
from absl.testing import absltest
import numpy as np
import pandas as pd

from yggdrasil_decision_forests.dataset import data_spec_pb2
from ydf.dataset import dataset
from ydf.learner import generic_learner
from ydf.learner import specialized_learners

DatasetForTesting = collections.namedtuple(
    "Dataset",
    [
        "train",
        "test",
        "label",
    ],
)


def data_root_path() -> str:
  return ""


def ydf_test_data_path() -> str:
  return os.path.join(
      data_root_path(), "external/ydf_cc/yggdrasil_decision_forests/test_data"
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


def adult_dataset() -> DatasetForTesting:
  """Adult/census binary classification dataset."""

  # Path to dataset.
  dataset_directory = os.path.join(ydf_test_data_path(), "dataset")
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
  ):
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
        ydf_test_data_path(), "model", "adult_binary_class_rf", "data_spec.pb"
    )
    predictions_path = os.path.join(
        ydf_test_data_path(), "prediction", "adult_test_binary_class_rf.csv"
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


class CARTLearnerTest(LearnerTest):

  def test_default_settings(self):
    ds = adult_dataset()
    learner = specialized_learners.CartLearner(label=ds.label)

    self._check_adult_model(learner=learner, ds=ds, minimum_accuracy=0.853)


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


if __name__ == "__main__":
  absltest.main()
