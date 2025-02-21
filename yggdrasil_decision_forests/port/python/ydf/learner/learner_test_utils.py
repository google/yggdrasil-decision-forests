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

"""Helpers for Learner testing."""

import os
from typing import Any, Optional, Tuple

from absl.testing import parameterized
import numpy as np
import numpy.testing as npt

from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.learner.gradient_boosted_trees import gradient_boosted_trees_pb2 as _
from ydf.dataset import dataspec
from ydf.learner import generic_learner
from ydf.metric import metric
from ydf.model import generic_model
from ydf.model import model_lib
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
  """Abstract class for testing learners."""

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
