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

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
import tensorflow as tf

from ydf.learner import generic_learner
from ydf.learner import specialized_learners


def toy_dataset():
  df = pd.DataFrame({
      "col1": ["A", "A", "B", "B", "C"],
      "col2": [1, 2.1, 1.3, 5.5, 2.4],
      "col3": ["bar", "foo", "foo", "foo", "foo"],
      "weights": [3, 2, 3.1, 28, 3],
      "label": [0, 0, 0, 1, 1],
  })
  return df


class RandomForestLearnerTest(parameterized.TestCase):

  @parameterized.parameters({"use_cache": True}, {"use_filter": True})
  def test_tensorflow_dataset(
      self, use_cache: bool = False, use_filter: bool = False
  ):
    learner = specialized_learners.RandomForestLearner(
        label="label", num_trees=1
    )
    tf_dataset = tf.data.Dataset.from_tensor_slices(dict(toy_dataset())).batch(
        10
    )
    for x in tf_dataset.take(2):
      print(x)
    if use_cache:
      tf_dataset = tf_dataset.cache()
    if use_filter:
      tf_dataset = tf_dataset.filter(lambda x: True)
    self.assertEqual(
        learner.train(tf_dataset).task(), generic_learner.Task.CLASSIFICATION
    )


if __name__ == "__main__":
  absltest.main()
