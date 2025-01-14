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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from ydf.deep import dataset as deep_dataset_lib


class DatasetTest(parameterized.TestCase):

  def test_get_num_examples(self):
    self.assertEqual(
        deep_dataset_lib.get_num_examples({"a": np.array([1, 2])}), 2
    )

  def test_batch_numpy_to_jax(self):
    result = deep_dataset_lib.batch_numpy_to_jax({"a": np.array([1, 2])})
    self.assertEqual(set(result.keys()), set(["a"]))
    np.testing.assert_array_equal(result["a"], jax.numpy.array([1, 2]))


if __name__ == "__main__":
  absltest.main()
