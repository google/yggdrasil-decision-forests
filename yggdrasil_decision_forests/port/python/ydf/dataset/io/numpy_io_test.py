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

"""Test dataspec utilities for pandas."""

from absl.testing import absltest
import numpy as np

from ydf.dataset.io import numpy_io
from ydf.utils import test_utils


class NumpyIOTest(absltest.TestCase):

  def test_numpy_generator(self):
    ds = numpy_io.NumpyDictBatchedExampleGenerator({
        "a": np.array([1, 2, 3]),
        "b": np.array(["x", "y", "z"]),
    })

    for batch_idx, batch in enumerate(ds.generate(batch_size=2, shuffle=False)):
      if batch_idx == 0:
        test_utils.assert_almost_equal(
            batch, {"a": np.array([1, 2]), "b": np.array(["x", "y"])}
        )
      elif batch_idx == 1:
        test_utils.assert_almost_equal(
            batch, {"a": np.array([3]), "b": np.array(["z"])}
        )
      else:
        assert False

  def test_numpy_generator_shuffle(self):
    ds = numpy_io.NumpyDictBatchedExampleGenerator({
        "a": np.array([1, 2, 3]),
        "b": np.array(["x", "y", "z"]),
    })
    count_per_first_a_value = [0] * 4
    num_runs = 100
    for i in range(100):
      num_sum_a = 0
      num_batches = 0
      for batch_idx, batch in enumerate(
          ds.generate(batch_size=2, shuffle=True, seed=i)
      ):
        num_sum_a += np.sum(batch["a"])
        num_batches += 1
        if batch_idx == 0:
          first_value = batch["a"][0]
          count_per_first_a_value[first_value] += 1
      self.assertEqual(num_batches, 2)
      self.assertEqual(num_sum_a, 1 + 2 + 3)
    for i in range(1, 3):
      self.assertGreater(count_per_first_a_value[i], num_runs / 10)


if __name__ == "__main__":
  absltest.main()
