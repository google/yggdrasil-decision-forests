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

"""Tests the model inference benchmark."""

import sys

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from ydf.model import benchmark


class ToJaxTest(parameterized.TestCase):

  def test_benchmark(self):
    benchmark.run_preconfigured(
        models=["reg gbt; num feat"]
    )

  def test_get_num_examples(self):
    self.assertEqual(
        benchmark.get_num_examples(
            {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
        ),
        3,
    )

  @parameterized.parameters(
      (
          {"numerical_feature": True, "categorical_feature": True},
          {"label", "n1", "n2", "c1", "c2"},
      ),
      (
          {
              "numerical_feature": True,
          },
          {"label", "n1", "n2"},
      ),
  )
  def test_build_synthetic_dataset(self, kwargs, expected_columns):
    dataset = benchmark.build_synthetic_dataset(
        10, seed=1, label_type="regression", **kwargs
    )
    self.assertEqual(benchmark.get_num_examples(dataset), 10)
    self.assertSetEqual(set(dataset), expected_columns)

  def test_gen_batch(self):
    ds = {"a": np.array([1, 2, 3, 4, 5]), "b": np.array([6, 7, 8, 9, 10])}
    num_iters = 0
    for iter_idx, batch in enumerate(benchmark.gen_batch(ds, 2)):
      self.assertEqual(
          benchmark.get_num_examples(batch), 1 if iter_idx == 2 else 2
      )
      num_iters += 1
    self.assertEqual(num_iters, 3)


if __name__ == "__main__":
  if sys.version_info < (3, 9):
    print("JAX is not supported anymore on python <= 3.8. Skipping JAX tests.")
  else:
    absltest.main()
