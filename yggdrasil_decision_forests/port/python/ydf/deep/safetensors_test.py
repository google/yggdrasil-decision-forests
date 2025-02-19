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

"""Test of the Safetensors helper lib."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from ydf.deep import safetensors as safetensors_lib


class SafetensorsTest(parameterized.TestCase):

  @parameterized.parameters(
      ({},),
      ({"a": np.array([1, 2])},),
      ({"a": np.array([1, 2]), "b": np.array([2])},),
      ({"a": {"a": np.array([1, 2])}},),
      (
          {
              "a": {"a": np.array([1, 2]), "b": np.array([2])},
              "b": {"a": np.array([1, 2]), "b": {"foobar": np.array([523])}},
              "c": np.array([6, 5, 4]),
          },
      ),
      ({"a": np.array([1, 2])},),
  )
  def test_flatten_and_deflatten(self, weights):
    flattened = safetensors_lib.flatten_weights(weights)
    unflattened = safetensors_lib.deflatten_weights(flattened)
    self.assertEqual(weights, unflattened)

  def test_jax(self):
    example = {"a": {"a": jnp.array([1, 2])}}
    flattened = safetensors_lib.flatten_weights(example)
    expected_flattened = {"a::a": jnp.array([1, 2])}
    self.assertEqual(flattened.keys(), expected_flattened.keys())
    np.testing.assert_array_equal(flattened["a::a"], expected_flattened["a::a"])
    unflattened = safetensors_lib.deflatten_weights(flattened)
    self.assertEqual(unflattened, example)


if __name__ == "__main__":
  absltest.main()
