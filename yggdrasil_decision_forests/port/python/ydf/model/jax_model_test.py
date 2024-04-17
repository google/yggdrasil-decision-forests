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
import jax.numpy as jnp
import numpy as np
from ydf.model import export_jax as to_jax


class JaxModelTest(parameterized.TestCase):

  @parameterized.parameters(
      ((0,), jnp.int8),
      ((0, 1, -1), jnp.int8),
      ((0, 1, 0x7F, -0x80), jnp.int8),
      ((0, 1, 0x7F + 1), jnp.int16),
      ((0, 1, -0x80 - 1), jnp.int16),
      ((0, 1, 0x7FFF), jnp.int16),
      ((0, 1, -0x8000), jnp.int16),
      ((0, 1, 0x7FFF + 1), jnp.int32),
      ((0, 1, -0x8000 - 1), jnp.int32),
      ((0, 1, 0x7FFFFFFF), jnp.int32),
      ((0, 1, -0x80000000), jnp.int32),
  )
  def test_compact_dtype(self, values, expected_dtype):
    self.assertEqual(to_jax.compact_dtype(values), expected_dtype)

    jax_array = to_jax.to_compact_jax_array(values)
    self.assertEqual(jax_array.dtype.type, expected_dtype)
    np.testing.assert_array_equal(jax_array, jnp.array(values, expected_dtype))

  def test_compact_dtype_non_supported(self):
    with self.assertRaisesRegex(ValueError, "No supported compact dtype"):
      to_jax.compact_dtype((0x80000000,))


if __name__ == "__main__":
  absltest.main()
