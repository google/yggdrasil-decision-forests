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
import jax.numpy as jnp
from ydf.deep import layer as layer_lib


FeatureType = layer_lib.FeatureType
Feature = layer_lib.Feature


class StandardFeatureFlattenerTest(parameterized.TestCase):

  def test_base(self):
    m = layer_lib.StandardFeatureFlattener()
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
        {"params": {"embedding_c2": {"embedding": (3, 20)}}},
    )
    y = m.apply(state, x)
    self.assertEqual(y.shape, (2, 24))


if __name__ == "__main__":
  absltest.main()
