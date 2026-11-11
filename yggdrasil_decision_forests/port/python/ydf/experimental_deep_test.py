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

"""Tests that the "experimental_deep" build rule enables `ydf.experimental`."""

from absl.testing import absltest

# In the world, use "import ydf".
import ydf
from ydf.deep import jax_fallback


class ExperimentalDeepTest(absltest.TestCase):

  def test_deep_models_are_available(self):
    for name in [
        "MultiLayerPerceptronLearner",
        "MultiLayerPerceptronModel",
        "TabularTransformerLearner",
        "TabularTransformerModel",
    ]:
      with self.subTest(name):
        self.assertIsNot(
            getattr(ydf.experimental, name), jax_fallback.JaxFallBack
        )


if __name__ == "__main__":
  absltest.main()
