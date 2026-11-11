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

"""Tests that the YDF API works without its optional dependencies.

JAX (and the JAX based deep models) and scikit-learn are optional dependencies
of YDF. YDF should be usable (e.g., load a model and generate predictions)
without them, and the features requiring them should raise a helpful error.
"""

import importlib.util
import os

from absl import flags
from absl.testing import absltest
import numpy as np

# In the world, use "import ydf".
import ydf


# Note: This test does not use the "test_utils" library as it depends on JAX and
# scikit-learn.
def _ydf_test_data_path() -> str:
  return os.path.join(
      "external/yggdrasil_decision_forests~",
      "yggdrasil_decision_forests/test_data",
  )
def _is_available(module: str) -> bool:
  return importlib.util.find_spec(module) is not None


def _model_path() -> str:
  return os.path.join(_ydf_test_data_path(), "model", "adult_binary_class_gbdt")


class ApiWithoutOptionalDepsTest(absltest.TestCase):

  def test_load_model_and_predict(self):
    model = ydf.load_model(_model_path())
    dataset_path = os.path.join(
        _ydf_test_data_path(), "dataset", "adult_test.csv"
    )
    predictions = model.predict("csv:" + dataset_path)
    self.assertEqual(predictions.ndim, 1)
    self.assertTrue(np.all((predictions >= 0) & (predictions <= 1)))

  def test_to_jax_function_without_jax(self):
    if _is_available("jax"):
      self.skipTest("JAX is available")
    model = ydf.load_model(_model_path())
    with self.assertRaisesRegex(ValueError, "jax"):
      model.to_jax_function()

  def test_from_sklearn_without_sklearn(self):
    if _is_available("sklearn"):
      self.skipTest("scikit-learn is available")
    with self.assertRaisesRegex(ValueError, "scikit-learn"):
      ydf.from_sklearn(None)

  def test_experimental_deep_models_without_jax(self):
    if _is_available("jax"):
      self.skipTest("JAX is available")
    with self.assertRaisesRegex(ValueError, "jax"):
      ydf.experimental.MultiLayerPerceptronLearner(label="label")


if __name__ == "__main__":
  absltest.main()
