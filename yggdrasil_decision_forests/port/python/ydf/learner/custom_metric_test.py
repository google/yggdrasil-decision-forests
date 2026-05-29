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

"""Tests for the custom metrics."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.learner import custom_metric


class CustomMetricTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          custom_metric.RegressionMetric,
          abstract_model_pb2.CLASSIFICATION,
      ),
      (
          custom_metric.BinaryClassificationMetric,
          abstract_model_pb2.REGRESSION,
      ),
      (
          custom_metric.MultiClassificationMetric,
          abstract_model_pb2.REGRESSION,
      ),
  )
  def test_incompatible_task(self, metric_cls, task):
    metric = metric_cls(
        name="test_metric",
        evaluation_func=lambda l, p, w: np.float32(0.0),
    )
    with self.assertRaisesRegex(ValueError, "A .* is only compatible with .*"):
      metric.check_is_compatible_task(task)

  def test_compatible_task(self):
    metric = custom_metric.RegressionMetric(
        name="rmse",
        evaluation_func=lambda l, p, w: np.float32(1.0),
    )
    metric.check_is_compatible_task(abstract_model_pb2.REGRESSION)
    self.assertEqual(metric.name, "rmse")


if __name__ == "__main__":
  absltest.main()
