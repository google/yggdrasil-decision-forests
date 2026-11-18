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


def _zero_metric(labels, predictions, weights):
  del labels, predictions, weights
  return np.float32(0.0)


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
        evaluation_func=_zero_metric,
    )
    with self.assertRaisesRegex(ValueError, "A .* is only compatible with .*"):
      metric.check_is_compatible_task(task)

  def test_compatible_task(self):
    metric = custom_metric.RegressionMetric(
        name="rmse",
        evaluation_func=_zero_metric,
    )
    metric.check_is_compatible_task(abstract_model_pb2.REGRESSION)
    self.assertEqual(metric.name, "rmse")

  @parameterized.named_parameters(
      ("regression", custom_metric.RegressionMetric),
      ("binary_classification", custom_metric.BinaryClassificationMetric),
      ("multi_classification", custom_metric.MultiClassificationMetric),
  )
  def test_higher_is_better_defaults_and_overrides(self, metric_cls):
    default_metric = metric_cls("m", _zero_metric)
    self.assertFalse(default_metric.higher_is_better)

    maximized_metric = metric_cls("m", _zero_metric, higher_is_better=True)
    self.assertTrue(maximized_metric.higher_is_better)


if __name__ == "__main__":
  absltest.main()
