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

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from ydf.deep import metric as metric_lib


class MetricTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          metric_lib.AccuracyBinaryClassificationMetric(),
          [0, 1, 1],
          [-1.0, -1.0, 1.0],
          2 / 3,
      ),
      (
          metric_lib.LossBinaryClassificationMetric(),
          [0, 1, 1],
          [-1.0, -1.0, 1.0],
          0.6465,
      ),
      (
          metric_lib.MeanSquaredErrorMetric(),
          [0.0, 1.0, 2.0],
          [1.0, 1.0, 2.0],
          1 / 3,
      ),
      (
          metric_lib.AccuracyMultiClassClassificationMetric(3),
          [0, 2],
          [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
          1 / 2,
      ),
      (
          metric_lib.LossMultiClassClassificationMetric(3),
          [0, 2],
          [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
          1.03972,
      ),
  )
  def test_base(
      self,
      metric: metric_lib.Metric,
      labels: Any,
      predictions: Any,
      expected: float,
  ):
    metric_value = metric(jnp.array(labels), jnp.array(predictions)).item()
    self.assertAlmostEqual(metric_value, expected, delta=0.001)


if __name__ == "__main__":
  absltest.main()
