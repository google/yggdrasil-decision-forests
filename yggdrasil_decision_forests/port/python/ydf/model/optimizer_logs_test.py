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
from yggdrasil_decision_forests.model import abstract_model_pb2
from yggdrasil_decision_forests.model import hyperparameter_pb2
from ydf.model import optimizer_logs

Value = hyperparameter_pb2.GenericHyperParameters.Value
HyperparametersOptimizerLogs = abstract_model_pb2.HyperparametersOptimizerLogs
Step = abstract_model_pb2.HyperparametersOptimizerLogs.Step
Field = hyperparameter_pb2.GenericHyperParameters.Field


class OptimizerLogsTest(parameterized.TestCase):

  @parameterized.parameters(
      (Value(categorical="hello"), "hello"),
      (Value(integer=1), 1),
      (Value(real=1), 1.0),
      (
          Value(categorical_list=Value.CategoricalList(values=["a", "b"])),
          ["a", "b"],
      ),
  )
  def test_valid_value(self, proto, expected_value):
    self.assertEqual(optimizer_logs.value_from_proto(proto), expected_value)

  def test_convert_valid_hyperparameters(self):
    proto = HyperparametersOptimizerLogs(
        steps=[
            Step(
                score=1,
                hyperparameters=hyperparameter_pb2.GenericHyperParameters(
                    fields=[
                        Field(
                            name="a",
                            value=Value(categorical="x"),
                        ),
                        Field(
                            name="b",
                            value=Value(
                                integer=5,
                            ),
                        ),
                    ]
                ),
            ),
            Step(score=2),
        ]
    )
    self.assertEqual(
        optimizer_logs.proto_optimizer_logs_to_optimizer_logs(proto),
        optimizer_logs.OptimizerLogs(
            trials=[
                optimizer_logs.Trial(score=1, params={"a": "x", "b": 5}),
                optimizer_logs.Trial(score=2, params={}),
            ]
        ),
    )


if __name__ == "__main__":
  absltest.main()
