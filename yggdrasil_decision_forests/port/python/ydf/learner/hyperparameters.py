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

"""Utility functions for YDF Hyperparameters."""
from typing import Dict, Union

from yggdrasil_decision_forests.model import hyperparameter_pb2

HyperParameters = Dict[str, Union[int, float, str, bool]]


def dict_to_generic_hyperparameter(
    src: HyperParameters,
) -> hyperparameter_pb2.GenericHyperParameters:
  """Transform a dictionary of hyperparameters to the corresponding proto."""
  generic_hps = hyperparameter_pb2.GenericHyperParameters()
  for key, value in src.items():
    if value is None:
      # The value is not defined, use default.
      continue
    # Boolean has to come first, since it is a subtype of int.
    if isinstance(value, bool):
      # The GenericHyperParameters proto demands this conversion.
      value_as_str = "true" if value else "false"
      generic_hps.fields.append(
          hyperparameter_pb2.GenericHyperParameters.Field(
              name=key,
              value=hyperparameter_pb2.GenericHyperParameters.Value(
                  categorical=value_as_str
              ),
          )
      )
    elif isinstance(value, int):
      generic_hps.fields.append(
          hyperparameter_pb2.GenericHyperParameters.Field(
              name=key,
              value=hyperparameter_pb2.GenericHyperParameters.Value(
                  integer=value
              ),
          )
      )
    elif isinstance(value, float):
      generic_hps.fields.append(
          hyperparameter_pb2.GenericHyperParameters.Field(
              name=key,
              value=hyperparameter_pb2.GenericHyperParameters.Value(real=value),
          )
      )
    elif isinstance(value, str):
      generic_hps.fields.append(
          hyperparameter_pb2.GenericHyperParameters.Field(
              name=key,
              value=hyperparameter_pb2.GenericHyperParameters.Value(
                  categorical=value
              ),
          )
      )
    else:
      raise ValueError(f"Invalid value {value} for parameter {key}")
  return generic_hps
