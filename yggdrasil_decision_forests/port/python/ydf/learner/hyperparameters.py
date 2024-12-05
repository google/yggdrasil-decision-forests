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

from collections.abc import Mapping
import dataclasses
from typing import Dict, Optional, Set, Union

from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.model import hyperparameter_pb2
from ydf.cc import ydf
from ydf.learner import custom_loss


HyperParameters = Dict[
    str, Optional[Union[int, float, str, bool, custom_loss.AbstractCustomLoss]]
]


def dict_to_generic_hyperparameter(
    src: HyperParameters,
) -> hyperparameter_pb2.GenericHyperParameters:
  """Transform a dictionary of hyperparameters to the corresponding proto."""
  generic_hps = hyperparameter_pb2.GenericHyperParameters()
  for key, value in src.items():
    if value is None:
      # The value is not defined, use default.
      continue
    if key == "loss" and isinstance(value, custom_loss.AbstractCustomLoss):
      # Custom Python fields must be treated separately.
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


def validate_hyperparameters(
    hp_dict: HyperParameters,
    train_config: abstract_learner_pb2.TrainingConfig,
    deployment_config: abstract_learner_pb2.DeploymentConfig,
):
  """Returns None if the hyperparameters are valid, raises otherwise."""
  not_none_hps = set(key for key, value in hp_dict.items() if value is not None)
  return ydf.ValidateHyperparameters(
      not_none_hps, train_config, deployment_config
  )


def fix_hyperparameters(
    hp_dict: HyperParameters,
    explicit_parameters: Set[str],
    train_config: abstract_learner_pb2.TrainingConfig,
    deployment_config: abstract_learner_pb2.DeploymentConfig,
) -> HyperParameters:
  """Returns exclusion-free hyperparameters."""
  not_none_hp_names = set(
      key for key, value in hp_dict.items() if value is not None
  )
  explicit_hp_names = explicit_parameters.intersection(not_none_hp_names)
  invalid_hp_names = ydf.GetInvalidHyperparameters(
      not_none_hp_names, explicit_hp_names, train_config, deployment_config
  )
  return {
      key: value
      for key, value in hp_dict.items()
      if key not in invalid_hp_names
  }


@dataclasses.dataclass
class HyperparameterTemplate(Mapping):
  """A named and versioned set of hyper-parameters.

  List of hyper-parameter sets that outperforms the default hyper-parameters
  (either generally or in specific scenarios). A template is also a mapping of
  hyperparameters and may be used with the double star operator.

  Usage example:

  ```python
  templates = ydf.GradientBoostedTreesLearner.hyperparameter_templates()
  better_default = templates["better_defaultv1"]
  # Apply the parameters of the template on the learner.
  learner = ydf.GradientBoostedTreesLearner(label, **better_default)
  ```
  """

  name: str
  version: int
  parameters: HyperParameters
  description: str

  def __iter__(self):
    for key in self.parameters.keys():
      yield key

  def __len__(self):
    return len(self.parameters)

  def __getitem__(self, item):
    if isinstance(self.parameters, dict) and item in self.parameters:
      return self.parameters[item]
    return None
