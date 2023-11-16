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

"""Python object wrapper around optimizer logs."""

import dataclasses
from typing import Dict, Optional, Sequence, Union
from yggdrasil_decision_forests.model import abstract_model_pb2
from yggdrasil_decision_forests.model import hyperparameter_pb2

# An hyper-parameter value
HyperParameterValue = Union[str, int, float, Sequence[str]]


@dataclasses.dataclass(frozen=True)
class Trial:
  """Results of a single trial.

  Attributes:
    score: Score of the trial. The semantic depends on the optimizer and the
      learner.
    params: Hyper-parameters of the learner.
  """

  score: Optional[float]
  params: Dict[str, HyperParameterValue]


@dataclasses.dataclass(frozen=True)
class OptimizerLogs:
  """Logs of an optimizer run.

  Attributes:
    trials: Collection of trials.
  """

  trials: Sequence[Trial]


def proto_optimizer_logs_to_optimizer_logs(
    proto: abstract_model_pb2.HyperparametersOptimizerLogs,
) -> OptimizerLogs:
  """Converts proto optimizer logs into user-facing optimizer logs."""

  return OptimizerLogs(trials=[_trial_from_proto(step) for step in proto.steps])


def _trial_from_proto(
    step: abstract_model_pb2.HyperparametersOptimizerLogs.Step,
) -> Trial:
  return Trial(
      score=step.score,
      params={
          field.name: value_from_proto(field.value)
          for field in step.hyperparameters.fields
      },
  )


def value_from_proto(
    value: hyperparameter_pb2.GenericHyperParameters.Value,
) -> HyperParameterValue:
  """Converts a proto value into a Python object value."""

  if value.HasField("categorical"):
    return value.categorical
  elif value.HasField("integer"):
    return value.integer
  elif value.HasField("real"):
    return value.real
  elif value.HasField("categorical_list"):
    return value.categorical_list.values
  else:
    raise ValueError(f"Unsupported value {value}")
