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

"""Logs of a feature selection algorithm."""

import dataclasses
from typing import Dict, List, Optional
from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.metric import metric


@dataclasses.dataclass
class Iteration:
  score: float
  features: List[str]
  metrics: Dict[str, float]


@dataclasses.dataclass
class FeatureSelectorLogs:
  iterations: List[Iteration] = dataclasses.field(default_factory=list)
  best_iteration_idx: Optional[int] = None


def proto_to_value(
    proto: abstract_model_pb2.FeatureSelectionLogs,
) -> FeatureSelectorLogs:
  return FeatureSelectorLogs(
      iterations=[
          Iteration(
              score=iteration.score,
              features=iteration.features[:],
              metrics=dict(iteration.metrics),
          )
          for iteration in proto.iterations
      ],
      best_iteration_idx=proto.best_iteration_idx,
  )


def value_to_proto(
    value: FeatureSelectorLogs,
) -> abstract_model_pb2.FeatureSelectionLogs:
  return abstract_model_pb2.FeatureSelectionLogs(
      iterations=[
          abstract_model_pb2.FeatureSelectionLogs.Iteration(
              score=iteration.score,
              features=iteration.features,
              metrics=iteration.metrics,
          )
          for iteration in value.iterations
      ],
      best_iteration_idx=value.best_iteration_idx,
  )
