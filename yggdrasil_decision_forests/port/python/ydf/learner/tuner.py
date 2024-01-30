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

"""A tuner is a utility to configure of the C++ hyperparameter optimizer v2.

Usage example:

```
import ydf

# Create a tuner
tuner = ydf.RandomSearchTuner(num_trials=20)
tuner.choice("num_candidate_attributes_ratio", [1.0, 0.8, 0.6])
tuner.choice("use_hessian_gain", [True, False])

local_search_space = tuner.choice("growing_strategy", ["LOCAL"])
local_search_space.choice("max_depth", [4, 5, 6, 7])

global_search_space = tuner.choice(
    "growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True)
global_search_space.choice("max_num_nodes", [16, 32, 64, 128])

# Configure a learner
learner = ydf.RandomForestLearner(tuner=tuner)
```
"""

from typing import Optional, Sequence, Union

from google.protobuf.internal import containers
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.learner.hyperparameters_optimizer import hyperparameters_optimizer_pb2
from yggdrasil_decision_forests.learner.hyperparameters_optimizer.optimizers import random_pb2
from yggdrasil_decision_forests.model import hyperparameter_pb2


# Single hyperparameter value
HyperParameterValue = Union[int, float, str, bool]

# List of values for a given hyperparameter
# Note: Hyperparameter types cannot be mixed.
HyperParameterSequence = Union[
    Sequence[int], Sequence[float], Sequence[str], Sequence[bool]
]

# Short aliases
TrainingConfig = abstract_learner_pb2.TrainingConfig
OptimizerConfig = (
    hyperparameters_optimizer_pb2.HyperParametersOptimizerLearnerTrainingConfig
)
Field = hyperparameter_pb2.HyperParameterSpace.Field
Value = hyperparameter_pb2.GenericHyperParameters.Value


class AbstractTuner:
  """Base class for tuners."""

  def __init__(
      self,
      optimizer_key: str,
      automatic_search_space: bool = False,
      parallel_trials: int = 1,
      max_trial_duration: Optional[float] = None,
  ):
    """Initializes tuner.

    Args:
      optimizer_key: Registered identifier of the optimizer.
      automatic_search_space: If true, automatically define the search space of
        hyperparameters. In this case, configuring the hyperparameters manually
        (e.g. calling "choice(...)" on the tuner) is not necessary.
      parallel_trials: Number of trials to evaluate in parallel. The training of
        an individual model uses "num_threads" threads (configured in the
        learner). Therefore, in the non-distributed training setting, the total
        number of threads will be `parallel_trials x num_threads`. In the
        distributed training setting, the average number of user threads per
        worker will be `parallel_trials x num_threads // num_workers`. In this
        case, make sure `parallel_trials` is a multiple of the number of
        workers.
      max_trial_duration: Maximum training duration of an individual trial
        expressed in seconds. This parameter is different from the
        `maximum_training_duration_seconds` learner parameter that defines the
        maximum total training and tuning duration. Set to None for no time
        limit.
    """

    self._automatic_search_space = automatic_search_space
    self._parallel_trials = parallel_trials
    self._max_trial_duration = max_trial_duration

    self._train_config = TrainingConfig(learner="HYPERPARAMETER_OPTIMIZER")

    optimizer_config = self._optimizer_config()
    optimizer_config.optimizer.optimizer_key = optimizer_key
    optimizer_config.optimizer.parallel_trials = parallel_trials

    if automatic_search_space:
      optimizer_config.predefined_search_space.SetInParent()

    if max_trial_duration is not None:
      optimizer_config.base_learner.maximum_training_duration_seconds = (
          max_trial_duration
      )

  @property
  def parallel_trials(self) -> int:
    return self._parallel_trials

  @property
  def train_config(self) -> TrainingConfig:
    """Gets the training configuration proto."""
    return self._train_config

  def _optimizer_config(self) -> OptimizerConfig:
    """Gets the optimizer configuration proto."""
    return self._train_config.Extensions[
        hyperparameters_optimizer_pb2.hyperparameters_optimizer_config
    ]

  def set_base_learner(self, learner: str) -> None:
    """Sets the base learner key."""
    self._optimizer_config().base_learner.learner = learner

  def set_base_learner_num_threads(self, num_threads: int) -> None:
    """Sets the number of threads in the base learner."""
    self._optimizer_config().base_learner_deployment.num_threads = num_threads

  def choice(
      self,
      key: str,
      values: HyperParameterSequence,
      merge: bool = False,
  ) -> "SearchSpace":
    """Adds a hyperparameter with a list of possible values.

    Args:
      key: Name of the hyperparameter.
      values: List of possible values.
      merge: If false (default), raises an error if the hyperparameter already
        exist. If true, and if the hyperparameter already exist, adds "values"
        to the already configured values. If true, and if the hyperparameter
        does not already exist, raises an error.

    Returns:
      The conditional SearchSpace corresponding to the values in "values".
    """

    sp = SearchSpace(self._optimizer_config().search_space.fields)
    return sp.choice(key, values, merge)


class RandomSearchTuner(AbstractTuner):
  """Tuner using random search.

  The candidate hyper-parameter can be evaluated independently and in parallel.

  Attributes:
    num_trials: Number of hyperparameter configurations to evaluate.
    automatic_search_space: If true, automatically define the search space of
      hyperparameters. In this case, configuring the hyperparameters manually
      (e.g. calling "choice(...)" on the tuner) is not necessary.
    parallel_trials: Number of trials to evaluate in parallel. The training of
      an individual model uses "num_threads" threads (configured in the
      learner). Therefore, in the non-distributed training setting, the total
      number of threads will be `parallel_trials x num_threads`. In the
      distributed training setting, the average number of user threads per
      worker will be `parallel_trials x num_threads // num_workers`. In this
      case, make sure `parallel_trials` is a multiple of the number of workers.
    max_trial_duration: Maximum training duration of an individual trial
      expressed in seconds. This parameter is different from the
      `maximum_training_duration_seconds` learner parameter that define the
      maximum total training and tuning duration. Set to None for no time limit.
  """

  def __init__(
      self,
      num_trials: int = 100,
      automatic_search_space: bool = False,
      parallel_trials: int = 1,
      max_trial_duration: Optional[float] = None,
  ):
    super().__init__(
        optimizer_key="RANDOM",
        automatic_search_space=automatic_search_space,
        parallel_trials=parallel_trials,
        max_trial_duration=max_trial_duration,
    )
    self._random_optimizer_config().num_trials = num_trials

  def _random_optimizer_config(self) -> random_pb2.RandomOptimizerConfig:
    return self._optimizer_config().optimizer.Extensions[random_pb2.random]


class VizierTuner(AbstractTuner):
  """Tuner using Vizier.

  Attributes:
    num_trials: Number of hyperparameter configurations to evaluate.
    automatic_search_space: If true, automatically define the search space of
      hyperparameters. In this case, configuring the hyperparameters manually
      (e.g. calling "choice(...)" on the tuner) is not necessary.
    parallel_trials: Number of trials to evaluate in parallel. The training of
      an individual model uses "num_threads" threads (configured in the
      learner). Therefore, in the non-distributed training setting, the total
      number of threads will be `parallel_trials x num_threads`. In the
      distributed training setting, the average number of user threads per
      worker will be `parallel_trials x num_threads // num_workers`. In this
      case, make sure `parallel_trials` is a multiple of the number of workers.
    max_trial_duration: Maximum training duration of an individual trial
      expressed in seconds. This parameter is different from the
      `maximum_training_duration_seconds` learner parameter that define the
      maximum total training and tuning duration. Set to None for no time limit.
  """

  def __init__(
      self,
      num_trials: int = 100,
      automatic_search_space: bool = False,
      parallel_trials: int = 1,
      max_trial_duration: Optional[float] = None,
  ):
    super().__init__(
        optimizer_key="VIZIER",
        automatic_search_space=automatic_search_space,
        parallel_trials=parallel_trials,
        max_trial_duration=max_trial_duration,
    )


class SearchSpace:
  """Set of hyper-parameter values to explore.

  For the users: Don't create this object directly. Instead, use a tuner e.g.
  `tuner.choice(...)`.
  """

  def __init__(
      self,
      fields: "containers.RepeatedCompositeFieldContainer[Field]",
      parent_values: Optional[
          hyperparameter_pb2.HyperParameterSpace.DiscreteCandidates
      ] = None,
  ):
    """Initializes the search space.

    Args:
      fields: Fields to populate in a hyperparameter proto.
      parent_values: Conditionnal parent value.
    """

    self._fields = fields
    self._parent_values = parent_values

  def choice(
      self,
      key: str,
      values: HyperParameterSequence,
      merge: bool = False,
  ) -> "SearchSpace":
    """Adds a hyperparameter with a list of possible values.

    Args:
      key: Name of the hyperparameter.
      values: List of possible values.
      merge: If false (default), raises an error if the hyperparameter already
        exist. If true, and if the hyperparameter already exist, adds "values"
        to the already configured values. If true, and if the hyperparameter
        does not already exist, raises an error.

    Returns:
      The conditional SearchSpace corresponding to the values in "values".
    """

    if not values:
      raise ValueError("The list of values is empty")

    field = self._find_field(key)
    if field is None:
      # New hyperparameter
      if merge:
        raise ValueError(
            f"Using `merge=true` but the hyperparameter {key!r} does not"
            " already exist"
        )
      field = self._fields.add(name=key)
      if self._parent_values:
        field.parent_discrete_values.MergeFrom(self._parent_values)
    else:
      # Existing hyperparameter
      if not merge:
        raise ValueError(
            f"The hyperparameter {key!r} already exist. Use `merge=True` to add"
            " values to an existing hyperparameter"
        )

    # Register new values
    possible_values = [_py_value_to_hp_value(key, value) for value in values]
    dst_values = hyperparameter_pb2.HyperParameterSpace.DiscreteCandidates(
        possible_values=possible_values[:]
    )
    field.discrete_candidates.possible_values.extend(possible_values)
    return SearchSpace(field.children, parent_values=dst_values)

  def _find_field(self, key: str) -> Optional[Field]:
    """Returns the hyperparameter with a given name.

    If the hyperparameter does not exist, return None.

    Args:
      key: Hyperparameter name.
    """

    for field in self._fields:
      if field.name == key:
        return field
    return None


def _py_value_to_hp_value(key: str, value: HyperParameterValue) -> Value:
  """Converts a user input / python primitive into a Value proto."""

  if isinstance(value, bool):
    return Value(categorical="true" if value else "false")
  elif isinstance(value, int):
    return Value(integer=value)
  elif isinstance(value, float):
    return Value(real=value)
  elif isinstance(value, str):
    return Value(categorical=value)
  else:
    raise ValueError(
        f"Not supported value {value!r} ({type(value)}) for {key!r}"
    )
