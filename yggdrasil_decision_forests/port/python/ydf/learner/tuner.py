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

import dataclasses
import enum
from typing import Optional, Sequence, Union

from google.protobuf.internal import containers
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.learner.hyperparameters_optimizer import hyperparameters_optimizer_pb2
from yggdrasil_decision_forests.learner.hyperparameters_optimizer.optimizers import random_pb2
from yggdrasil_decision_forests.model import hyperparameter_pb2
from ydf.model import generic_model
from ydf.utils import log

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


@enum.unique
class OptimizeMetric(enum.Enum):
  """Metrics that can be optimized by the hyper-parameter tuner."""

  LOSS = "loss"
  ACCURACY = "accuracy"
  AUC = "auc"
  PR_AUC = "pr-auc"
  RMSE = "rmse"
  MAE = "mae"
  MSE = "mse"
  NDCG_5 = "ndcg@5"
  MRR_10 = "mrr@10"
  QINI = "qini"
  CATE_CALIBRATION = "cate_calibration"


@dataclasses.dataclass
class _MetricConfig:
  protobuf_path: Sequence[str]
  compatible_tasks: Sequence[generic_model.Task]
  requires_binary_classification: bool = False


_METRIC_CONFIGS = {
    OptimizeMetric.LOSS: _MetricConfig(
        ["loss"],
        [
            generic_model.Task.CLASSIFICATION,
            generic_model.Task.REGRESSION,
            generic_model.Task.RANKING,
            generic_model.Task.CATEGORICAL_UPLIFT,
            generic_model.Task.NUMERICAL_UPLIFT,
        ],
    ),
    OptimizeMetric.ACCURACY: _MetricConfig(
        ["classification", "accuracy"], [generic_model.Task.CLASSIFICATION]
    ),
    OptimizeMetric.AUC: _MetricConfig(
        ["classification", "one_vs_other", "auc"],
        [generic_model.Task.CLASSIFICATION],
        requires_binary_classification=True,
    ),
    OptimizeMetric.PR_AUC: _MetricConfig(
        ["classification", "one_vs_other", "pr_auc"],
        [generic_model.Task.CLASSIFICATION],
        requires_binary_classification=True,
    ),
    OptimizeMetric.RMSE: _MetricConfig(
        ["regression", "rmse"], [generic_model.Task.REGRESSION]
    ),
    OptimizeMetric.MAE: _MetricConfig(
        ["regression", "mae"], [generic_model.Task.REGRESSION]
    ),
    OptimizeMetric.MSE: _MetricConfig(
        ["regression", "mse"], [generic_model.Task.REGRESSION]
    ),
    OptimizeMetric.NDCG_5: _MetricConfig(
        ["ranking", "ndcg"], [generic_model.Task.RANKING]
    ),
    OptimizeMetric.MRR_10: _MetricConfig(
        ["ranking", "mrr"], [generic_model.Task.RANKING]
    ),
    OptimizeMetric.QINI: _MetricConfig(
        ["uplift", "qini"],
        [
            generic_model.Task.CATEGORICAL_UPLIFT,
            generic_model.Task.NUMERICAL_UPLIFT,
        ],
    ),
    OptimizeMetric.CATE_CALIBRATION: _MetricConfig(
        ["uplift", "cate_calibration"],
        [
            generic_model.Task.CATEGORICAL_UPLIFT,
            generic_model.Task.NUMERICAL_UPLIFT,
        ],
    ),
}


class AbstractTuner:
  """Base class for tuners."""

  def __init__(
      self,
      optimizer_key: str,
      automatic_search_space: bool = False,
      parallel_trials: int = 1,
      max_trial_duration: Optional[float] = None,
      cross_validation: bool = False,
      cross_validation_num_folds: Optional[int] = None,
      optimize_metric: Optional[str] = None,
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
      cross_validation: Use cross-validation for evaluating the hyperparameter
        candidates. Cross-validation can provide better evaluation quality but
        is much slower.
      cross_validation_num_folds: Number of folds to use for cross-validation.
        Defaults to 10 if not set.
      optimize_metric: Metric to optimize. If not set, the default metric is
        chosen (loss > auc (binary classification only) > accuracy > rmse > ndcg
        > qini). Supported metrics are: loss, accuracy, auc, pr-auc, rmse, mae,
        mse, ndcg@5, mrr@10, qini, cate_calibration (case-insensitive).
    """

    self._automatic_search_space = automatic_search_space
    self._parallel_trials = parallel_trials
    self._max_trial_duration = max_trial_duration
    if cross_validation_num_folds is not None and not cross_validation:
      raise ValueError(
          "cross_validation_num_folds is only available if"
          " cross_validation=True."
      )

    self._train_config = TrainingConfig(learner="HYPERPARAMETER_OPTIMIZER")

    optimizer_config = self._optimizer_config()
    optimizer_config.optimizer.optimizer_key = optimizer_key
    optimizer_config.optimizer.parallel_trials = parallel_trials
    if cross_validation:
      optimizer_config.evaluation.cross_validation.SetInParent()
      if cross_validation_num_folds is not None:
        optimizer_config.evaluation.cross_validation.fold_generator.num_folds = (
            cross_validation_num_folds
        )
    else:
      optimizer_config.evaluation.self_model_evaluation.SetInParent()

    if automatic_search_space:
      optimizer_config.predefined_search_space.SetInParent()

    if max_trial_duration is not None:
      optimizer_config.base_learner.maximum_training_duration_seconds = (
          max_trial_duration
      )

    if optimize_metric is not None:
      try:
        self._optimize_metric = OptimizeMetric(optimize_metric.lower())
      except ValueError:
        supported = ", ".join(v.value for v in OptimizeMetric)
        raise ValueError(  # pylint:disable=raise-missing-from
            f"Unknown metric '{optimize_metric}'. Supported metrics are:"
            f" {supported}"
        )
    else:
      self._optimize_metric = None

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

  def _set_base_learner(self, learner: str) -> None:
    """Sets the base learner key."""
    self._optimizer_config().base_learner.learner = learner

  def _set_task(self, task: generic_model.Task) -> None:
    """Validates and sets the task for the metric optimization."""
    if self._optimize_metric is None:
      return

    config = _METRIC_CONFIGS[self._optimize_metric]
    if (
        config.compatible_tasks is not None
        and task not in config.compatible_tasks
    ):
      raise ValueError(
          f"Metric {self._optimize_metric.name} is not compatible with task"
          f" {task.name}"
      )

    current = getattr(self._optimizer_config().evaluation, "metric")
    for step in config.protobuf_path:
      current = getattr(current, step)
    current.SetInParent()

  def _validate_data_spec(
      self,
      label: str,
      data_spec: data_spec_pb2.DataSpecification,
      raise_error: bool,
  ) -> None:
    """Validates the metric against the data specification."""
    if self._optimize_metric is None:
      return

    config = _METRIC_CONFIGS[self._optimize_metric]
    if config.requires_binary_classification:
      label_col = next(c for c in data_spec.columns if c.name == label)
      if label_col.categorical.number_of_unique_values != 3:
        if raise_error:
          raise ValueError(
              f"Metric {self._optimize_metric.name} is only compatible with"
              " binary classification."
          )
        else:
          log.warning(
              f"Metric {self._optimize_metric.name} is only compatible with"
              " binary classification."
          )

  def _set_base_learner_num_threads(self, num_threads: int) -> None:
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
    cross_validation: Use cross-validation for evaluating the hyperparameter
      candidates. Cross-validation can provide better evaluation quality but is
      much slower.
    cross_validation_num_folds: Number of folds to use for cross-validation.
      Defaults to 10 if not set.
    optimize_metric: Metric to optimize. If not set, the default metric is
      chosen (loss > auc (binary classification only) > accuracy > rmse > ndcg >
      qini). Supported metrics are: loss, accuracy, auc, pr-auc, rmse, mae, mse,
      ndcg@5, mrr@10, qini, cate_calibration (case-insensitive).
  """

  def __init__(
      self,
      num_trials: int = 100,
      automatic_search_space: bool = False,
      parallel_trials: int = 1,
      max_trial_duration: Optional[float] = None,
      *,
      cross_validation: bool = False,
      cross_validation_num_folds: Optional[int] = None,
      optimize_metric: Optional[str] = None,
  ):
    super().__init__(
        optimizer_key="RANDOM",
        automatic_search_space=automatic_search_space,
        parallel_trials=parallel_trials,
        max_trial_duration=max_trial_duration,
        cross_validation=cross_validation,
        cross_validation_num_folds=cross_validation_num_folds,
        optimize_metric=optimize_metric,
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
    cross_validation: Use cross-validation for evaluating the hyperparameter
      candidates. Cross-validation can provide better evaluation quality but is
      much slower.
    cross_validation_num_folds: Number of folds to use for cross-validation.
      Defaults to 5 if not set.
    optimize_metric: Metric to optimize. If not set, the default metric is
      chosen (loss > auc (binary classification only) > accuracy > rmse > ndcg >
      qini). Supported metrics are: loss, accuracy, auc, pr-auc, rmse, mae, mse,
      ndcg@5, mrr@10, qini, cate_calibration (case-insensitive).
  """

  def __init__(
      self,
      num_trials: int = 100,
      automatic_search_space: bool = False,
      parallel_trials: int = 1,
      max_trial_duration: Optional[float] = None,
      *,
      cross_validation: bool = False,
      cross_validation_num_folds: Optional[int] = None,
      optimize_metric: Optional[str] = None,
  ):
    super().__init__(
        optimizer_key="VIZIER",
        automatic_search_space=automatic_search_space,
        parallel_trials=parallel_trials,
        max_trial_duration=max_trial_duration,
        cross_validation=cross_validation,
        cross_validation_num_folds=cross_validation_num_folds,
        optimize_metric=optimize_metric,
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
