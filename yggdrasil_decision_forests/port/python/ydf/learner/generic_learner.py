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

"""Definitions for Generic learners."""

import copy
import os
from typing import Optional, Union

from absl import logging

from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.dataset import weight_pb2
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.metric import metric_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.cc import ydf
from ydf.dataset import dataset
from ydf.learner import hyperparameters
from ydf.learner import tuner as tuner_lib
from ydf.metric import metric
from ydf.model import generic_model
from ydf.model import model_lib
from yggdrasil_decision_forests.utils import fold_generator_pb2

Task = generic_model.Task


class GenericLearner:
  """A generic YDF learner."""

  def __init__(
      self,
      learner_name: str,
      task: Task,
      label: str,
      weights: Optional[str],
      ranking_group: Optional[str],
      uplift_treatment: Optional[str],
      data_spec_args: dataset.DataSpecInferenceArgs,
      data_spec: Optional[data_spec_pb2.DataSpecification],
      hyper_parameters: hyperparameters.HyperParameters,
      deployment_config: abstract_learner_pb2.DeploymentConfig,
      tuner: Optional[tuner_lib.AbstractTuner],
  ):
    # TODO: Refactor to a single hyperparameter dictionary with edit
    # access to these options.
    self._task = task
    self._learner_name = learner_name
    self._label = label
    self._weights = weights
    self._ranking_group = ranking_group
    self._uplift_treatment = uplift_treatment
    self._hyperparameters = hyper_parameters
    self._data_spec = data_spec
    self._data_spec_args = data_spec_args
    self._deployment_config = deployment_config
    self._tuner = tuner

    if not self._label:
      raise ValueError("Constructing the learner requires a non-empty label.")
    if not self._task or task == abstract_model_pb2.Task.UNDEFINED:
      raise ValueError(
          "Constructing the learner requires a task that is not undefined."
      )
    if self._ranking_group is not None and task != Task.RANKING:
      raise ValueError(
          "The ranking group should only be specified for ranking tasks."
          f" Got task={Task.Name(task)}"
      )
    if self._ranking_group is None and task == Task.RANKING:
      raise ValueError("The ranking group must be specified for ranking tasks.")
    if self._uplift_treatment is not None and task not in [
        Task.NUMERICAL_UPLIFT,
        Task.CATEGORICAL_UPLIFT,
    ]:
      raise ValueError(
          "The uplift treatment should only be specified for uplifting tasks."
          f" Got task={Task.Name(task)}"
      )
    if self._uplift_treatment is None and task in [
        Task.NUMERICAL_UPLIFT,
        Task.CATEGORICAL_UPLIFT,
    ]:
      raise ValueError(
          "The uplift treatment must be specified for uplifting tasks."
      )
    if data_spec is not None:
      logging.info(
          "Data spec was provided explicitly, so any other dataspec"
          " configuration options will be ignored."
      )
    if tuner:
      tuner.set_base_learner(learner_name)

  def train(
      self,
      ds: dataset.InputDataset,
  ) -> generic_model.GenericModel:
    """Trains a model on the given dataset."""

    vertical_dataset = self._get_vertical_dataset(ds)
    learner = self._get_learner()
    return model_lib.load_cc_model(learner.Train(vertical_dataset._dataset))  # pylint: disable=protected-access

  def _get_learner(self) -> ydf.GenericCCLearner:
    training_config = abstract_learner_pb2.TrainingConfig(
        learner=self._learner_name,
        label=self._label,
        weight_definition=self._build_weight_definition(),
        ranking_group=self._ranking_group,
        uplift_treatment=self._uplift_treatment,
        task=self._task,
    )
    if self._tuner:
      training_config.MergeFrom(self._tuner.train_config)

    hp_proto = hyperparameters.dict_to_generic_hyperparameter(
        self._hyperparameters
    )
    return ydf.GetLearner(training_config, hp_proto, self._deployment_config)

  def _get_vertical_dataset(
      self, ds: dataset.InputDataset
  ) -> dataset.VerticalDataset:
    if isinstance(ds, dataset.VerticalDataset):
      return ds
      # TODO: Check that the user has not specified a data spec guide.
    else:
      effective_data_spec_args = None
      if self._data_spec is None:
        effective_data_spec_args = self._build_data_spec_args()
      return dataset.create_vertical_dataset_with_spec_or_args(
          ds,
          data_spec=self._data_spec,
          inference_args=effective_data_spec_args,
      )

  def cross_validation(
      self,
      ds: dataset.InputDataset,
      folds: int = 10,
      bootstrapping: Union[bool, int] = False,
      parallel_evaluations: int = 1,
  ) -> metric.Evaluation:
    """Cross-validates the learner and return the evaluation.

    Usage example:

    ```python
    import pandas as pd
    import ydf

    dataset = pd.read_csv("my_dataset.csv")
    learner = ydf.RandomForestLearner(label="label")
    evaluation = learner.cross_validation(dataset)

    # In notebook, display an interractive evaluation
    evaluation

    # Print the evaluation
    print(evaluation)

    # Look at specific metrics
    print(evaluation.accuracy)
    ```

    Args:
      ds: Dataset for the cross-validation.
      folds: Number of cross-validation folds.
      bootstrapping: Controls whether bootstrapping is used to evaluate the
        confidence intervals and statistical tests (i.e., all the metrics ending
        with "[B]"). If set to false, bootstrapping is disabled. If set to true,
        bootstrapping is enabled and 2000 bootstrapping samples are used. If set
        to an integer, it specifies the number of bootstrapping samples to use.
        In this case, if the number is less than 100, an error is raised as
        bootstrapping will not yield useful results.
      parallel_evaluations: Number of model to train and evaluate in parallel
        using multi-threading. Note that each model is potentially already
        trained with multithreading (see `num_threads` argument of Learner
        constructor).

    Returns:
      The cross-validation evaluation.
    """

    fold_generator = fold_generator_pb2.FoldGenerator(
        cross_validation=fold_generator_pb2.FoldGenerator.CrossValidation(
            num_folds=folds,
        )
    )

    if isinstance(bootstrapping, bool):
      bootstrapping_samples = 2000 if bootstrapping else -1
    elif isinstance(bootstrapping, int) and bootstrapping >= 100:
      bootstrapping_samples = bootstrapping
    else:
      raise ValueError(
          "bootstrapping argument should be boolean or an integer greater than"
          " 100 as bootstrapping will not yield useful results. Got"
          f" {bootstrapping!r} instead"
      )
    evaluation_options = metric_pb2.EvaluationOptions(
        bootstrapping_samples=bootstrapping_samples
    )

    deployment_evaluation = abstract_learner_pb2.DeploymentConfig(
        num_threads=parallel_evaluations,
    )

    vertical_dataset = self._get_vertical_dataset(ds)
    learner = self._get_learner()

    evaluation_proto = learner.Evaluate(
        vertical_dataset._dataset,  # pylint: disable=protected-access
        fold_generator,
        evaluation_options,
        deployment_evaluation,
    )
    return metric.Evaluation(evaluation_proto)

  def _build_data_spec_args(self) -> dataset.DataSpecInferenceArgs:
    """Builds DS args with user inputs and guides for labels / special columns.

    Create a copy of self._data_spec_args and adds column definitions for the
    columns label / weights / ranking group / uplift treatment with the correct
    semantic and dataspec inference arguments.

    Returns:
      A copy of the data spec arguments with information about the special
      columns added.

    Raises:
      ValueError: If the label / weights / ranking group / uplift treatment
      column are specified as features.
    """

    def create_label_column(name: str, task: Task) -> dataset.Column:
      if task in [Task.CLASSIFICATION, Task.CATEGORICAL_UPLIFT]:
        return dataset.Column(
            name=name,
            semantic=dataset.Semantic.CATEGORICAL,
            max_vocab_count=-1,
            min_vocab_frequency=1,
        )
      elif task in [Task.REGRESSION, Task.RANKING, Task.NUMERICAL_UPLIFT]:
        return dataset.Column(name=name, semantic=dataset.Semantic.NUMERICAL)
      else:
        raise ValueError(
            f"Unsupported task {abstract_model_pb2.Task(task)} for label column"
        )

    data_spec_args = copy.deepcopy(self._data_spec_args)
    # If no columns have been specified, make sure that all columns are used,
    # since this function will specify some.
    #
    # TODO: If `label` becomes an optional argument, this function needs
    # to be adapted.
    if data_spec_args.columns is None:
      data_spec_args.include_all_columns = True
      data_spec_args.columns = []
    column_defs = data_spec_args.columns
    if dataset.column_defs_contains_column(self._label, column_defs):
      raise ValueError(
          f"Label column {self._label} is also an input feature. A column"
          " cannot be both a label and input feature."
      )
    column_defs.append(create_label_column(self._label, self._task))
    if self._weights is not None:
      if dataset.column_defs_contains_column(self._weights, column_defs):
        raise ValueError(
            f"Weights column {self._weights} is also an input feature. A column"
            " cannot be both a weights and input feature."
        )
      column_defs.append(
          dataset.Column(
              name=self._weights, semantic=dataset.Semantic.NUMERICAL
          )
      )
    if self._ranking_group is not None:
      assert self._task == Task.RANKING

      if dataset.column_defs_contains_column(self._ranking_group, column_defs):
        raise ValueError(
            f"Ranking group column {self._ranking_group} is also an input"
            " feature. A column cannot be both a ranking group and input"
            " feature."
        )
      column_defs.append(
          dataset.Column(
              name=self._ranking_group, semantic=dataset.Semantic.HASH
          )
      )
    if self._uplift_treatment is not None:
      assert self._task in [Task.NUMERICAL_UPLIFT, Task.CATEGORICAL_UPLIFT]

      if dataset.column_defs_contains_column(
          self._uplift_treatment, column_defs
      ):
        raise ValueError(
            "The uplift_treatment column should not be specified as a feature"
        )
      column_defs.append(
          dataset.Column(
              name=self._uplift_treatment,
              semantic=dataset.Semantic.CATEGORICAL,
              max_vocab_count=-1,
              min_vocab_frequency=1,
          )
      )
    return data_spec_args

  def _build_weight_definition(
      self,
  ) -> Optional[weight_pb2.WeightDefinition]:
    weight_definition = None
    if self._weights is not None:
      weight_definition = weight_pb2.WeightDefinition(
          attribute=self._weights,
          numerical=weight_pb2.WeightDefinition.NumericalWeight(),
      )
    return weight_definition

  def _build_deployment_config(
      self,
      num_threads: Optional[int],
      try_resume_training: bool,
      resume_training_snapshot_interval_seconds: int,
      cache_path: Optional[str],
  ):
    if num_threads is None:
      num_threads = self.determine_optimal_num_threads()
    return abstract_learner_pb2.DeploymentConfig(
        num_threads=num_threads,
        try_resume_training=try_resume_training,
        cache_path=cache_path,
        resume_training_snapshot_interval_seconds=resume_training_snapshot_interval_seconds,
    )

  def determine_optimal_num_threads(self):
    """Sets  number of threads to min(num_cpus, 32) or 6 if num_cpus unclear."""
    num_threads = os.cpu_count()
    if num_threads is None:
      logging.warning("Cannot determine the number of CPUs. Set num_threads=6")
      num_threads = 6
    else:
      if num_threads >= 32:
        logging.warning(
            "The `num_threads` constructor argument is not set and the "
            "number of CPU is os.cpu_count()=%d > 32. Setting num_threads "
            "to 32. Set num_threads manually to use more than 32 cpus.",
            num_threads,
        )
        num_threads = 32
      else:
        logging.info("Use %d thread(s) for training", num_threads)
    return num_threads
