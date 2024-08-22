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
import datetime
import os
import re
from typing import Optional, Sequence, Set, Union

from absl import logging

from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.dataset import weight_pb2
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.metric import metric_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.cc import ydf
from ydf.dataset import dataset
from ydf.dataset import dataspec
from ydf.learner import custom_loss
from ydf.learner import hyperparameters as hp_lib
from ydf.learner import tuner as tuner_lib
from ydf.metric import metric
from ydf.model import generic_model
from ydf.model import model_lib
from ydf.utils import log
from yggdrasil_decision_forests.utils import fold_generator_pb2
from yggdrasil_decision_forests.utils.distribute.implementations.grpc import grpc_pb2

Task = generic_model.Task

_FRAMEWORK_NAME = "Python YDF"


class GenericLearner:
  """A generic YDF learner."""

  def __init__(
      self,
      learner_name: str,
      task: Task,
      label: Optional[str],
      weights: Optional[str],
      ranking_group: Optional[str],
      uplift_treatment: Optional[str],
      data_spec_args: dataspec.DataSpecInferenceArgs,
      data_spec: Optional[data_spec_pb2.DataSpecification],
      hyper_parameters: hp_lib.HyperParameters,
      explicit_learner_arguments: Optional[Set[str]],
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

    if self._label is not None and not isinstance(label, str):
      raise ValueError("The 'label' should be a string")
    if task != Task.ANOMALY_DETECTION and not self._label:
      raise ValueError("This learner requires a label.")

    if self._ranking_group is not None and task != Task.RANKING:
      raise ValueError(
          "The ranking group should only be specified for ranking tasks."
          f" Got task={task.name}"
      )
    if self._ranking_group is None and task == Task.RANKING:
      raise ValueError("The ranking group must be specified for ranking tasks.")
    if self._uplift_treatment is not None and task not in [
        Task.NUMERICAL_UPLIFT,
        Task.CATEGORICAL_UPLIFT,
    ]:
      raise ValueError(
          "The uplift treatment should only be specified for uplifting tasks."
          f" Got task={task.name}"
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

    if explicit_learner_arguments is not None:
      self._hyperparameters = self._clean_up_hyperparameters(
          explicit_learner_arguments
      )

    self.validate_hyperparameters()

  @property
  def hyperparameters(self) -> hp_lib.HyperParameters:
    """A (mutable) dictionary of this learner's hyperparameters.

    This object can be used to inspect or modify hyperparameters after creating
    the learner. Modifying hyperparameters after constructing the learner is
    suitable for some advanced use cases. Since this approach bypasses some
    feasibility checks for the given set of hyperparameters, it generally better
    to re-create the learner for each model. The current set of hyperparameters
    can be validated manually with `validate_hyperparameters()`.
    """
    return self._hyperparameters

  def validate_hyperparameters(self):
    """Returns None if the hyperparameters are valid, raises otherwise.

    This method is called automatically before training, but users may call it
    to fail early. It makes sense to call this method when changing manually the
    hyper-paramters of the learner. This is a relatively advanced approach that
    is not recommende (it is better to re-create the learner in most cases).

    Usage example:

    ```
    import ydf
    import pandas as pd

    train_ds = pd.read_csv(...)

    learner = ydf.GradientBoostedTreesLearner(label="label")
    learner.hyperparameters["max_depth"] = 20
    learner.validate_hyperparameters()
    model = learner.train(train_ds)
    evaluation = model.evaluate(test_ds)
    ```
    """
    return hp_lib.validate_hyperparameters(
        self._hyperparameters,
        self._get_training_config(),
        self._deployment_config,
    )

  def _clean_up_hyperparameters(
      self, explicit_parameters: Set[str]
  ) -> hp_lib.HyperParameters:
    """Returns the hyperparameters purged from the mutually exlusive ones."""
    return hp_lib.fix_hyperparameters(
        self._hyperparameters,
        explicit_parameters,
        self._get_training_config(),
        self._deployment_config,
    )

  def train(
      self,
      ds: dataset.InputDataset,
      valid: Optional[dataset.InputDataset] = None,
      verbose: Optional[Union[int, bool]] = None,
  ) -> generic_model.ModelType:
    """Trains a model on the given dataset.

    Options for dataset reading are given on the learner. Consult the
    documentation of the learner or ydf.create_vertical_dataset() for additional
    information on dataset reading in YDF.

    Usage example:

    ```
    import ydf
    import pandas as pd

    train_ds = pd.read_csv(...)
    test_ds = pd.read_csv(...)

    learner = ydf.GradientBoostedTreesLearner(label="label")
    model = learner.train(train_ds)
    evaluation = model.evaluate(test_ds)
    ```

    Usage example with a validation dataset:

    ```
    import ydf
    import pandas as pd

    train_ds = pd.read_csv(...)
    valid_ds = pd.read_csv(...)
    test_ds = pd.read_csv(...)

    learner = ydf.GradientBoostedTreesLearner(label="label")
    model = learner.train(train_ds, valid=valid_ds)
    evaluation = model.evaluate(test_ds)
    ```

    If training is interrupted (for example, by interrupting the cell execution
    in Colab), the model will be returned to the state it was in at the moment
    of interruption.

    Args:
      ds: Training dataset.
      valid: Optional validation dataset. Some learners, such as Random Forest,
        do not need validation dataset. Some learners, such as
        GradientBoostedTrees, automatically extract a validation dataset from
        the training dataset if the validation dataset is not provided.
      verbose: Verbose level during training. If None, uses the global verbose
        level of `ydf.verbose`. Levels are: 0 of False: No logs, 1 or True:
        Print a few logs in a notebook; prints all the logs in a terminal. 2:
        Prints all the logs on all surfaces.

    Returns:
      A trained model.
    """

    if valid is not None:
      if not self.__class__.capabilities().support_validation_dataset:
        raise ValueError(
            f"The learner {self.__class__.__name__!r} does not use a"
            " validation dataset. If you can, add the validation examples to"
            " the training dataset."
        )

    if isinstance(ds, str):
      if valid is not None and not isinstance(valid, str):
        raise ValueError(
            "If the training dataset is a path, the validation dataset must"
            " also be a path."
        )
      return self._train_from_path(ds, valid)
    if valid is not None and isinstance(valid, str):
      raise ValueError(
          "The validation dataset may only be a path if the training dataset is"
          " a path."
      )

    self.validate_hyperparameters()

    saved_verbose = log.verbose(verbose) if verbose is not None else None
    try:
      model = self._train_from_dataset(ds, valid)
    finally:
      if saved_verbose is not None:
        log.verbose(saved_verbose)

    return model

  def __str__(self) -> str:
    return f"""\
Learner: {self._learner_name}
Task: {self._task}
Class: ydf.{self.__class__.__name__}
Hyper-parameters: ydf.{self._hyperparameters}
"""

  def _train_from_path(
      self, ds: str, valid: Optional[str]
  ) -> generic_model.ModelType:
    """Trains a model from a file path (dataset reading in YDF C++)."""
    with log.cc_log_context():
      if self._data_spec is not None:
        cc_model = self._get_learner().TrainFromPathWithDataSpec(
            ds, self._data_spec, valid
        )
      else:
        guide = self._build_data_spec_args().to_proto_guide()
        cc_model = self._get_learner().TrainFromPathWithGuide(ds, guide, valid)
      model = model_lib.load_cc_model(cc_model)

      # Note: We don't know the number of training examples before the training
      # call.
      log.maybe_warning_large_dataset(
          model.data_spec().created_num_rows,
          distributed=True,
          discretize_numerical_columns=self._data_spec_args.discretize_numerical_columns,
      )
      return model

  def _train_from_dataset(
      self,
      ds: dataset.InputDataset,
      valid: Optional[dataset.InputDataset] = None,
  ) -> generic_model.ModelType:
    """Trains a model from in-memory data."""

    with log.cc_log_context():
      train_ds = self._get_vertical_dataset(ds)._dataset  # pylint: disable=protected-access

      train_args = {"dataset": train_ds}

      if valid is not None:
        valid_ds = self._get_vertical_dataset(valid)._dataset  # pylint: disable=protected-access
        train_args["validation_dataset"] = valid_ds
        log.info(
            "Train model on %d training examples and %d validation examples",
            train_ds.nrow(),
            valid_ds.nrow(),
        )
      else:
        log.info(
            "Train model on %d examples",
            train_ds.nrow(),
        )

      log.maybe_warning_large_dataset(
          train_ds.nrow(),
          distributed=False,
          discretize_numerical_columns=self._data_spec_args.discretize_numerical_columns,
      )

      time_begin_training_model = datetime.datetime.now()
      learner = self._get_learner()
      cc_model = learner.Train(**train_args)
      log.info(
          "Model trained in %s",
          datetime.datetime.now() - time_begin_training_model,
      )

      return model_lib.load_cc_model(cc_model)

  def _get_training_config(self) -> abstract_learner_pb2.TrainingConfig:
    """Gets the training config proto."""

    training_config = abstract_learner_pb2.TrainingConfig(
        learner=self._learner_name,
        label=self._label,
        weight_definition=self._build_weight_definition(),
        ranking_group=self._ranking_group,
        uplift_treatment=self._uplift_treatment,
        task=self._task._to_proto_type(),
        metadata=abstract_model_pb2.Metadata(framework=_FRAMEWORK_NAME),
    )

    # Apply monotonic constraints.
    if self._data_spec_args.columns:
      for feature in self._data_spec_args.columns:
        assert feature is not None
        if not feature.normalized_monotonic:
          continue

        proto_direction = (
            abstract_learner_pb2.MonotonicConstraint.INCREASING
            if feature.normalized_monotonic == dataspec.Monotonic.INCREASING
            else abstract_learner_pb2.MonotonicConstraint.DECREASING
        )
        training_config.monotonic_constraints.append(
            abstract_learner_pb2.MonotonicConstraint(
                feature=_feature_name_to_regex(feature.name),
                direction=proto_direction,
            )
        )

    if self._tuner:
      training_config.MergeFrom(self._tuner.train_config)
    return training_config

  def _get_learner(self) -> ydf.GenericCCLearner:
    """Gets a ready-to-train learner."""

    training_config = self._get_training_config()
    cc_custom_loss = None
    if "loss" in self._hyperparameters and isinstance(
        self._hyperparameters["loss"], custom_loss.AbstractCustomLoss
    ):
      log.info(
          "Using a custom loss. Note when using custom losses, hyperparameter"
          " `apply_link_function` is ignored. Use the losses' activation"
          " function instead."
      )
      py_custom_loss: custom_loss.AbstractCustomLoss = self._hyperparameters[
          "loss"
      ]
      py_custom_loss.check_is_compatible_task(training_config.task)
      cc_custom_loss = py_custom_loss._to_cc()  # pylint: disable=protected-access
      # TODO: b/322763329 - Fail if the user set apply_link_function.
      if py_custom_loss.activation.name == "IDENTITY":
        self._hyperparameters["apply_link_function"] = False
      else:
        self._hyperparameters["apply_link_function"] = True

    hp_proto = hp_lib.dict_to_generic_hyperparameter(self._hyperparameters)
    return ydf.GetLearner(
        training_config, hp_proto, self._deployment_config, cc_custom_loss
    )

  def _get_vertical_dataset(
      self, ds: dataset.InputDataset
  ) -> dataset.VerticalDataset:
    if isinstance(ds, dataset.VerticalDataset):
      return ds
      # TODO: Check that the user has not specified a data spec guide.
    else:

      # List of columns that cannot be unrolled.
      single_dim_columns = [self._label]
      for column in [
          self._weights,
          self._ranking_group,
          self._uplift_treatment,
      ]:
        if column:
          single_dim_columns.append(column)

      effective_data_spec_args = None
      if self._data_spec is None:
        effective_data_spec_args = self._build_data_spec_args()

      required_columns = None  # All columns in the dataspec are required.
      if self._task == Task.ANOMALY_DETECTION:
        if self._data_spec is not None:
          required_columns = [
              col.name
              for col in self._data_spec.columns
              if col.name != self._label
          ]
        if effective_data_spec_args is not None:
          required_columns = [
              col.name
              for col in effective_data_spec_args.columns
              if col is not None and col.name != self._label
          ]
      return dataset.create_vertical_dataset_with_spec_or_args(
          ds,
          data_spec=self._data_spec,
          inference_args=effective_data_spec_args,
          required_columns=required_columns,
          single_dim_columns=single_dim_columns,
          label=self._label if self._task != Task.ANOMALY_DETECTION else None,
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

    # In a notebook, display an interractive evaluation
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
        bootstrapping_samples=bootstrapping_samples,
        task=self._task._to_proto_type(),
    )

    deployment_evaluation = abstract_learner_pb2.DeploymentConfig(
        num_threads=parallel_evaluations,
    )

    vertical_dataset = self._get_vertical_dataset(ds)
    learner = self._get_learner()

    with log.cc_log_context():
      evaluation_proto = learner.Evaluate(
          vertical_dataset._dataset,  # pylint: disable=protected-access
          fold_generator,
          evaluation_options,
          deployment_evaluation,
      )
      return metric.Evaluation(evaluation_proto)

  def _build_data_spec_args(self) -> dataspec.DataSpecInferenceArgs:
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

    def create_label_column(
        name: Optional[str], task: Task
    ) -> Optional[dataspec.Column]:
      if task in [Task.CLASSIFICATION, Task.CATEGORICAL_UPLIFT]:
        return dataspec.Column(
            name=name,
            semantic=dataspec.Semantic.CATEGORICAL,
            max_vocab_count=-1,
            min_vocab_frequency=1,
        )
      elif task in [Task.REGRESSION, Task.RANKING, Task.NUMERICAL_UPLIFT]:
        return dataspec.Column(name=name, semantic=dataspec.Semantic.NUMERICAL)
      elif task in [Task.ANOMALY_DETECTION]:
        if name is None:
          # No label column
          return None
        else:
          return dataspec.Column(
              name=name,
              semantic=dataspec.Semantic.CATEGORICAL,
              max_vocab_count=-1,
              min_vocab_frequency=1,
          )
      else:
        raise ValueError(f"Unsupported task {task.name} for label column")

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
    if dataspec.column_defs_contains_column(self._label, column_defs):
      raise ValueError(
          f"Label column {self._label} is also an input feature. A column"
          " cannot be both a label and input feature."
      )
    if (
        label_column := create_label_column(self._label, self._task)
    ) is not None:
      column_defs.append(label_column)
    if self._weights is not None:
      if dataspec.column_defs_contains_column(self._weights, column_defs):
        raise ValueError(
            f"Weights column {self._weights} is also an input feature. A column"
            " cannot be both a weights and input feature."
        )
      column_defs.append(
          dataspec.Column(
              name=self._weights, semantic=dataspec.Semantic.NUMERICAL
          )
      )
    if self._ranking_group is not None:
      assert self._task == Task.RANKING

      if dataspec.column_defs_contains_column(self._ranking_group, column_defs):
        raise ValueError(
            f"Ranking group column {self._ranking_group} is also an input"
            " feature. A column cannot be both a ranking group and input"
            " feature."
        )
      column_defs.append(
          dataspec.Column(
              name=self._ranking_group, semantic=dataspec.Semantic.HASH
          )
      )
    if self._uplift_treatment is not None:
      assert self._task in [Task.NUMERICAL_UPLIFT, Task.CATEGORICAL_UPLIFT]

      if dataspec.column_defs_contains_column(
          self._uplift_treatment, column_defs
      ):
        raise ValueError(
            "The uplift_treatment column should not be specified as a feature"
        )
      column_defs.append(
          dataspec.Column(
              name=self._uplift_treatment,
              semantic=dataspec.Semantic.CATEGORICAL,
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
      resume_training: bool,
      resume_training_snapshot_interval_seconds: int,
      working_dir: Optional[str],
      workers: Optional[Sequence[str]],
  ):
    """Merges constructor arguments into a deployment configuration."""

    if num_threads is None:
      num_threads = self._determine_optimal_num_threads()
    config = abstract_learner_pb2.DeploymentConfig(
        num_threads=num_threads,
        try_resume_training=resume_training,
        cache_path=working_dir,
        resume_training_snapshot_interval_seconds=resume_training_snapshot_interval_seconds,
    )

    if workers is not None:
      if not workers:
        raise ValueError("At least one worker should be provided")
      config.distribute.implementation_key = "GRPC"
      grpc_config = config.distribute.Extensions[grpc_pb2.grpc]
      grpc_config.grpc_addresses.addresses[:] = workers

    return config

  def _determine_optimal_num_threads(self):
    """Sets  number of threads to min(num_cpus, 32) or 6 if num_cpus unclear."""
    num_threads = os.cpu_count()
    if num_threads is None:
      logging.warning("Cannot determine the number of CPUs. Set num_threads=6")
      num_threads = 6
    else:
      if num_threads > 32:
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

  @classmethod
  def capabilities(cls) -> abstract_learner_pb2.LearnerCapabilities:
    raise NotImplementedError("Not implemented")


def _feature_name_to_regex(name: str) -> str:
  """Generates a regular expression capturing a feature by name."""

  return "^" + re.escape(name) + "$"
