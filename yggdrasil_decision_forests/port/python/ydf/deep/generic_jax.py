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

"""JAX backed models."""

import abc
import dataclasses
import functools
import logging
import math
import os
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple, Type, TypeVar, Union

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import optax
import safetensors
import safetensors.flax

from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.learner import abstract_learner_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.cc import ydf as ydf_cc
from ydf.dataset import dataset as dataset_lib
from ydf.dataset import dataspec as dataspec_lib
from ydf.dataset.io import dataset_io as dataset_io_lib
from ydf.dataset.io import generator as generator_lib
from ydf.deep import analysis as py_analysis_lib
from ydf.deep import dataset as deep_dataset_lib
from ydf.deep import deep_model_pb2
from ydf.deep import hyperparameter as hyperparameter_lib
from ydf.deep import metric as deep_metric_lib
from ydf.deep import preprocessor as preprocessor_lib
from ydf.deep import safetensors as safetensors_lib
from ydf.learner import abstract_feature_selector as abstract_feature_selector_lib
from ydf.learner import generic_learner
from ydf.learner import hyperparameters as hp_lib
from ydf.learner import tuner as tuner_lib
from ydf.metric import evaluate as evaluate_lib
from ydf.metric import metric as metric_lib
from ydf.model import analysis as analysis_lib
from ydf.model import feature_selector_logs
from ydf.model import generic_model
from ydf.model import model_metadata
from ydf.model import optimizer_logs
from ydf.utils import concurrency
from ydf.utils import filesystem
from ydf.utils import html
from ydf.utils import log
from yggdrasil_decision_forests.utils import model_analysis_pb2

# Generic hyperparameters of all the deep learning models.
# Number of training examples in each mini-batch
_HP_BATCH_SIZE = "batch_size"
# Number of times the training dataset is scanned during training.
_HP_NUM_EPOCHS = "num_epochs"
# Learning rate
_HP_LEARNING_RATE = "learning_rate"
# How the learning changes during the course of training
_HP_LEARNING_RATE_POLICY = "learning_rate_policy"
# Maximum number of training steps. If None, ignored.
_HP_NUM_STEPS = "num_steps"
# Random seed for the training.
_HP_RANDOM_SEED = "random_seed"
# Stops training is the validation loss does not decrease over the last N
# epochs. If None, ignored, and early stopping is disabled. Ignored if no
# validation dataset is provided.
_HP_EARLY_STOPPING_EPOCH_PATIENCE = "early_stopping_epoch_patience"
# When early stopping triggers, should the model parameters be reverted to the
# best performing model? (costs more memory, produces better models).
_HP_EARLY_STOPPING_REVERT_PARAMS = "early_stopping_revert_params"

_HP_VALUE_LEARNING_RATE_POLICY_CONSTANT = "constant"
_HP_VALUE_LEARNING_RATE_POLICY_COSINE_DECAY = "cosine_decay"
LearningRatePolicy = Literal[
    _HP_VALUE_LEARNING_RATE_POLICY_CONSTANT,
    _HP_VALUE_LEARNING_RATE_POLICY_COSINE_DECAY,
]

_HP_MAXIMUM_TRAINING_DURATION_SECONDS = "maximum_training_duration_seconds"

_WEIGHTS_FILE_NAME = "weights"
_ABSTRACT_MODEL_FILE_NAME = "abstract_model.pb"
_DATA_SPEC_FILE_NAME = "data_spec.pb"
_DEEP_MODEL_PROTO_FILE_NAME = "config.pb"
_DONE_FILE_NAME = "done"

PyTree = jaxtyping.PyTree
Batch = deep_dataset_lib.JaxExampleBatch  # Batch of observations
ModelState = PyTree  # The state of the model (generally Params + BatchState).
Params = PyTree  # The parameters of the model learned with backprop.
OptState = PyTree  # The state of the optimizer.
BatchState = PyTree  # The state of the batch norm layers.
ApplyModelFn = Callable[
    [ModelState, Batch, bool, jax.Array], Tuple[jax.Array, Optional[BatchState]]
]

GenericJAXModelClass = Type[TypeVar("B", bound="GenericJAXModel")]


# Batch size used to generate model predictions
# TODO: Is there value to parametrize it?
_PREDICT_BATCH_SIZE = 256

# If true, print the value of intermediate JAX arrays during training. Great for
# debugging.
_DEBUG_PRINT = False

def jax_debug(name: str, x: jax.Array) -> None:
  """Prints the content of a jax array at runtime if debug prints is enabled."""
  if not _DEBUG_PRINT:
    return
  if isinstance(x, jax.Array):
    info = f"{x.shape!r}"
  else:
    info = ""
  jax.debug.print(f"[JAX] {name} {info}:{{x}}", x=x, ordered=True)


@dataclasses.dataclass
class Objective:
  """Collection of functions to train a model and measure its quality."""

  loss: deep_metric_lib.Metric
  metrics: List[deep_metric_lib.Metric]
  activation_fn: Callable[[jax.Array], jax.Array]
  output_dim: int = 1


class GenericJAXModel(generic_model.GenericModel):
  """Abstract base class for all the JAX based YDF models."""

  def __init__(
      self,
      dataspec: data_spec_pb2.DataSpecification,
      preprocessor: preprocessor_lib.Preprocessor,
      abstract_model_proto: abstract_model_pb2.AbstractModel,
  ):
    self._dataspec = dataspec
    self._preprocessor = preprocessor
    self._model_state = None
    self._abstract_model_proto = abstract_model_proto
    self._jit_apply_model = None

  @abc.abstractmethod
  def set_config_from_hyperparameters(
      self, hps: hyperparameter_lib.HyperparameterConsumer
  ) -> None:
    """Configures the model structure from the hyperparameters."""
    raise NotImplementedError

  @abc.abstractmethod
  def make_jax_module(self):
    """Creates a FLAX/JAX module running the core model.

    This module does not include the preprocessing steps (premodel and inmodel)
    as well as the post-model / post-metrics activation functions.
    """
    raise NotImplementedError

  def set_model_state(self, model_state: ModelState) -> None:
    """Sets the model state a.k.a. trainable weights."""
    self._model_state = model_state

  def _output_dim(self) -> int:
    """Number of dimensions of the raw model output."""
    return self._get_objective().output_dim

  def _get_objective(self) -> Objective:
    """Gets the objective to train the model."""

    if (
        self._abstract_model_proto.task
        == abstract_model_pb2.Task.CLASSIFICATION
    ):
      label_classes = self.label_classes()
      num_classes = len(label_classes)
      if len(label_classes) == 2:
        return Objective(
            loss=deep_metric_lib.LossBinaryClassificationMetric(),
            metrics=[deep_metric_lib.AccuracyBinaryClassificationMetric()],
            activation_fn=jax.nn.sigmoid,
        )
      else:
        return Objective(
            loss=deep_metric_lib.LossMultiClassClassificationMetric(
                num_classes
            ),
            metrics=[
                deep_metric_lib.AccuracyMultiClassClassificationMetric(
                    num_classes
                )
            ],
            activation_fn=jax.nn.softmax,
            output_dim=num_classes,
        )
    elif self._abstract_model_proto.task == abstract_model_pb2.Task.REGRESSION:
      return Objective(
          loss=deep_metric_lib.MeanSquaredErrorMetric(),
          metrics=[],
          activation_fn=lambda x: x,
      )
    else:
      raise ValueError(
          "Non supported task"
          f" {generic_model.Task._from_proto_type(self._abstract_model_proto.task)}"  # pylint: disable=protected-access
      )

  @classmethod
  @abc.abstractmethod
  def name(cls) -> str:
    raise NotImplementedError  # TODO: Implement.

  def __getstate__(self):
    raise NotImplementedError  # TODO: Implement.

  def __setstate__(self, state):
    raise NotImplementedError  # TODO: Implement.

  @abc.abstractmethod
  def _build_proto_config(self, model_proto: deep_model_pb2.DeepModel) -> None:
    raise NotImplementedError

  def task(self) -> generic_model.Task:
    return generic_model.Task._from_proto_type(self._abstract_model_proto.task)  # pylint: disable=protected-access

  def metadata(self) -> model_metadata.ModelMetadata:
    return model_metadata.ModelMetadata._from_proto_type(  # pylint: disable=protected-access
        self._abstract_model_proto.metadata
    )

  def set_metadata(self, metadata: model_metadata.ModelMetadata):
    self._abstract_model_proto.metadata.CopyFrom(
        model_metadata.ModelMetadata._to_proto_type(metadata)  # pylint: disable=protected-access
    )

  def set_feature_selection_logs(
      self, value: Optional[feature_selector_logs.FeatureSelectorLogs]
  ) -> None:
    self._abstract_model_proto.feature_selection_logs.CopyFrom(
        feature_selector_logs.value_to_proto(value)
    )

  def feature_selection_logs(
      self,
  ) -> Optional[feature_selector_logs.FeatureSelectorLogs]:
    return feature_selector_logs.proto_to_value(
        self._abstract_model_proto.feature_selection_logs
    )

  def describe(
      self,
      output_format: Literal["auto", "text", "notebook", "html"] = "auto",
      full_details: bool = False,
  ) -> Union[str, html.HtmlNotebookDisplay]:
    # TODO: Implement. Store and show the FLAX model description in
    # a describe tab.
    return "<Model description not implemented>"

  def data_spec(self) -> data_spec_pb2.DataSpecification:
    return self._dataspec

  def benchmark(
      self,
      ds: dataset_lib.InputDataset,
      benchmark_duration: float = 3,
      warmup_duration: float = 1,
      batch_size: int = 100,
      num_threads: Optional[int] = None,
  ) -> ydf_cc.BenchmarkInferenceCCResult:

    if benchmark_duration <= 0:
      raise ValueError(
          "The duration of the benchmark must be positive, got"
          f" {benchmark_duration}"
      )
    if warmup_duration <= 0:
      raise ValueError(
          "The duration of the warmup phase must be positive, got"
          f" {warmup_duration}."
      )
    if batch_size <= 0:
      raise ValueError(
          f"The batch size of the benchmark must be positive, got {batch_size}."
      )

    raise NotImplementedError

  def save(
      self,
      path: str,
      advanced_options=generic_model.ModelIOOptions(),
      pure_serving: bool = False,
  ) -> None:
    if self._model_state is None:
      raise ValueError("This model has not been trained")

    weights_file = filesystem.Path(path) / _WEIGHTS_FILE_NAME
    abstract_model_file = filesystem.Path(path) / _ABSTRACT_MODEL_FILE_NAME
    data_spec_file = filesystem.Path(path) / _DATA_SPEC_FILE_NAME
    deep_model_proto_file = filesystem.Path(path) / _DEEP_MODEL_PROTO_FILE_NAME
    done_file = filesystem.Path(path) / _DONE_FILE_NAME

    deep_model_proto = deep_model_pb2.DeepModel(
        preprocessor=self._preprocessor.to_proto(),
        weights=deep_model_pb2.Weights(
            format=deep_model_pb2.Weights.SAFETENSORS
        ),
    )
    self._build_proto_config(deep_model_proto)

    os.makedirs(path, exist_ok=False)
    with filesystem.Open(weights_file, "wb") as f:
      f.write(
          safetensors.flax.save(
              safetensors_lib.flatten_weights(self._model_state)
          )
      )
    with filesystem.Open(abstract_model_file, "wb") as f:
      f.write(self._abstract_model_proto.SerializeToString())
    with filesystem.Open(data_spec_file, "wb") as f:
      f.write(self._dataspec.SerializeToString())
    with filesystem.Open(deep_model_proto_file, "wb") as f:
      f.write(deep_model_proto.SerializeToString())
    done_file.touch()

  def serialize(self) -> bytes:
    raise NotImplementedError  # TODO: Implement.

  def predict(
      self,
      data: dataset_lib.InputDataset,
      *,
      use_slow_engine: bool = False,
      num_threads: Optional[int] = None,
  ) -> np.ndarray:

    del num_threads
    # TODO: Control of the number of threads in JAX evaluation.

    if self._jit_apply_model is None:
      objective = self._get_objective()
      module = self.make_jax_module()

      def _apply_model(batch_features: Batch):
        raw_preds = module.apply(
            self._model_state,
            batch_features,
            training=False,
        )
        if objective.output_dim == 1:
          raw_preds = jnp.squeeze(raw_preds, axis=1)
        return objective.activation_fn(raw_preds)

      self._jit_apply_model = jax.jit(_apply_model)

    with jax.profiler.TraceAnnotation("predict"):
      ds_generator = dataset_io_lib.build_batched_example_generator(data)
      all_predictions = []
      for batch_idx, raw_numpy_batch in enumerate(
          ds_generator.generate(batch_size=_PREDICT_BATCH_SIZE, shuffle=False)
      ):
        with jax.profiler.StepTraceAnnotation(
            "prepare_data", step_num=batch_idx
        ):
          numpy_batch = self._preprocessor.apply_premodel(
              raw_numpy_batch, has_labels=True
          )
          jax_batch = deep_dataset_lib.batch_numpy_to_jax(numpy_batch)
        with jax.profiler.StepTraceAnnotation(
            "apply_model", step_num=batch_idx
        ):
          predictions = self._jit_apply_model(jax_batch)
        # TODO: Group to avoid sync at each step.
        all_predictions.append(np.asarray(predictions))
      return np.concatenate(all_predictions, axis=0)

  def evaluate(
      self,
      data: dataset_lib.InputDataset,
      *,
      weighted: Optional[bool] = None,
      task: Optional[generic_model.Task] = None,
      label: Optional[str] = None,
      group: Optional[str] = None,
      bootstrapping: Union[bool, int] = False,
      ndcg_truncation: int = 5,
      mrr_truncation: int = 5,
      evaluation_task: Optional[generic_model.Task] = None,
      use_slow_engine: bool = False,
      num_threads: Optional[int] = None,
  ) -> metric_lib.Evaluation:
    if num_threads is None:
      num_threads = concurrency.determine_optimal_num_threads(training=False)

    predictions = self.predict(
        data, use_slow_engine=use_slow_engine, num_threads=num_threads
    )

    label_classes = None
    try:
      label_classes = self.label_classes()
    except ValueError:
      pass

    if label is None:
      label = self.label()
    labels = []

    # Warning about deprecation of "evaluation_task"
    if evaluation_task is not None:
      log.warning(
          "The `evaluation_task` argument is deprecated. Use `task` instead.",
          message_id=log.WarningMessage.DEPRECATED_EVALUATION_TASK,
      )
      if task is not None:
        raise ValueError("Cannot specify both `task` and `evaluation_task`")
      task = evaluation_task

    # TODO: Implement non-supported cases.
    if weighted is not None:
      raise NotImplementedError(
          "Weighted evaluation is not implemented for Generic JAX models"
      )
    if group is not None:
      raise NotImplementedError(
          "Ranking evaluation with groups is not implemented for Generic JAX"
          " models"
      )

    if task is None:
      task = self.task()

    ds_generator = dataset_io_lib.build_batched_example_generator(data)
    for numpy_batch in ds_generator.generate(
        batch_size=_PREDICT_BATCH_SIZE, shuffle=False
    ):
      labels.append(numpy_batch[label])

    return evaluate_lib.evaluate_predictions(
        predictions=predictions,
        labels=np.concatenate(labels, axis=0) if labels else None,
        task=task,
        label_classes=label_classes,
        bootstrapping=bootstrapping,
        ndcg_truncation=ndcg_truncation,
        mrr_truncation=mrr_truncation,
        num_threads=num_threads,
    )

  def analyze_prediction(
      self,
      single_example: dataset_lib.InputDataset,
  ) -> analysis_lib.PredictionAnalysis:
    raise NotImplementedError

  def analyze(
      self,
      data: dataset_lib.InputDataset,
      sampling: float = 1.0,
      num_bins: int = 50,
      partial_dependence_plot: bool = True,
      conditional_expectation_plot: bool = True,
      permutation_variable_importance_rounds: int = 1,
      num_threads: Optional[int] = None,
      maximum_duration: Optional[float] = 20,
  ) -> analysis_lib.Analysis:
    enable_permutation_variable_importances = (
        permutation_variable_importance_rounds > 0
    )
    options_proto = model_analysis_pb2.Options(
        num_threads=num_threads,
        maximum_duration_seconds=maximum_duration,
        pdp=model_analysis_pb2.Options.PlotConfig(
            enabled=partial_dependence_plot,
            example_sampling=sampling,
            num_numerical_bins=num_bins,
        ),
        cep=model_analysis_pb2.Options.PlotConfig(
            enabled=conditional_expectation_plot,
            example_sampling=sampling,
            num_numerical_bins=num_bins,
        ),
        permuted_variable_importance=model_analysis_pb2.Options.PermutedVariableImportance(
            enabled=enable_permutation_variable_importances,
            num_rounds=permutation_variable_importance_rounds,
        ),
        include_model_structural_variable_importances=True,
    )

    return py_analysis_lib.model_analysis(
        self,
        data,
        options_proto,
    )

  def to_cpp(self, key: str = "my_model") -> str:
    raise NotImplementedError  # TODO: Implement.

  # TODO: Change default value of "mode" before 1.0 release.
  def to_tensorflow_saved_model(  # pylint: disable=dangerous-default-value
      self,
      path: str,
      input_model_signature_fn: Any = None,
      *,
      mode: Literal["keras", "tf"] = "keras",
      feature_dtypes: Dict[str, "export_tf.TFDType"] = {},  # pytype: disable=name-error
      servo_api: bool = False,
      feed_example_proto: bool = False,
      pre_processing: Optional[Callable] = None,  # pylint: disable=g-bare-generic
      post_processing: Optional[Callable] = None,  # pylint: disable=g-bare-generic
      temp_dir: Optional[str] = None,
      tensor_specs: Optional[Dict[str, Any]] = None,
      feature_specs: Optional[Dict[str, Any]] = None,
      force: bool = False,
  ) -> None:
    raise NotImplementedError  # TODO: Implement.

  def to_tensorflow_function(  # pytype: disable=name-error
      self,
      temp_dir: Optional[str] = None,
      can_be_saved: bool = True,
      squeeze_binary_classification: bool = True,
      force: bool = False,
  ) -> Any:
    raise NotImplementedError  # TODO: Implement.

  def to_jax_function(  # pytype: disable=name-error
      self,
      jit: bool = True,
      apply_activation: bool = True,
      leaves_as_params: bool = False,
      compatibility: Union[str, Any] = "XLA",
  ) -> Any:
    raise NotImplementedError  # TODO: Implement.

  def update_with_jax_params(self, params: Dict[str, Any]):
    raise NotImplementedError  # TODO: Implement.

  def to_docker(
      self,
      path: str,
      exist_ok: bool = False,
  ) -> None:
    raise NotImplementedError  # TODO: Implement.

  def hyperparameter_optimizer_logs(
      self,
  ) -> Optional[optimizer_logs.OptimizerLogs]:
    raise NotImplementedError  # TODO: Implement.

  def variable_importances(self) -> Dict[str, List[Tuple[float, str]]]:
    variable_importances = {}
    # Collect the variable importances stored in the model.
    for (
        name,
        importance_set,
    ) in self._abstract_model_proto.precomputed_variable_importances.items():
      variable_importances[name] = [
          (src.importance, self.data_spec().columns[src.attribute_idx].name)
          for src in importance_set.variable_importances
      ]
    return variable_importances

  def label_col_idx(self) -> int:
    return self._abstract_model_proto.label_col_idx

  def input_features_col_idxs(self) -> Sequence[int]:
    return list(self._abstract_model_proto.input_features)

  def list_compatible_engines(self) -> Sequence[str]:
    return ["jax"]

  def force_engine(self, engine_name: Optional[str]) -> None:
    if engine_name != "jax":
      raise ValueError("The only supported engine is jax")


class GenericJaxLearner(generic_learner.GenericLearner):
  """A generic YDF learner using JAX for training."""

  def __init__(
      self,
      learner_name: str,
      task: generic_model.Task,
      label: Optional[str],
      weights: Optional[str],
      class_weights: Optional[Dict[str, float]],
      ranking_group: Optional[str],
      uplift_treatment: Optional[str],
      data_spec: Optional[data_spec_pb2.DataSpecification],
      features: Optional[dataspec_lib.ColumnDefs],
      include_all_columns: bool,
      max_vocab_count: int,
      min_vocab_frequency: int,
      max_num_scanned_rows_to_infer_semantic: int,
      max_num_scanned_rows_to_compute_statistics: int,
      maximum_training_duration_seconds: float,
      working_dir: Optional[str],
      num_threads: Optional[int],
      hyper_parameters: hp_lib.HyperParameters,
      explicit_learner_arguments: Optional[Set[str]],
      tuner: Optional[tuner_lib.AbstractTuner],
      feature_selector: Optional[
          abstract_feature_selector_lib.AbstractFeatureSelector
      ],
      batch_size: int,
      num_epochs: int,
      learning_rate: float,
      learning_rate_policy: LearningRatePolicy,
      num_steps: Optional[int],
      random_seed: int,
      early_stopping_epoch_patience: Optional[int],
      early_stopping_revert_params: bool,
      num_quantiles: int,
      zscore: bool,
  ):
    hyper_parameters = hyper_parameters.copy()
    hyper_parameters[_HP_BATCH_SIZE] = batch_size
    hyper_parameters[_HP_NUM_EPOCHS] = num_epochs
    hyper_parameters[_HP_LEARNING_RATE] = learning_rate
    hyper_parameters[_HP_LEARNING_RATE_POLICY] = learning_rate_policy
    hyper_parameters[_HP_NUM_STEPS] = num_steps
    hyper_parameters[_HP_RANDOM_SEED] = random_seed
    hyper_parameters[_HP_EARLY_STOPPING_EPOCH_PATIENCE] = (
        early_stopping_epoch_patience
    )
    hyper_parameters[_HP_EARLY_STOPPING_REVERT_PARAMS] = (
        early_stopping_revert_params
    )
    hyper_parameters[_HP_MAXIMUM_TRAINING_DURATION_SECONDS] = (
        maximum_training_duration_seconds
    )

    if weights is not None or class_weights is not None:
      raise ValueError(
          "Training with sample weights or class weights is not yet supported"
          " for deep models."
      )

    data_spec_args = dataspec_lib.DataSpecInferenceArgs(
        columns=dataspec_lib.normalize_column_defs(features),
        include_all_columns=include_all_columns,
        max_vocab_count=max_vocab_count,
        min_vocab_frequency=min_vocab_frequency,
        discretize_numerical_columns=num_quantiles > 0,
        num_discretized_numerical_bins=num_quantiles,
        max_num_scanned_rows_to_infer_semantic=max_num_scanned_rows_to_infer_semantic,
        max_num_scanned_rows_to_compute_statistics=max_num_scanned_rows_to_compute_statistics,
        zscore_numerical_columns=zscore,
    )
    super().__init__(
        learner_name=learner_name,
        task=task,
        label=label,
        weights=weights,
        class_weights=class_weights,
        ranking_group=ranking_group,
        uplift_treatment=uplift_treatment,
        data_spec_args=data_spec_args,
        data_spec=data_spec,
        hyper_parameters=hyper_parameters,
        explicit_learner_arguments=explicit_learner_arguments,
        deployment_config=abstract_learner_pb2.DeploymentConfig(),
        tuner=tuner,
        feature_selector=feature_selector,
        extra_training_config=abstract_learner_pb2.TrainingConfig(),
    )

  @classmethod
  def _capabilities(cls) -> abstract_learner_pb2.LearnerCapabilities:
    return abstract_learner_pb2.LearnerCapabilities(
        support_validation_dataset=True,
        require_label=True,
        support_max_training_duration=True,
    )

  @abc.abstractmethod
  def model_class(self) -> GenericJAXModelClass:
    raise NotImplementedError

  def _post_init(self):
    pass

  def _learning_rate(
      self,
      hps: hyperparameter_lib.HyperparameterConsumer,
      steps_per_epoch: Optional[int],
      num_epochs: int,
  ) -> Union[float, optax.Schedule]:
    learning_rate = hps.get_float(_HP_LEARNING_RATE)
    learning_rate_policy = hps.get_str(_HP_LEARNING_RATE_POLICY)
    if learning_rate_policy == _HP_VALUE_LEARNING_RATE_POLICY_CONSTANT:
      return learning_rate
    elif learning_rate_policy == _HP_VALUE_LEARNING_RATE_POLICY_COSINE_DECAY:
      if num_epochs <= 1:
        raise ValueError(
            "Need num_epochs>=2 for"
            f" {_HP_VALUE_LEARNING_RATE_POLICY_COSINE_DECAY} learning rate"
            " policy"
        )
      warmup_epochs = 1  # TODO: Make it a parameter
      if steps_per_epoch is None:
        log.warning(
            "Impossible to determine the number of steps per epoch. Assuming"
            " 1000 steps/epochs for the dynamic learning-rate policy."
        )
        steps_per_epoch = 1000
      transition_steps = int(warmup_epochs * steps_per_epoch)
      warmup_fn = optax.linear_schedule(
          init_value=0.0,
          end_value=learning_rate,
          transition_steps=transition_steps,
      )
      cosine_fn = optax.cosine_decay_schedule(
          init_value=learning_rate,
          decay_steps=int(steps_per_epoch * num_epochs - transition_steps),
      )
      return optax.join_schedules(
          schedules=[warmup_fn, cosine_fn],
          boundaries=[transition_steps],
      )
    else:
      raise ValueError(
          f"Unknown value {learning_rate_policy!r} for argument"
          f" {_HP_LEARNING_RATE_POLICY!r}. Supported values are:"
          f" {_HP_VALUE_LEARNING_RATE_POLICY_CONSTANT},"
          f" {_HP_VALUE_LEARNING_RATE_POLICY_COSINE_DECAY}."
      )

  def _label_col_idx(self, dataspec: data_spec_pb2.DataSpecification) -> int:
    for col_idx, col in enumerate(dataspec.columns):
      if col.name == self._label:
        return col_idx
    raise ValueError(
        f"Cannot find label column {self._label!r} in dataspec. The available"
        f" columns are: {[c.name for c in dataspec.columns]!r}"
    )

  def _train_imp(
      self,
      ds: dataset_lib.InputDataset,
      valid: Optional[dataset_lib.InputDataset],
      verbose: Optional[Union[int, bool]],
  ) -> generic_model.ModelType:

    if _DEBUG_PRINT:
      jax.config.update("jax_debug_nans", True)

    if valid is None:
      log.warning(
          "Training this model without a validation dataset can take a long"
          " time and lead to sub-optimal performance",
          message_id=log.WarningMessage.TRAINING_NEURAL_NET_WITHOUT_VALID,
      )
    # Prepare the datasets
    ds_generator = dataset_io_lib.build_batched_example_generator(ds)
    valid_ds_generator = (
        dataset_io_lib.build_batched_example_generator(valid)
        if valid is not None
        else None
    )

    if self._data_spec is None:
      log.debug("Determine dataspec")
      dataspec = self._infer_dataspec(ds_generator)
    else:
      dataspec = self._data_spec

    log.debug("Dataspec:\n%s", ydf_cc.DataspecToTextReport(dataspec))
    log.debug("Devices:%s", jax.devices())

    # Parse the training hyper-parameters
    hp_consumer = hyperparameter_lib.HyperparameterConsumer(
        self._hyperparameters
    )
    num_epochs = hp_consumer.get_int(_HP_NUM_EPOCHS)
    batch_size = hp_consumer.get_int(_HP_BATCH_SIZE)
    num_steps = hp_consumer.get_optional_int(_HP_NUM_STEPS)
    random_seed = hp_consumer.get_int(_HP_RANDOM_SEED)
    early_stopping_epoch_patience = hp_consumer.get_int(
        _HP_EARLY_STOPPING_EPOCH_PATIENCE
    )
    early_stopping_revert_params = hp_consumer.get_bool(
        _HP_EARLY_STOPPING_REVERT_PARAMS
    )
    maximum_training_duration_seconds = hp_consumer.get_float(
        _HP_MAXIMUM_TRAINING_DURATION_SECONDS
    )

    # Create a non-trained model
    non_input_features_cols = set(self._non_input_feature_columns())
    input_features_col_idxs = [
        col_idx
        for col_idx, col in enumerate(dataspec.columns)
        if col.name not in non_input_features_cols
    ]
    model_class = self.model_class()
    model = model_class(
        dataspec=dataspec,
        preprocessor=preprocessor_lib.Preprocessor(
            dataspec=dataspec,
            input_features_col_idxs=input_features_col_idxs,
            numerical_zscore=self._data_spec_args.zscore_numerical_columns,
            numerical_quantiles=self._data_spec_args.discretize_numerical_columns,
        ),
        abstract_model_proto=abstract_model_pb2.AbstractModel(
            name=model_class.name(),
            task=self._task._to_proto_type(),  # pylint: disable=protected-access
            label_col_idx=self._label_col_idx(dataspec),
            input_features=input_features_col_idxs,
        ),
    )
    model.set_config_from_hyperparameters(hp_consumer)

    # Prepare training variables accessible from the traced jax functions.
    steps_per_epoch = ds_generator.num_batches(batch_size)
    optimizer = optax.adamw(
        self._learning_rate(
            hp_consumer, steps_per_epoch=steps_per_epoch, num_epochs=num_epochs
        )
    )
    objective = model._get_objective()  # pylint: disable=protected-access
    hp_consumer.finalize()

    module = model.make_jax_module()

    log.debug("Training loss: %s", objective.loss.name())

    # TODO: Implement weighted training.

    # Generate model predictions
    @functools.partial(jax.jit, static_argnames=("training",))
    def apply_model(
        model_state: ModelState,
        batch_features: Batch,
        training: bool,
        random_key: jax.Array,
    ) -> Tuple[jax.Array, Optional[BatchState]]:
      log.debug("Compiling model")
      kwargs = {}
      if training and "batch_stats" in model_state:
        # During training, if the model is trained with batch normalization, the
        # batch variable should be mutable from the point of view of FLAX.
        kwargs["mutable"] = ["batch_stats"]
        return_batch_stats = True
      else:
        return_batch_stats = False
      preds = module.apply(
          model_state,
          batch_features,
          training=training,
          rngs={"dropout": random_key},
          **kwargs,
      )

      def normalize_preds(x):
        if objective.output_dim == 1:
          return jnp.squeeze(x, axis=1)
        else:
          return x

      jax_debug("preds", preds)
      if return_batch_stats:
        assert isinstance(preds, tuple)
        return normalize_preds(preds[0]), preds[1]
      else:
        assert isinstance(preds, jax.Array)
        return normalize_preds(preds), None

    # Computes the loss of the model
    def train_loss(
        params: Params,
        batch_stats: Optional[BatchState],
        batch_features: Batch,
        batch_labels: jax.Array,
        key: jax.Array,
    ) -> Tuple[jax.Array, Optional[BatchState]]:
      model_state = {"params": params}
      if batch_stats is not None:
        model_state["batch_stats"] = batch_stats
      preds, batch_stats = apply_model(model_state, batch_features, True, key)
      return objective.loss(batch_labels, preds), batch_stats

    loss_gradient_fn = jax.value_and_grad(train_loss, has_aux=True)

    # Runs a single step of gradient descent
    @jax.jit
    def train_step(
        opt_state: Any,
        model_state: ModelState,
        batch_features: Batch,
        batch_labels: jax.Array,
        step_rnd_key: jax.Array,
        epoch_sum_train_loss: jax.Array,
        rapid_logs_sum_train_loss: jax.Array,
    ) -> Tuple[OptState, ModelState, jax.Array, jax.Array]:
      logging.debug("Tracing train-step")

      jax_debug("opt_state", opt_state)
      jax_debug("model_state", model_state)

      (loss, new_batch_stats), grads = loss_gradient_fn(
          model_state["params"],
          model_state.get("batch_stats", None),
          batch_features,
          batch_labels,
          step_rnd_key,
      )

      jax_debug("loss", loss)
      jax_debug("grads", grads)

      updates, opt_state = optimizer.update(
          grads, opt_state, params=model_state["params"]
      )

      jax_debug("updates", updates)
      jax_debug("opt_state", opt_state)

      new_params = optax.apply_updates(model_state["params"], updates)

      jax_debug("new_params", new_params)
      jax_debug("new_batch_stats", new_batch_stats)

      model_state.update(params=new_params)
      if new_batch_stats is not None:
        model_state.update(batch_stats=new_batch_stats["batch_stats"])
      logging.debug("Tracing train-step DONE")

      epoch_sum_train_loss += loss
      rapid_logs_sum_train_loss += loss
      return (
          opt_state,
          model_state,
          epoch_sum_train_loss,
          rapid_logs_sum_train_loss,
      )

    # Prepare the training variable not shared with the traced jax functions
    rng = jax.random.PRNGKey(random_seed)
    model_state = None
    opt_state = None

    epoch_sum_train_loss = jnp.array(0)
    epoch_num_batches = 0

    rapid_logs_sum_train_loss = jnp.array(0)
    rapid_logs_num_batches = 0

    rapid_log_interval = 30
    begin_training = None
    next_rapid_log = time.time() + rapid_log_interval
    cur_num_steps = 0

    # Best status for early stopping
    best_valid_loss = None
    best_model_state = None
    best_epoch_idx = None

    metric_computer = MetricComputer(
        learner=self,
        batch_size=batch_size * 4,
        objective=objective,
        preprocessor=model._preprocessor,  # pylint: disable=protected-access
        has_train=valid_ds_generator is None,
        has_valid=valid_ds_generator is not None,
        apply_model=apply_model,
    )

    # TODO: This code uses raw JAX. If we can identify a mature and
    # open-source alternative (e.g., jaxloop?), this would be better.
    stop_training = False
    for epoch_idx in range(num_epochs):
      if stop_training:
        break

      begin_epoch = time.time()

      rng, batch_generator_rng = jax.random.split(rng, 2)

      # Note: We don't use a simple for-in in order to monitor the data
      # ingestion step.
      ds_iterator = enumerate(
          ds_generator.generate(
              batch_size=batch_size,
              shuffle=True,
              seed=batch_generator_rng[0].item(),
          )
      )

      while True:
        with jax.profiler.StepTraceAnnotation("train", step_num=cur_num_steps):

          if begin_training is not None and (
              maximum_training_duration_seconds >= 0
              and time.time() - begin_training
              > maximum_training_duration_seconds
          ):
            log.debug("Hit maximum training duration")
            stop_training = True
            break

          if num_steps is not None and cur_num_steps >= num_steps:
            log.debug("Hit maximum number of steps")
            stop_training = True
            break

          with jax.profiler.TraceAnnotation("generate_data"):
            # TODO: Use async.
            batch_idx, raw_numpy_batch = next(ds_iterator, (None, None))
            if batch_idx is None:
              break

          with jax.profiler.TraceAnnotation(
              "prepare_data", step_num=cur_num_steps
          ):
            with jax.profiler.TraceAnnotation("apply_premodel"):
              numpy_batch = model._preprocessor.apply_premodel(  # pylint: disable=protected-access
                  raw_numpy_batch, has_labels=True
              )
            with jax.profiler.TraceAnnotation("batch_numpy_to_jax"):
              jax_batch = deep_dataset_lib.batch_numpy_to_jax(numpy_batch)
            with jax.profiler.TraceAnnotation("decompose_batch"):
              batch_features, batch_labels = self._decompose_batch(jax_batch)

          if _DEBUG_PRINT:
            log.debug("====================================================")
            log.debug("numpy_batch:\n%s", numpy_batch)
            log.debug("jax_batch:\n%s", jax_batch)
            log.debug("batch_features:\n%s", batch_features)
            log.debug("batch_labels:\n%s", batch_labels)

          if epoch_idx == 0 and batch_idx == 0:
            # Initialize model
            with jax.profiler.TraceAnnotation("Initialize model"):
              log.debug("Initialize model")
              rng, init_model_rng = jax.random.split(rng, 2)
              model_state = module.init(
                  init_model_rng,
                  batch_features,
                  training=True,
              )
              opt_state = optimizer.init(model_state["params"])
              tabulate_str = module.tabulate(
                  init_model_rng,
                  batch_features,
                  training=True,
                  compute_flops=True,
                  compute_vjp_flops=False,
                  console_kwargs={"color_system": None, "width": None},
              )
              log.debug("Model structure\n%s", tabulate_str)
              begin_training = time.time()

          with jax.profiler.TraceAnnotation("train_step"):
            rng, train_step_rng = jax.random.split(rng, 2)
            (
                opt_state,
                model_state,
                epoch_sum_train_loss,
                rapid_logs_sum_train_loss,
            ) = train_step(
                opt_state,
                model_state,
                batch_features,
                batch_labels,
                train_step_rng,
                epoch_sum_train_loss,
                rapid_logs_sum_train_loss,
            )

          if _DEBUG_PRINT:
            if jnp.isnan(epoch_sum_train_loss):
              raise ValueError("Training has diverged")

        cur_num_steps += 1
        epoch_num_batches += 1
        rapid_logs_num_batches += 1

        current_time = time.time()
        if current_time >= next_rapid_log:
          next_rapid_log = current_time + rapid_log_interval
          log.info("\tstep:%d", cur_num_steps)
          # TODO: Show the training loss (e.g.,
          # epoch_sum_train_loss.item() / rapid_logs_num_batches) without
          # stalling the pipeline (i.e., in an async way).
          # TODO: Check for training divergence i.e., loss is NaN.

      if stop_training and epoch_num_batches == 0:
        # No training was done in this epoch
        break

      # Valid and train evaluation
      begin_eval = time.time()
      with jax.profiler.StepTraceAnnotation(
          "evaluation", step_num=cur_num_steps
      ):
        extra_snippet, valid_loss = self._evaluation_snippet(
            metric_computer,
            ds_generator,
            valid_ds_generator,
            model_state=model_state,
        )

      if maximum_training_duration_seconds >= 0:
        extra_snippet = (
            f" time:{time.time() - begin_training:.1f}/{maximum_training_duration_seconds}"
        ) + extra_snippet

      # End of epoch logging
      now = time.time()
      train_loss = epoch_sum_train_loss.item() / epoch_num_batches
      if math.isnan(train_loss):
        raise ValueError("Training has diverged. The loss is NaN")
      delta_since_begin_epoch = now - begin_epoch
      if delta_since_begin_epoch == 0:
        delta_since_begin_epoch = 1
      step_per_seconds = epoch_num_batches / delta_since_begin_epoch
      ratio_eval_time = (now - begin_eval) / delta_since_begin_epoch
      log.info(
          "epoch:%d/%d step:%d step/s:%g eval-rtime:%.1f train-loss:%.2f%s",
          epoch_idx + 1,
          num_epochs,
          cur_num_steps,
          step_per_seconds,
          ratio_eval_time,
          train_loss,
          extra_snippet,
      )
      epoch_sum_train_loss = jnp.array(0)
      epoch_num_batches = 0
      next_rapid_log = time.time() + rapid_log_interval

      if valid_loss is not None:
        if best_valid_loss is None or valid_loss < best_valid_loss:
          best_valid_loss = valid_loss
          best_epoch_idx = epoch_idx
          if early_stopping_revert_params:
            best_model_state = model_state

        if early_stopping_epoch_patience is not None:
          if epoch_idx - best_epoch_idx >= early_stopping_epoch_patience:
            log.info(
                "The loss did not improve for %d epochs. Stopping training",
                early_stopping_epoch_patience,
            )
            break

    end_training = time.time()
    log.info("Training done in %ss", end_training - begin_training)
    if best_valid_loss is not None:
      log.debug(
          "The best loss %g was observed at epoch %d",
          best_valid_loss,
          best_epoch_idx + 1,
      )
      if best_model_state is not None and best_epoch_idx != num_epochs:
        log.debug("Restoring model state from best loss")
        model_state = best_model_state

    model.set_model_state(model_state)
    return model

  def _evaluation_snippet(
      self,
      metric_computer: "MetricComputer",
      ds_generator: generator_lib.BatchedExampleGenerator,
      valid_ds_generator: Optional[generator_lib.BatchedExampleGenerator],
      model_state: PyTree,
  ) -> Tuple[str, Optional[float]]:
    """Returns a human readable evaluation snippet and validation loss."""

    metric_values, valid_loss = metric_computer.compute_metrics(
        model_state, ds_generator, valid_ds_generator
    )

    # Generate the text snippet.
    text = ""
    for metric_value, metric_name in zip(
        metric_values, metric_computer.metric_names()
    ):
      text += f" {metric_name}:{metric_value:.4f}"
    return text, valid_loss

  def _decompose_batch(self, batch: Batch) -> Tuple[Batch, jax.Array]:
    """Decomposes a batch into features and labels."""
    batch = batch.copy()
    label = batch.pop(self._label)
    # TODO: Add support for training weights.
    return batch, label

  def validate_hyperparameters(self) -> None:
    pass

  def cross_validation(
      self,
      ds: dataset_lib.InputDataset,
      folds: int = 10,
      bootstrapping: Union[bool, int] = False,
      parallel_evaluations: int = 1,
  ) -> metric_lib.Evaluation:
    raise NotImplementedError  # TODO: Implement.

  def extract_input_feature_names(
      self, ds: dataset_lib.InputDataset
  ) -> List[str]:
    raise NotImplementedError  # TODO: Implement.

  def _infer_dataspec(
      self,
      ds_generator: generator_lib.BatchedExampleGenerator,
  ) -> data_spec_pb2.DataSpecification:
    effective_data_spec_args = self._build_data_spec_args()
    return dataset_lib.infer_dataspec(ds_generator, effective_data_spec_args)


class MetricComputer:
  """Computes metrics of a model during training."""

  def __init__(
      self,
      learner: GenericJaxLearner,
      batch_size: int,
      objective: Objective,
      preprocessor: preprocessor_lib.Preprocessor,
      has_train: bool,
      has_valid: bool,
      apply_model: ApplyModelFn,
  ):
    """Initializes the metric computer.

    Args:
      learner: Learner to evaluate.
      batch_size: Batch size for the example generation.
      objective: Objective + metrics to evaluate.
      preprocessor: Data preprocessor.
      has_train: Will we compute metrics on a training dataset.
      has_valid: Will we compute metrics on a validation dataset.
      apply_model: Jitted apply model function.
    """
    self._learner = learner
    self._batch_size = batch_size
    self._objective = objective
    self._preprocessor = preprocessor
    self._metrics_keys = []
    self._apply_model = apply_model

    self._train_metrics = []
    self._valid_metrics = []

    if has_train:
      for m in objective.metrics:
        self._train_metrics.append(m)
        self._metrics_keys.append(f"train-{m.short_name()}")
    if has_valid:
      self._valid_metrics.append(objective.loss)
      self._metrics_keys.append("valid-loss")
      for m in objective.metrics:
        self._valid_metrics.append(m)
        self._metrics_keys.append(f"valid-{m.short_name()}")

    def compute_train_metric_on_batch(
        model_state: ModelState,
        batch_features: Batch,
        batch_labels: jax.Array,
    ) -> jax.Array:
      return self._compute_metric_on_batch(
          model_state, batch_features, batch_labels, self._train_metrics
      )

    def compute_valid_metric_on_batch(
        model_state: ModelState,
        batch_features: Batch,
        batch_labels: jax.Array,
    ) -> jax.Array:
      return self._compute_metric_on_batch(
          model_state, batch_features, batch_labels, self._valid_metrics
      )

    if self._train_metrics:
      self._jit_compute_train_metric_on_batch = jax.jit(
          compute_train_metric_on_batch
      )
    else:
      self._jit_compute_train_metric_on_batch = None
    if self._valid_metrics:
      self._jit_compute_valid_metric_on_batch = jax.jit(
          compute_valid_metric_on_batch
      )
    else:
      self._jit_compute_valid_metric_on_batch = None

  def _compute_metric_on_batch(
      self,
      model_state: ModelState,
      batch_features: Batch,
      batch_labels: jax.Array,
      metrics: List[deep_metric_lib.Metric],
  ):
    """Computes metrics on a single batch of data."""
    assert metrics
    preds, _ = self._apply_model(
        model_state, batch_features, False, jax.random.PRNGKey(0)
    )
    values = []
    for metric in metrics:
      values.append(metric(batch_labels, preds))
    return jnp.stack(values)

  def metric_names(self) -> List[str]:
    """Names of the metrics returned by `compute_metrics`."""
    return self._metrics_keys

  def compute_metrics(
      self,
      model_state: PyTree,
      train_ds_generator: generator_lib.BatchedExampleGenerator,
      valid_ds_generator: Optional[generator_lib.BatchedExampleGenerator],
  ) -> Tuple[List[float], Optional[float]]:
    """Computes metrics on a training and validation dataset.

    The first call to this function will be slow as it trigger the compilation
    of JAX functions.

    Args:
      model_state: State of the model.
      train_ds_generator: Optional training dataset.
      valid_ds_generator: Optional validation dataset.

    Returns:
      The list of metrics (corresponding to "metric_names") and the valid loss.
    """

    # TODO Use a subset of the training dataset for evaluation.

    def compute_metric_on_generator(
        metric_fn: Callable[..., jax.Array],
        generator: generator_lib.BatchedExampleGenerator,
    ) -> List[float]:
      num_examples = 0
      sum_metric_values = None

      # Note: We don't use a simple for-in in order to monitor the data
      # ingession step.
      ds_iterator = generator.generate(
          batch_size=self._batch_size, shuffle=False
      )
      step_idx = 0
      while True:
        with jax.profiler.StepTraceAnnotation("evaluate", step_num=step_idx):
          with jax.profiler.TraceAnnotation("generate_data"):
            # Note: Dataset reading is taking the main thread, while the
            # training operations are async.
            # TODO: Use async for the dataset reading.
            raw_numpy_batch = next(ds_iterator, None)
            if raw_numpy_batch is None:
              break

          with jax.profiler.TraceAnnotation("prepare_data"):
            with jax.profiler.TraceAnnotation("apply_premodel"):
              numpy_batch = self._preprocessor.apply_premodel(
                  raw_numpy_batch, has_labels=True
              )
            with jax.profiler.TraceAnnotation("batch_numpy_to_jax"):
              jax_batch = deep_dataset_lib.batch_numpy_to_jax(numpy_batch)
            with jax.profiler.TraceAnnotation("decompose_batch"):
              batch_features, batch_labels = self._learner._decompose_batch(  # pylint: disable=protected-access
                  jax_batch
              )
            num_examples_in_batch = deep_dataset_lib.get_num_examples(
                numpy_batch
            )

          with jax.profiler.TraceAnnotation("apply_model_and_compute_metric"):
            metric_values = metric_fn(model_state, batch_features, batch_labels)
            if sum_metric_values is None:
              sum_metric_values = metric_values * num_examples_in_batch
            else:
              sum_metric_values += metric_values * num_examples_in_batch
            num_examples += num_examples_in_batch

      if num_examples == 0:
        raise ValueError("No examples to evaluate")
      return (sum_metric_values / num_examples).tolist()

    if self._jit_compute_train_metric_on_batch is not None:
      train_metric_values = compute_metric_on_generator(
          self._jit_compute_train_metric_on_batch, train_ds_generator
      )
    else:
      train_metric_values = []

    if self._jit_compute_valid_metric_on_batch is not None:
      assert valid_ds_generator is not None
      valid_metric_values = compute_metric_on_generator(
          self._jit_compute_valid_metric_on_batch, valid_ds_generator
      )
      valid_loss = valid_metric_values[0]
    else:
      valid_metric_values = []
      valid_loss = None

    return train_metric_values + valid_metric_values, valid_loss
