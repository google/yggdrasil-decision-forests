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

"""Multi-Layer Perceptron."""

import dataclasses
from typing import Dict, Optional, Set
from flax import linen as nn
import jax
from yggdrasil_decision_forests.dataset import data_spec_pb2
from ydf.dataset import dataspec as dataspec_lib
from ydf.deep import deep_model_pb2
from ydf.deep import generic_jax
from ydf.deep import hyperparameter as hyperparameter_lib
from ydf.deep import layer as layer_lib
from ydf.learner import abstract_feature_selector as abstract_feature_selector_lib
from ydf.learner import generic_learner
from ydf.learner import tuner as tuner_lib
from ydf.model import generic_model
from ydf.utils import func_helpers

_MODEL_AND_LEARNER_KEY = "MULTILAYER_PERCEPTRON"

# Hyperparameters
_HP_NUM_LAYERS = "num_layers"
_HP_LAYER_SIZE = "layer_size"
_HP_DROP_OUT = "drop_out"


# TODO: Add support for batch-norm and residual connections.


class MultiLayerPerceptronModel(generic_jax.GenericJAXModel):
  """Multi-Layer Perceptron model."""

  config: Optional["MultiLayerPerceptronImpl.Config"] = None

  @classmethod
  def name(cls) -> str:
    return _MODEL_AND_LEARNER_KEY

  def _build_proto_config(self, model_proto: deep_model_pb2.DeepModel) -> None:
    if self.config is None:
      raise ValueError("Empty configuration")
    model_proto.Extensions[deep_model_pb2.mlp_config].CopyFrom(
        self.config._to_proto()
    )

  def set_config_from_hyperparameters(
      self, hps: hyperparameter_lib.HyperparameterConsumer
  ) -> None:
    self.config = MultiLayerPerceptronImpl.Config(
        num_layers=hps.get_int(_HP_NUM_LAYERS),
        layer_size=hps.get_int(_HP_LAYER_SIZE),
        drop_out=hps.get_float(_HP_DROP_OUT),
    )

  def set_config_from_proto(self, config_proto: deep_model_pb2.MLP):
    self.config = MultiLayerPerceptronImpl.Config._from_proto(config_proto)

  def make_jax_module(self):
    return MultiLayerPerceptronImpl(model=self, config=self.config)


class MultiLayerPerceptronImpl(nn.Module):
  """Multi-Layer Perceptron implementation."""

  @dataclasses.dataclass
  class Config:
    """Configuration objects for the MLP.

    Attributes:
      num_layers: Number of hidden layers.
      layer_size: Number of neurons in each hidden layer.
      drop_out: Dropout rate.
    """
    # LINT.IfChange(MLP)
    num_layers: int
    layer_size: int
    drop_out: float

    def _to_proto(self) -> deep_model_pb2.MLP:
      return deep_model_pb2.MLP(**dataclasses.asdict(self))

    @classmethod
    def _from_proto(
        cls, proto: deep_model_pb2.MLP
    ) -> "MultiLayerPerceptronImpl.Config":
      proto_fields = {
          field_name.name: value for field_name, value in proto.ListFields()
      }
      return cls(**proto_fields)

  model: MultiLayerPerceptronModel
  config: Config

  @nn.compact
  def __call__(self, x: generic_jax.Batch, training: bool) -> jax.Array:
    x = self.model._preprocessor.apply_inmodel(x)
    x = layer_lib.StandardFeatureFlattener()(x)
    for i in range(self.config.num_layers - 1):
      x = nn.Dense(features=self.config.layer_size, name=f"layer_{i}")(x)
      x = nn.relu(x)
      x = nn.Dropout(rate=self.config.drop_out, deterministic=not training)(x)
    x = nn.Dense(features=self.model._output_dim(), name="final_layer")(x)
    return x


class MultiLayerPerceptronLearner(generic_jax.GenericJaxLearner):
  """Multi-Layer Perceptron learner.

  A multi-layer-perceptron (MLP), also known as Feedforward Neural Network (FFN)
  is a simple Neural Network that extends linear models with hidden layers.

  This implementation of MLP includes optional modern updates to the MLP model,
  e.g. batch norm, residual.
  """

  # TODO: Document the hyper-parameters (and fine a way not to
  # repeat the common ones).
  # TODO: Select carefully the default hyper-parameter values.

  def model_class(self) -> generic_jax.GenericJAXModelClass:
    return MultiLayerPerceptronModel

  @func_helpers.list_explicit_arguments
  def __init__(
      self,
      label: Optional[str],
      task: generic_model.Task = generic_learner.Task.CLASSIFICATION,
      *,
      # Jax learner hps
      batch_size: int = 512,
      num_epochs: int = 1000,
      learning_rate: float = 0.01,
      learning_rate_policy: generic_jax.LearningRatePolicy = "cosine_decay",
      num_steps: Optional[int] = None,
      maximum_training_duration_seconds: float = -1.0,
      # MLP hps
      num_layers: int = 8,
      layer_size: int = 200,
      drop_out: float = 0.05,
      random_seed: int = 1234,
      early_stopping_epoch_patience: Optional[int] = 10,
      early_stopping_revert_params: bool = True,
      num_quantiles: int = 1000,
      zscore: bool = False,
      # General hps
      weights: Optional[str] = None,
      class_weights: Optional[Dict[str, float]] = None,
      ranking_group: Optional[str] = None,
      uplift_treatment: Optional[str] = None,
      data_spec: Optional[data_spec_pb2.DataSpecification] = None,
      features: Optional[dataspec_lib.ColumnDefs] = None,
      include_all_columns: bool = False,
      max_vocab_count: int = 2000,
      min_vocab_frequency: int = 5,
      max_num_scanned_rows_to_infer_semantic: int = 100_000,
      max_num_scanned_rows_to_compute_statistics: int = 100_000,
      working_dir: Optional[str] = None,
      num_threads: Optional[int] = None,
      tuner: Optional[tuner_lib.AbstractTuner] = None,
      feature_selector: Optional[
          abstract_feature_selector_lib.AbstractFeatureSelector
      ] = None,
      explicit_args: Optional[Set[str]] = None,
  ):

    hyper_parameters = {
        _HP_NUM_LAYERS: num_layers,
        _HP_LAYER_SIZE: layer_size,
        _HP_DROP_OUT: drop_out,
    }
    super().__init__(
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        learning_rate_policy=learning_rate_policy,
        num_steps=num_steps,
        maximum_training_duration_seconds=maximum_training_duration_seconds,
        random_seed=random_seed,
        early_stopping_epoch_patience=early_stopping_epoch_patience,
        early_stopping_revert_params=early_stopping_revert_params,
        num_quantiles=num_quantiles,
        zscore=zscore,
        learner_name=_MODEL_AND_LEARNER_KEY,
        task=task,
        label=label,
        weights=weights,
        class_weights=class_weights,
        ranking_group=ranking_group,
        uplift_treatment=uplift_treatment,
        data_spec=data_spec,
        features=features,
        include_all_columns=include_all_columns,
        max_vocab_count=max_vocab_count,
        min_vocab_frequency=min_vocab_frequency,
        max_num_scanned_rows_to_infer_semantic=max_num_scanned_rows_to_infer_semantic,
        max_num_scanned_rows_to_compute_statistics=max_num_scanned_rows_to_compute_statistics,
        working_dir=working_dir,
        num_threads=num_threads,
        hyper_parameters=hyper_parameters,
        explicit_learner_arguments=explicit_args,
        tuner=tuner,
        feature_selector=feature_selector,
    )
