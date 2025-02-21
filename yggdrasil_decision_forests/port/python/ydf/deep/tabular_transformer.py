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

"""Tabular Transformer."""

import dataclasses
from typing import Dict, List, Optional, Set, Tuple
from flax import linen as nn
import jax
import jax.numpy as jnp
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
from ydf.utils import log

_MODEL_AND_LEARNER_KEY = "TABULAR_TRANSFORMER"


# Hyperparameters
_HP_NUM_LAYERS = "num_layers"
_HP_DROP_OUT = "drop_out"
_HP_TOKEN_DIM = "token_dim"
_HP_NUM_HEADS = "num_heads"
_HP_QKV_FEATURES = "qkv_features"


class TabularTransformerModel(generic_jax.GenericJAXModel):
  """Tabular Transformer model."""

  config: Optional["TabularTransformerImpl.Config"] = None

  @classmethod
  def name(cls) -> str:
    return _MODEL_AND_LEARNER_KEY

  def _build_proto_config(self, model_proto: deep_model_pb2.DeepModel) -> None:
    if self.config is not None:
      model_proto.Extensions[
          deep_model_pb2.tabular_transformer_config
      ].CopyFrom(self.config._to_proto())

  def set_config_from_hyperparameters(
      self, hps: hyperparameter_lib.HyperparameterConsumer
  ) -> None:
    self.config = TabularTransformerImpl.Config(
        num_layers=hps.get_int(_HP_NUM_LAYERS),
        drop_out=hps.get_float(_HP_DROP_OUT),
        num_heads=hps.get_int(_HP_NUM_HEADS),
        qkv_features=hps.get_int(_HP_QKV_FEATURES),
        tokenizer=FTTransformerTokenizer.Config(
            token_dim=hps.get_int(_HP_TOKEN_DIM)
        ),
    )

  def make_jax_module(self):
    return TabularTransformerImpl(model=self, config=self.config)

  def set_config_from_proto(
      self, config_proto: deep_model_pb2.TabularTransformer
  ) -> None:
    self.config = TabularTransformerImpl.Config._from_proto(config_proto)


class TabularTransformerImpl(nn.Module):
  """Tabular Transformer implementation."""

  @dataclasses.dataclass
  class Config:
    """Configuration objects for the Tabular Transformer.

    Attributes:
      num_layers: Number of attention layers.
      drop_out: Dropout rate.
      num_heads: Number of attention heads per layer.
      qkv_features: Dimension of the key, query, and value inside the attention
        module.
      tokenizer: Configuration of the Tokenizer
    """
    # LINT.IfChange(TabularTransformer)
    num_layers: int
    drop_out: float
    num_heads: int
    qkv_features: int
    tokenizer: "FTTransformerTokenizer.Config"

    def _to_proto(self) -> deep_model_pb2.TabularTransformer:
      return deep_model_pb2.TabularTransformer(
          num_layers=self.num_layers,
          drop_out=self.drop_out,
          num_heads=self.num_heads,
          qkv_features=self.qkv_features,
          ft_tokenizer=deep_model_pb2.FTTokenizer(
              token_dim=self.tokenizer.token_dim
          ),
      )

    @classmethod
    def _from_proto(cls, config_proto: deep_model_pb2.TabularTransformer):
      return TabularTransformerImpl.Config(
          num_layers=config_proto.num_layers,
          drop_out=config_proto.drop_out,
          num_heads=config_proto.num_heads,
          qkv_features=config_proto.qkv_features,
          tokenizer=FTTransformerTokenizer.Config(
              token_dim=config_proto.ft_tokenizer.token_dim
          ),
      )

  model: TabularTransformerModel
  config: Config

  @nn.compact
  def __call__(self, x: generic_jax.Batch, training: bool) -> jax.Array:
    batch_norm = lambda z, name: nn.BatchNorm(
        use_running_average=not training, name=name
    )(z)

    with jax.profiler.TraceAnnotation("preprocess"):
      x = self.model._preprocessor.apply_inmodel(x)

    with jax.profiler.TraceAnnotation("tokenize"):
      x = FTTransformerTokenizer(config=self.config.tokenizer)(x)
      assert len(x.shape) == 3

    for i in range(self.config.num_layers):
      save_x = x  # For the residual
      with jax.profiler.TraceAnnotation("batch_norm"):
        x = batch_norm(x, name=f"layer_{i}_batchnorm_1")
      with jax.profiler.TraceAnnotation("attention"):
        x = nn.MultiHeadDotProductAttention(
            qkv_features=self.config.qkv_features,
            num_heads=self.config.num_heads,
            dropout_rate=self.config.drop_out,
            deterministic=not training,
            name=f"layer_{i}_selfattention",
        )(x)
      x = x + save_x
      assert len(x.shape) == 3

      with jax.profiler.TraceAnnotation("dense"):
        save_x = x
        x = batch_norm(x, name=f"layer_{i}_batchnorm_2")
        x = nn.Dense(
            features=self.config.tokenizer.token_dim, name=f"layer_{i}_dense_1"
        )(x)
        x = nn.gelu(x)
        x = nn.Dense(
            features=self.config.tokenizer.token_dim, name=f"layer_{i}_dense_2"
        )(x)
        x = x + save_x
        assert len(x.shape) == 3

    with jax.profiler.TraceAnnotation("final"):
      x = x[:, 0, :]
      x = batch_norm(x, name="final_layer_batchnorm")
      x = nn.gelu(x)
      x = nn.Dense(features=self.model._output_dim(), name="final_layer")(x)
    return x


class FTTransformerTokenizer(nn.Module):
  """Tokenizer input features following the FT-Transformer algorithm.

  Attributes:
    token_dim: Number of dimensions of the feature embeddings.
  """

  @dataclasses.dataclass
  class Config:
    """Configuration objects for the FT-Transformer tokenizer.

    Attributes:
      token_dim: Number of dimensions of the feature embeddings.
    """

    # LINT.IfChange(FTTokenizer)
    token_dim: int = 192

  config: Config

  @nn.compact
  def __call__(self, x: List[Tuple[layer_lib.Feature, jax.Array]]) -> jax.Array:
    def ensure_shape2(v: jax.Array) -> jax.Array:
      if len(v.shape) == 1:
        v = jnp.expand_dims(v, axis=1)
      return v

    # List of feature tokens. Each token is of shape [batch, n, config.token_dim]
    tokens: List[jax.Array] = []

    # Encode numerical features
    numerical_like_features = []
    for feature, value in x:
      if feature.type in [
          layer_lib.FeatureType.NUMERICAL,
          layer_lib.FeatureType.BOOLEAN,
      ]:
        numerical_like_features.append(ensure_shape2(value))
    if numerical_like_features:
      numerical_values = jnp.concatenate(
          numerical_like_features, axis=1, dtype=jnp.float32
      )
      numerical_kernel = self.param(
          "numerical_kernel",
          nn.initializers.kaiming_uniform(),
          (
              numerical_values.shape[1],
              self.config.token_dim,
          ),
      )
      numerical_tokens = jnp.einsum(
          "bf,fd->bfd", numerical_values, numerical_kernel
      )
      assert len(numerical_tokens.shape) == 3
      tokens.append(numerical_tokens)

    # Encode categorical features
    for feature, value in x:
      if feature.type == layer_lib.FeatureType.CATEGORICAL:
        value = ensure_shape2(value)
        embedded_value = nn.Embed(
            num_embeddings=feature.num_categorical_values,
            features=self.config.token_dim,
            name=f"embedding_{feature.name}",
        )(value)
        if len(embedded_value.shape) == 2:
          embedded_value = jnp.expand_dims(embedded_value, axis=1)
        assert len(embedded_value.shape) == 3
        tokens.append(embedded_value)

    if not tokens:
      raise ValueError("The model has not input features")

    # Start token
    batch_size = tokens[0].shape[0]
    cls_token = jnp.zeros(
        (batch_size, 1, self.config.token_dim), dtype=jnp.float32
    )

    # Group tokens
    token_array = jnp.concatenate([cls_token] + tokens, axis=1)

    # Add bias
    bias_kernel = self.param(
        "bias_kernel", nn.initializers.kaiming_uniform(), token_array.shape[1:]
    )
    token_array = token_array + jnp.expand_dims(bias_kernel, axis=0)
    return token_array


class TabularTransformerLearner(generic_jax.GenericJaxLearner):
  """Tabular Transformer learner.

  The Tabular Transformer are neural networks models based on the Transformer
  architecture and able to consume tabular data. Those models are slower to
  train and run than classical neural network architectures (e.g., multi-layer
  perceptron) but show better generalization and sample efficiency.

  The default logic and behavior follow the FT-Transformer architecture
  (https://arxiv.org/pdf/2106.11959) by Gorishniy et al.
  """

  # TODO: Document the hyper-parameters (and fine a way not to
  # repeat the common ones).
  # TODO: Select carefully the default hyper-parameter values.
  # TODO: Allow `maximum_training_duration_seconds=None`, here
  # and in all the other learners.

  def model_class(self) -> generic_jax.GenericJAXModelClass:
    return TabularTransformerModel

  @func_helpers.list_explicit_arguments
  def __init__(
      self,
      label: Optional[str],
      task: generic_model.Task = generic_learner.Task.CLASSIFICATION,
      *,
      # Jax learner hps
      batch_size: int = 256,
      num_epochs: int = 1000,
      learning_rate: float = 0.001,
      learning_rate_policy: generic_jax.LearningRatePolicy = "cosine_decay",
      num_steps: Optional[int] = None,
      maximum_training_duration_seconds: float = -1.0,
      # Transformer hps
      num_layers: int = 3,
      drop_out: float = 0.05,
      token_dim: int = 50,
      num_heads: int = 4,
      qkv_features: int = 16,
      random_seed: int = 1234,
      early_stopping_epoch_patience: Optional[int] = 10,
      early_stopping_revert_params: bool = True,
      num_quantiles: int = 1000,
      zscore: bool = False,
      allow_cpu: bool = False,
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

    if jax.devices()[0].platform == "cpu":
      if allow_cpu:
        log.warning(
            "Training a transformer model on CPU will be slow. Make sure the"
            " transformer is small (e.g. small token_dim) or use a CPU/TPU.",
            message_id=log.WarningMessage.TRAIN_TRANSFORMER_ON_CPU,
        )
      else:
        raise ValueError(
            "No GPU/TPU found. To allow for the transformer to train on CPU,"
            " set `TabularTransformerLearner(allow_cpu=True, ...)`."
        )

    hyper_parameters = {
        _HP_NUM_LAYERS: num_layers,
        _HP_DROP_OUT: drop_out,
        _HP_TOKEN_DIM: token_dim,
        _HP_NUM_HEADS: num_heads,
        _HP_QKV_FEATURES: qkv_features,
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
