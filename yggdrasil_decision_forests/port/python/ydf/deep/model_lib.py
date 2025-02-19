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

"""Utilities for loading deep YDF models."""

import safetensors
import safetensors.flax
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.deep import deep_model_pb2
from ydf.deep import generic_jax
from ydf.deep import mlp
from ydf.deep import preprocessor as preprocessor_lib
from ydf.deep import safetensors as safetensors_lib
from ydf.deep import tabular_transformer
from ydf.model import generic_model
from ydf.utils import filesystem


def load_model(
    directory: str,
    advanced_options: generic_model.ModelIOOptions = generic_model.ModelIOOptions(),  # pylint:disable=unused-argument
) -> generic_model.ModelType:
  """Loads a model from disk."""
  weights_file = filesystem.Path(directory) / generic_jax._WEIGHTS_FILE_NAME  # pylint:disable=protected-access
  abstract_model_file = (
      filesystem.Path(directory) / generic_jax._ABSTRACT_MODEL_FILE_NAME  # pylint:disable=protected-access
  )
  data_spec_file = filesystem.Path(directory) / generic_jax._DATA_SPEC_FILE_NAME  # pylint:disable=protected-access
  config_file = (
      filesystem.Path(directory) / generic_jax._DEEP_MODEL_PROTO_FILE_NAME  # pylint:disable=protected-access
  )
  done_file = filesystem.Path(directory) / generic_jax._DONE_FILE_NAME  # pylint:disable=protected-access

  if not done_file.exists():
    raise ValueError(f'The model is missing a "done" file at {done_file}')

  with filesystem.Open(data_spec_file, "rb") as f:
    dataspec = data_spec_pb2.DataSpecification.FromString(f.read())
  with filesystem.Open(abstract_model_file, "rb") as f:
    abstract_model_proto = abstract_model_pb2.AbstractModel.FromString(f.read())
  with filesystem.Open(config_file, "rb") as f:
    deep_model_proto = deep_model_pb2.DeepModel.FromString(f.read())
  with filesystem.Open(weights_file, "rb") as f:
    weights = safetensors.flax.load(f.read())

  preprocessor = preprocessor_lib.Preprocessor.build(
      deep_model_proto.preprocessor, abstract_model_proto, dataspec
  )
  # TODO: Implement model registration.
  if abstract_model_proto.name == tabular_transformer._MODEL_AND_LEARNER_KEY:  # pylint:disable=protected-access
    model = tabular_transformer.TabularTransformerModel(
        dataspec, preprocessor, abstract_model_proto
    )
    model.set_config_from_proto(
        deep_model_proto.Extensions[deep_model_pb2.tabular_transformer_config]
    )
  elif abstract_model_proto.name == mlp._MODEL_AND_LEARNER_KEY:  # pylint:disable=protected-access
    model = mlp.MultiLayerPerceptronModel(
        dataspec, preprocessor, abstract_model_proto
    )
    model.set_config_from_proto(
        deep_model_proto.Extensions[deep_model_pb2.mlp_config]
    )
  else:
    raise ValueError(f"Invalid model name {abstract_model_proto.name}")

  if deep_model_proto.weights.format == deep_model_pb2.Weights.SAFETENSORS:
    weights = safetensors_lib.deflatten_weights(weights)
  else:
    raise ValueError(f"Unknown weight format {deep_model_proto.weights.format}")
  model.set_model_state(weights)

  return model
