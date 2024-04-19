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

"""Utilities to export JAX models."""

import dataclasses
from typing import Any, Sequence, Dict, Optional, List, Set

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from ydf.dataset import dataspec as dataspec_lib
from ydf.model import generic_model

# pytype: disable=import-error
# pylint: disable=g-import-not-at-top
try:
  import jax.numpy as jnp
  import jax
except ImportError as exc:
  raise ImportError(
      "JAX is needed for this operation. Install JAX following"
      " https://jax.readthedocs.io/en/latest/installation.html and try again."
  ) from exc
# pylint: enable=g-import-not-at-top
# pytype: enable=import-error


def compact_dtype(values: Sequence[int]) -> Any:
  """Selects the most compact dtype to represent a list of signed integers.

  Only supports: int{8, 16, 32}.

  Note: Jax operations between unsigned and signed integers can be expensive.

  Args:
    values: List of integer values.

  Returns:
    Dtype compatible with all the values.
  """

  if not values:
    raise ValueError("No values provided")

  min_value = min(values)
  max_value = max(values)

  for candidate in [jnp.int8, jnp.int16, jnp.int32]:
    info = jnp.iinfo(candidate)
    if min_value >= info.min and max_value <= info.max:
      return candidate
  raise ValueError("No supported compact dtype")


def to_compact_jax_array(values: Sequence[int]) -> jax.Array:
  """Converts a list of integers to a compact Jax array."""

  return jnp.asarray(values, dtype=compact_dtype(values))


@dataclasses.dataclass
class FeatureEncoding:
  """Utility to prepare feature values before being fed into the Jax model.

  Does the following:
  - Encodes categorical strings into categorical integers.

  Attributes:
    categorical: Mapping between categorical-string feature to the dictionary of
      categorical-string value to categorical-integer value.
    categorical_out_of_vocab_item: Integer value representing an out of
      vocabulary item.
  """

  categorical: Dict[str, Dict[str, int]]
  categorical_out_of_vocab_item: int = 0

  @classmethod
  def build(
      cls,
      input_features: Sequence[generic_model.InputFeature],
      dataspec: ds_pb.DataSpecification,
  ) -> Optional["FeatureEncoding"]:
    """Creates a FeatureEncoding object.

    If the input feature does not require feature encoding, returns None.

    Args:
      input_features: All the input features of a model.
      dataspec: Dataspec of the model.

    Returns:
      A FeatureEncoding or None.
    """

    categorical = {}
    for input_feature in input_features:
      column_spec = dataspec.columns[input_feature.column_idx]
      if (
          input_feature.semantic
          in [
              dataspec_lib.Semantic.CATEGORICAL,
              dataspec_lib.Semantic.CATEGORICAL_SET,
          ]
          and not column_spec.categorical.is_already_integerized
      ):
        categorical[input_feature.name] = {
            key: item.index
            for key, item in column_spec.categorical.items.items()
        }
    if not categorical:
      return None
    return FeatureEncoding(categorical=categorical)

  def encode(self, feature_values: Dict[str, Any]) -> Dict[str, jax.Array]:
    """Encodes feature values for a model."""

    def encode_item(key: str, value: Any) -> jax.Array:
      categorical_map = self.categorical.get(key)
      if categorical_map is not None:
        # Categorical string encoding.
        value = [
            categorical_map.get(x, self.categorical_out_of_vocab_item)
            for x in value
        ]
      return jax.numpy.asarray(value)

    return {k: encode_item(k, v) for k, v in feature_values.items()}


@dataclasses.dataclass
class InternalFeatureValues:
  """Internal representation of feature values.

  In the internal model format, features with the same semantic are grouped
  together i.e. densified.
  """

  numerical: jax.Array
  categorical: jax.Array
  boolean: jax.Array


@dataclasses.dataclass
class InternalFeatureSpec:
  """Spec of the internal feature value representation.

  Attributes:
    input_features: Input features of the model.
    numerical: Name of numerical features in internal order.
    categorical: Name of categorical features in internal order.
    boolean: Name of boolean features in internal order.
    inv_numerical: Column idx to internal idx mapping for numerical features.
    inv_categorical: Column idx to internal idx mapping for categorical features
    inv_boolean: Column idx to internal idx mapping for boolean features.
    feature_names: Name of all the input features.
  """

  input_features: dataclasses.InitVar[Sequence[generic_model.InputFeature]]

  numerical: List[str] = dataclasses.field(default_factory=list)
  categorical: List[str] = dataclasses.field(default_factory=list)
  boolean: List[str] = dataclasses.field(default_factory=list)

  inv_numerical: Dict[int, int] = dataclasses.field(default_factory=dict)
  inv_categorical: Dict[int, int] = dataclasses.field(default_factory=dict)
  inv_boolean: Dict[int, int] = dataclasses.field(default_factory=dict)

  feature_names: Set[str] = dataclasses.field(default_factory=set)

  def __post_init__(self, input_features: Sequence[generic_model.InputFeature]):
    for input_feature in input_features:
      self.feature_names.add(input_feature.name)
      if input_feature.semantic == dataspec_lib.Semantic.NUMERICAL:
        self.inv_numerical[input_feature.column_idx] = len(self.numerical)
        self.numerical.append(input_feature.name)

      elif input_feature.semantic == dataspec_lib.Semantic.CATEGORICAL:
        self.inv_categorical[input_feature.column_idx] = len(self.categorical)
        self.categorical.append(input_feature.name)

      elif input_feature.semantic == dataspec_lib.Semantic.BOOLEAN:
        self.inv_boolean[input_feature.column_idx] = len(self.boolean)
        self.boolean.append(input_feature.name)

      else:
        raise ValueError(
            f"The semantic of feature {input_feature} is not supported by the"
            " YDF to Jax exporter"
        )

  def convert_features(
      self, feature_values: Dict[str, jax.Array]
  ) -> InternalFeatureValues:
    """Converts user provided user values into the internal model format.

    Args:
      feature_values: User input features.

    Returns:
      Internal feature values.
    """

    if not feature_values:
      raise ValueError("At least one feature should be provided")

    batch_size = next(iter(feature_values.values())).shape[0]

    if set(feature_values) != self.feature_names:
      raise ValueError(
          f"Expecting values with keys {set(self.feature_names)!r}. Got"
          f" {set(feature_values.keys())!r}"
      )

    def stack(features, dtype):
      if not features:
        return jnp.zeros(shape=[batch_size, 0], dtype=dtype)
      return jnp.stack(
          [feature_values[feature] for feature in features],
          dtype=dtype,
          axis=1,
      )

    return InternalFeatureValues(
        numerical=stack(self.numerical, jnp.float32),
        categorical=stack(self.categorical, jnp.int32),
        boolean=stack(self.boolean, jnp.bool_),
    )
