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
from typing import Any, Sequence, Dict, Optional

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
