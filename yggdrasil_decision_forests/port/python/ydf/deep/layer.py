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

"""Implementation of standard neural nets layers."""

import dataclasses
import enum
from typing import List, Optional, Tuple
from flax import linen as nn
import jax
import jax.numpy as jnp


class FeatureType(enum.Enum):
  """The semantic of a feature ready to be consumed by a neural network.

  UNKNOWN: A value with a semantic unknown by the YDF logic. Used for advanced
    user custom models.
  NUMERICAL: A numerical value stored as a floating point value that is
    normalized for neural network consumption (e.g., zscore, quantile
    normalization).
  CATEGORICAL: A categorical value stored as an integer.
  BOOLEAN: A special type of CATEGORICAL with only two values are stored as a
    boolean.
  """

  UNKNOWN = 0
  NUMERICAL = 1
  CATEGORICAL = 2
  BOOLEAN = 3


@dataclasses.dataclass
class Feature:
  """The description of a feature ready to be consumed by a neural network.

  Attributes:
    name: Name of the feature.
    type: Type of the feature.
    num_categorical_values: For categorical features only. Number of possible
      values i.e. the categorical values are in [0, num_categorical_values).
  """

  name: str
  type: FeatureType
  num_categorical_values: Optional[int] = None

  def __post_init__(self):
    if (
        self.type == FeatureType.CATEGORICAL
        and self.num_categorical_values is None
    ):
      raise ValueError(
          "Categorical features require num_categorical_values to be set"
      )
    if (
        self.type != FeatureType.CATEGORICAL
        and self.num_categorical_values is not None
    ):
      raise ValueError(
          "Only categorical features require num_categorical_values to be set"
      )


@dataclasses.dataclass
class StandardFeatureFlattener(nn.Module):
  """A layer than flaten features into a fixed-size numerical array.

  Attributes:
    categorical_embedding_size: Number of dimensions of the embedding used to
      consume CATEGORICAL features.
  """

  categorical_embedding_size: int = 20

  @nn.compact
  def __call__(self, x: List[Tuple[Feature, jax.Array]]) -> jax.Array:
    """Flattens all the input features into a fixed-size numerical array."""

    input_layer = []

    def ensure_shape2(v: jax.Array) -> jax.Array:
      if len(v.shape) == 1:
        v = jnp.expand_dims(v, axis=1)
      return v

    for feature, value in x:
      if feature.type == FeatureType.NUMERICAL:
        input_layer.append(ensure_shape2(value))
      elif feature.type == FeatureType.BOOLEAN:
        input_layer.append(ensure_shape2(value))
      elif feature.type == FeatureType.CATEGORICAL:
        input_layer.append(
            nn.Embed(
                num_embeddings=feature.num_categorical_values,
                features=self.categorical_embedding_size,
                name=f"embedding_{feature.name}",
            )(value)
        )
      else:
        raise ValueError(
            f"The input feature {feature} is not supported by this layer."
        )
    return jnp.concatenate(input_layer, axis=1, dtype=jnp.float32)
