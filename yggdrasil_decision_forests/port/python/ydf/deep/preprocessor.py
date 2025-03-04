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

"""Utility to pre-process the examples before feeding them to a neural net.

This files notably contains the processing that cannot be expressed in JAX.
"""

import dataclasses
from typing import Dict, List, Sequence, Set, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model import abstract_model_pb2
from ydf.dataset import dataspec as dataspec_lib
from ydf.deep import dataset as deep_dataset_lib
from ydf.deep import deep_model_pb2
from ydf.deep import layer as layer_lib


@dataclasses.dataclass
class CategoricalDictionary:
  """Dictionary for a categorical column.

  Attributes:
    sorted_keys: Item's key sorted in alphabetical order.
    key_order: Order of the items's key i.e., how to go from the dataspec item
      index to the index in `sorted_keys`.
  """

  sorted_keys: np.ndarray
  key_order: np.ndarray


@dataclasses.dataclass
class Preprocessor:
  """Process the data before being fed to a neural net.

  The processing is done in two steps:
    apply_premodel: This function operates on Numpy arrays and contains all the
      processing that cannot be expressed in Jax.
    apply_inmodel: This function operates on Jax and contains all the remaining
      processing.

  Attributes:
    dataspec: Definition of the data.
    input_features_col_idxs: Columns in the dataspec that represent input
      features of the model.
    numerical_zscore: Computes the z-scores of numerical features.
    numerical_quantiles: Computes the quantiles of numerical features.
    input_features_col_idxs_set: Set containing the values of
      "input_features_col_idxs".
    categorical_dicts: Index the key of categorical columns.
  """

  dataspec: data_spec_pb2.DataSpecification
  input_features_col_idxs: Sequence[int]
  # LINT.IfChange(Preprocessor)
  numerical_zscore: bool
  numerical_quantiles: bool
  input_features_col_idxs_set: Set[int] = dataclasses.field(init=False)
  categorical_dicts: Dict[int, CategoricalDictionary] = dataclasses.field(
      init=False
  )

  def __post_init__(self):
    # Index the column indexes
    self.input_features_col_idxs_set = set(self.input_features_col_idxs)

    # Index the dictionaries
    self.categorical_dicts = {}
    for column_idx, column in enumerate(self.dataspec.columns):
      if column.type != data_spec_pb2.ColumnType.CATEGORICAL:
        continue

      items = sorted([
          (item.index, key.decode())
          for key, item in dataspec_lib.categorical_vocab_iterator(
              column.categorical
          )
      ])
      # Item key sorted by index
      keys = np.array([x[1] for x in items]).astype(np.bytes_)

      # Key sorted by value
      key_order = np.argsort(keys).astype(np.int32)
      sorted_key = keys[key_order]
      self.categorical_dicts[column_idx] = CategoricalDictionary(
          sorted_keys=sorted_key,
          key_order=key_order,
      )

  def to_proto(self) -> deep_model_pb2.Preprocessor:
    return deep_model_pb2.Preprocessor(
        numerical_zscore=self.numerical_zscore,
        numerical_quantiles=self.numerical_quantiles,
    )

  @classmethod
  def build(
      cls,
      preprocessor: deep_model_pb2.Preprocessor,
      abstract_model: abstract_model_pb2.AbstractModel,
      dataspec: data_spec_pb2.DataSpecification,
  ) -> "Preprocessor":
    return cls(
        dataspec=dataspec,
        input_features_col_idxs=list(abstract_model.input_features),
        numerical_zscore=preprocessor.numerical_zscore
        if preprocessor.HasField("numerical_zscore")
        else None,
        numerical_quantiles=preprocessor.numerical_quantiles
        if preprocessor.HasField("numerical_quantiles")
        else None,
    )

  def apply_premodel(
      self,
      src: deep_dataset_lib.NumpyExampleBatch,
      has_labels: bool = False,
  ) -> deep_dataset_lib.NumpyExampleBatch:
    """Applies the first step of the processing."""

    dst = {}
    for column_idx, column in enumerate(self.dataspec.columns):
      is_label = column_idx not in self.input_features_col_idxs_set
      if not has_labels and is_label:
        continue
      src_values = src[column.name]
      if column.type != data_spec_pb2.ColumnType.CATEGORICAL:
        # Simply pass the value
        dst[column.name] = src_values
      else:
        # TODO: Speed up in c++.

        src_values = src_values.astype(np.bytes_)
        col_dict = self.categorical_dicts[column_idx]

        # Binary search to find values
        sorted_index = np.minimum(
            np.searchsorted(col_dict.sorted_keys, src_values),
            len(col_dict.sorted_keys) - 1,
        )
        # Handle out-of-dictionary values
        sorted_index = np.where(
            col_dict.sorted_keys[sorted_index] == src_values, sorted_index, 0
        )
        dst_values = col_dict.key_order[sorted_index]
        if is_label:
          dst_values -= 1
        dst[column.name] = dst_values
    return dst

  def apply_inmodel(
      self,
      src: deep_dataset_lib.JaxExampleBatch,
      has_labels: bool = False,
  ) -> List[Tuple[layer_lib.Feature, jax.Array]]:
    """Applies the second step of the processing."""

    dst = []
    for column_idx, column in enumerate(self.dataspec.columns):
      is_label = column_idx not in self.input_features_col_idxs_set
      if not has_labels and is_label:
        continue
      src_values = src[column.name]
      if column.type == data_spec_pb2.ColumnType.NUMERICAL:
        src_values = jnp.nan_to_num(src_values, nan=column.numerical.mean)
        if self.numerical_zscore:
          z_score_value = (
              src_values - column.numerical.mean
          ) / column.numerical.standard_deviation
          dst.append((
              layer_lib.Feature(
                  f"{column.name}_ZSCORE", layer_lib.FeatureType.NUMERICAL
              ),
              z_score_value,
          ))
        if self.numerical_quantiles:
          boundaries = jnp.array(column.discretized_numerical.boundaries)
          bucket_idx = (
              jnp.searchsorted(boundaries, src_values, side="right") - 1
          )
          bucket_idx = jnp.clip(bucket_idx, 0, len(boundaries) - 2)
          lower = boundaries[bucket_idx]
          upper = boundaries[bucket_idx + 1]
          # Linear interpolation
          interpolated_values = bucket_idx + (src_values - lower) / (
              upper - lower
          )
          dst.append((
              layer_lib.Feature(
                  f"{column.name}_QUANTILE", layer_lib.FeatureType.NUMERICAL
              ),
              interpolated_values / (len(boundaries) - 1),
          ))
      elif column.type == data_spec_pb2.ColumnType.CATEGORICAL:
        dst.append((
            layer_lib.Feature(
                column.name,
                layer_lib.FeatureType.CATEGORICAL,
                num_categorical_values=column.categorical.number_of_unique_values
                - (1 if is_label else 0),
            ),
            src_values,
        ))
      elif column.type == data_spec_pb2.ColumnType.BOOLEAN:
        src_values = jnp.nan_to_num(src_values.astype(jnp.float32), nan=0.5)
        dst.append((
            layer_lib.Feature(column.name, layer_lib.FeatureType.BOOLEAN),
            src_values,
        ))
      else:
        dst.append((
            layer_lib.Feature(column.name, layer_lib.FeatureType.UNKNOWN),
            src_values,
        ))
    return dst
