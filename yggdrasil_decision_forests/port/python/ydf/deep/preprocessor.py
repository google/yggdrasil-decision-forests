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
from typing import List, Sequence, Tuple
import jax
import numpy as np
from yggdrasil_decision_forests.dataset import data_spec_pb2
from ydf.deep import dataset as deep_dataset_lib
from ydf.deep import layer as layer_lib


@dataclasses.dataclass
class Preprocessor:
  """Process the data before being feed to a neural net.

  The processing is done in two steps:
    apply_premodel: This function operates on Numpy arrays and contains all the
      processing that cannot be expressed in Jax.
    apply_inmodel: This function operates on Jax and contains all the remaining
      processing.

  Attributes:
    dataspec: Definition of the data.
    input_features_col_idxs: Columns in the dataspec that represent input
      features of the model.
  """

  dataspec: data_spec_pb2.DataSpecification
  input_features_col_idxs: Sequence[int]

  def __post_init__(self):
    self._input_features_col_idxs_set = set(self.input_features_col_idxs)

  def apply_premodel(
      self,
      src: deep_dataset_lib.NumpyExampleBatch,
      has_labels: bool = False,
  ) -> deep_dataset_lib.NumpyExampleBatch:
    """Applies the first step of the processing."""

    dst = {}
    for column_idx, column in enumerate(self.dataspec.columns):
      is_label = column_idx not in self._input_features_col_idxs_set
      if not has_labels and is_label:
        continue
      src_values = src[column.name]
      if column.type != data_spec_pb2.ColumnType.CATEGORICAL:
        # Simply pass the value
        dst[column.name] = src_values
      else:
        dst[column.name] = np.array(
            [
                self._encode_categorical_string_value(
                    v, column.categorical, is_label=is_label
                )
                for v in src_values.astype(np.bytes_)
            ],
            dtype=np.int32,
        )
    return dst

  def apply_inmodel(
      self,
      src: deep_dataset_lib.JaxExampleBatch,
      has_labels: bool = False,
  ) -> List[Tuple[layer_lib.Feature, jax.Array]]:
    """Applies the second step of the processing."""

    dst = []
    for column_idx, column in enumerate(self.dataspec.columns):
      is_label = column_idx not in self._input_features_col_idxs_set
      if not has_labels and is_label:
        continue
      src_values = src[column.name]
      if column.type == data_spec_pb2.ColumnType.NUMERICAL:
        dst.append((
            layer_lib.Feature(
                f"{column.name}_ZSCORE", layer_lib.FeatureType.NUMERICAL
            ),
            (src[column.name] - column.numerical.mean)
            / column.numerical.standard_deviation,
        ))
        # TODO: Implement quantile normalization
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

  def _encode_categorical_string_value(
      self,
      value: str,
      categorical_col_spec: data_spec_pb2.CategoricalSpec,
      is_label: bool,
  ) -> int:

    item = categorical_col_spec.items.get(value, None)
    if item is None:
      if is_label:
        raise ValueError("Out-of-dictionary value is allowed for labels")
      return 0  # Out of dictionary
    if is_label:
      return item.index - 1  # Skip of the OOD item
    else:
      return item.index
