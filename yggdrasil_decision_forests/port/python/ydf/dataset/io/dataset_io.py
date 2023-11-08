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

"""Common functionality for all dataset I/O connectors."""

import math
from typing import Dict, Iterator, Tuple

import numpy as np

from ydf.dataset.io import dataset_io_types
from ydf.dataset.io import pandas_io
from ydf.dataset.io import tensorflow_io


def _unroll_column(
    name: str,
    src: dataset_io_types.InputValues,
) -> Iterator[Tuple[str, dataset_io_types.InputValues]]:
  """Unrolls a possibly multi-dim. column into multiple single-dim columns.

  Yield the results. If the "src" column is not multi-dimensional, yields "src"
  directly. Fails if "src" contains more than two dimensions.

  Args:
    name: Name of the source column.
    src: Single-dimensional or multi-dimensional value.

  Yields:
    Tuple of key and values of single-dimentional features.
  """

  # Numpy is currently the only way to pass multi-dim features.
  if not isinstance(src, np.ndarray) or src.ndim <= 1:
    yield name, src
    return

  if src.ndim > 2:
    raise ValueError(
        "Input features can only be one or two dimensional. Feature"
        f" {name!r} has {src.ndim} dimensions."
    )

  num_features = src.shape[1]
  if num_features == 0:
    raise ValueError(f"Multi-dimention feature {name!r} has no features.")

  # For example:
  #   num_features=1 => num_leading_zeroes = 1
  #   num_features=9 => num_leading_zeroes = 1
  #   num_features=10 => num_leading_zeroes = 2
  num_leading_zeroes = int(math.log10(num_features)) + 1

  postfix = f"_of_{num_features:0{num_leading_zeroes}}"
  for dim_idx in range(num_features):
    sub_name = f"{name}.{dim_idx:0{num_leading_zeroes}}{postfix}"
    yield sub_name, src[:, dim_idx]


def _unroll_dict(
    src: Dict[str, dataset_io_types.InputValues]
) -> Dict[str, dataset_io_types.InputValues]:
  """Unrolls multi-dim. columns into multiple single-dim. columns.

  Args:
    src: Dictionary of single and multi-dim values.

  Returns:
    Dictionary containing only single-dimensional values.
  """

  # Note: We only create a one dictionary independently of the number of
  # features.
  dst = {}
  for name, value in src.items():
    for sub_name, sub_value in _unroll_column(name, value):
      dst[sub_name] = sub_value
  return dst


def cast_input_dataset_to_dict(
    data: dataset_io_types.IODataset,
) -> Dict[str, dataset_io_types.InputValues]:
  """Transforms the input dataset into a dictionary of values."""
  if pandas_io.is_pandas_dataframe(data):
    return _unroll_dict(pandas_io.to_dict(data))
  elif tensorflow_io.is_tensorflow_dataset(data):
    return _unroll_dict(tensorflow_io.to_dict(data))

  elif isinstance(data, dict):
    # Dictionary of values
    return _unroll_dict(data)

  # TODO: Maybe this error should be raised at a layer above this one?
  raise ValueError(
      "Cannot import dataset from"
      f" {type(data)}.\n{dataset_io_types.SUPPORTED_INPUT_DATA_DESCRIPTION}"
  )
