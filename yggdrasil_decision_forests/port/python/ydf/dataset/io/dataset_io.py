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

import dataclasses
import math
import re
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ydf.dataset.io import dataset_io_types
from ydf.dataset.io import pandas_io
from ydf.dataset.io import tensorflow_io


def unrolled_feature_names(name: str, num_dims: int) -> Sequence[str]:
  """Returns the names of an unrolled feature."""

  if num_dims <= 0:
    raise ValueError("num_dims should be strictly positive.")

  # For example:
  #   num_features=1 => num_leading_zeroes = 1
  #   num_features=9 => num_leading_zeroes = 1
  #   num_features=10 => num_leading_zeroes = 2
  num_leading_zeroes = int(math.log10(num_dims)) + 1

  postfix = f"_of_{num_dims:0{num_leading_zeroes}}"

  return [
      f"{name}.{dim_idx:0{num_leading_zeroes}}{postfix}"
      for dim_idx in range(num_dims)
  ]


def parse_unrolled_feature_name(name: str) -> Optional[Tuple[str, int, int]]:
  """Splits the components of an unrolled feature name."""
  match = re.fullmatch(
      r"(?P<base>.*)\.(?P<idx>[0-9]+)_of_(?P<num>[0-9]+)", name
  )
  if match is None:
    return None
  return match["base"], int(match["idx"]), int(match["num"])


def _unroll_column(
    name: str, src: dataset_io_types.InputValues, allow_unroll: bool
) -> Iterator[Tuple[str, dataset_io_types.InputValues, bool]]:
  """Unrolls a possibly multi-dim. column into multiple single-dim columns.

  Yield the results. If the "src" column is not multi-dimensional, yields "src"
  directly. Fails if "src" contains more than two dimensions.

  Args:
    name: Name of the source column.
    src: Single-dimensional or multi-dimensional value.
    allow_unroll: If false, fails if the column should be unrolled.

  Yields:
    Tuple of key and values of single-dimentional features, and boolean
    indicating if the feature is unrolled.
  """

  # Numpy is currently the only way to pass multi-dim features.
  if not isinstance(src, np.ndarray) or src.ndim <= 1:
    yield name, src, False
    return

  if not allow_unroll:
    raise ValueError(
        f"The column {name!r} is multi-dimensional (shape={src.shape}) while"
        " the model requires this column to be single-dimensional (e.g."
        " shape=[num_examples])."
    )

  if src.ndim > 2:
    raise ValueError(
        "Input features can only be one or two dimensional. Feature"
        f" {name!r} has {src.ndim} dimensions."
    )

  num_features = src.shape[1]
  if num_features == 0:
    raise ValueError(f"Multi-dimention feature {name!r} has no features.")

  sub_names = unrolled_feature_names(name, num_features)
  for dim_idx, sub_name in enumerate(sub_names):
    yield sub_name, src[:, dim_idx], True


def _unroll_dict(
    src: dataset_io_types.DictInputValues,
    dont_unroll_columns: Optional[Sequence[str]] = None,
) -> Tuple[
    dataset_io_types.DictInputValues, dataset_io_types.UnrolledFeaturesInfo
]:
  """Unrolls multi-dim. columns into multiple single-dim. columns.

  Args:
    src: Dictionary of single and multi-dim values.
    dont_unroll_columns: List of columns that cannot be unrolled. If one such
      column needs to be unrolled, raise an error.

  Returns:
    Dictionary containing only single-dimensional values.
  """

  # Index the columns for fast query.
  dont_unroll_columns_set = (
      set(dont_unroll_columns) if dont_unroll_columns else set()
  )

  unrolled_features_info = {}

  # Note: We only create a one dictionary independently of the number of
  # features.
  dst = {}
  for name, value in src.items():

    any_is_unrolling = False
    sub_dst = {}
    for sub_name, sub_value, is_unrolling in _unroll_column(
        name, value, allow_unroll=name not in dont_unroll_columns_set
    ):
      sub_dst[sub_name] = sub_value
      any_is_unrolling |= is_unrolling

    dst.update(sub_dst)

    if any_is_unrolling:
      unrolled_features_info[name] = list(sub_dst.keys())

  return dst, unrolled_features_info


def cast_input_dataset_to_dict(
    data: dataset_io_types.IODataset,
    dont_unroll_columns: Optional[Sequence[str]] = None,
) -> Tuple[
    dataset_io_types.DictInputValues, dataset_io_types.UnrolledFeaturesInfo
]:
  """Normalizes the input dataset into a dictionary of values.

    Also unrolls the multi-dimentional features.

  Args:
    data: Input data.
    dont_unroll_columns: Column in "dont_unroll_columns" will not be unrolled.

  Returns:
    The normalized features, and information about unrolled features.
  """

  unroll_dict_kwargs = {
      "dont_unroll_columns": dont_unroll_columns,
  }

  if pandas_io.is_pandas_dataframe(data):
    return _unroll_dict(pandas_io.to_dict(data), **unroll_dict_kwargs)
  elif tensorflow_io.is_tensorflow_dataset(data):
    return _unroll_dict(tensorflow_io.to_dict(data), **unroll_dict_kwargs)

  elif isinstance(data, dict):
    # Dictionary of values
    return _unroll_dict(data, **unroll_dict_kwargs)

  # TODO: Maybe this error should be raised at a layer above this one?

  if isinstance(data, np.ndarray):
    raise ValueError(
        "Unsupported dataset type:"
        f" {type(data)}\n{dataset_io_types.HOW_TO_FEED_NUMPY}\n\n{dataset_io_types.SUPPORTED_INPUT_DATA_DESCRIPTION}"
    )
  else:
    raise ValueError(
        "Unsupported dataset type: "
        f"{type(data)}\n{dataset_io_types.SUPPORTED_INPUT_DATA_DESCRIPTION}"
    )
