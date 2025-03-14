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
import re
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from ydf.dataset.io import dataset_io_types
from ydf.dataset.io import numpy_io
from ydf.dataset.io import pandas_io
from ydf.dataset.io import polars_io
from ydf.dataset.io import pygrain_io
from ydf.dataset.io import tensorflow_io
from ydf.dataset.io import xarray_io


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
    name: str,
    src: dataset_io_types.InputValues,
    allow_unroll: bool,
    expect_unroll_info: Optional[ds_pb.Unstacked],
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
  if not isinstance(src, np.ndarray):
    yield name, src, False
    return

  if src.ndim <= 1:
    # The data is a numpy array containing objects that are numpy arrays.
    # If the arrays all have the same size, using unrolled multi-dimensional
    # features is better than CATEGORICAL_SET.
    if src.ndim > 0 and src.size > 0 and isinstance(src[0], np.ndarray):
      try:
        # If columns can be stacked, do it to prevent accidental cast to
        # CATEGORICAL_SET.
        src = np.vstack(src)
      except ValueError:
        yield name, src, False
        return
    else:
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
    if expect_unroll_info is not None:
      raise ValueError(
          f"Feature {name!r} is expected as a multi-dimensional feature of"
          f" dimension {expect_unroll_info.size} and type"
          f" {ds_pb.ColumnType.Name(expect_unroll_info.type)}. The given"
          " dataset has a feature of dimension 0. If the feature is expected"
          " to be a variable_length categorical feature, specify semantic"
          " CATEGORICAL_SET during training. Note that variable length"
          " numerical features are not supported by YDF."
      )

    else:
      error_msg = f"""Feature {name!r} is detected as a multi-dimensional fixed-length feature with dimension zero i.e. the feature does not contain any data. This is not allowed. You can:
1) If the feature is effectively a multi-dimensional fixed-length feature, remove it from the training.
2) If the feature is a possibly missing value that is always missing from the training dataset, remove it from the training.
3) If the feature is in fact a variable length categorical feature (a.k.a. Categorical-set feature), specify the feature semantic in the trainer constructor e.g. `features=[("Feature_name", ydf.Semantic.CATEGORICAL_SET)]`.
4) If the feature is in fact a variable length numerical feature, you need to feature engineer it as YDF does not support numerical-set features natively. For example, you can compute feature statistics (e.g., mean, max, sd), use Temporian if the feature is a timeseries, or use the ydf.Semantic.NUMERICAL_VECTOR_SEQUENCE semantic if the feature is a collection of embeddings."""
      raise ValueError(error_msg)

  sub_names = unrolled_feature_names(name, num_features)
  for dim_idx, sub_name in enumerate(sub_names):
    yield sub_name, src[:, dim_idx], True


def _unroll_dict(
    src: dataset_io_types.DictInputValues,
    single_dim_columns: Sequence[str],
    not_unrolled_multi_dim_columns: Sequence[str],
    expected_unrolled_columns: Sequence[ds_pb.Unstacked],
) -> Tuple[
    dataset_io_types.DictInputValues, dataset_io_types.UnrolledFeaturesInfo
]:
  """Unrolls multi-dim. columns into multiple single-dim. columns.

  Args:
    src: Dictionary of single and multi-dim values.
    single_dim_columns: List of columns that should be single-dimensional. If
      one such column is multi-dimensional, raise an error.
    not_unrolled_multi_dim_columns: List of columns that should be
      multi-dimensional and not unrolled.
    expected_unrolled_columns: List of columns that are known to be
      multi-dimensional and unrolled.

  Returns:
    Dictionary containing only single-dimensional values.
  """

  # Index the columns for fast query.
  single_dim_columns_set = set(single_dim_columns)
  not_unrolled_multi_dim_columns_set = set(not_unrolled_multi_dim_columns)
  expected_unrolled_columns_dict = {
      c.original_name: c for c in expected_unrolled_columns
  }

  unrolled_features_info = {}

  # Note: We only create a single dictionary independently of the number of
  # features.
  dst = {}
  for name, value in src.items():
    if name in not_unrolled_multi_dim_columns_set:
      dst[name] = value
      continue

    any_is_unrolling = False
    sub_dst = {}
    expect_unroll_info = expected_unrolled_columns_dict.get(name, None)
    for sub_name, sub_value, is_unrolling in _unroll_column(
        name,
        value,
        allow_unroll=name not in single_dim_columns_set,
        expect_unroll_info=expect_unroll_info,
    ):
      sub_dst[sub_name] = sub_value
      any_is_unrolling |= is_unrolling

    dst.update(sub_dst)

    if any_is_unrolling:
      unrolled_features_info[name] = list(sub_dst.keys())

  return dst, unrolled_features_info


def cast_input_dataset_to_dict(
    data: dataset_io_types.IODataset,
    single_dim_columns: Optional[Sequence[str]] = None,
    not_unrolled_multi_dim_columns: Optional[Sequence[str]] = None,
    expected_unrolled_columns: Optional[Sequence[ds_pb.Unstacked]] = None,
) -> Tuple[
    dataset_io_types.DictInputValues, dataset_io_types.UnrolledFeaturesInfo
]:
  """Normalizes the input dataset into a dictionary of values.

    Also unrolls the multi-dimentional features.

  Args:
    data: Input data.
    single_dim_columns: Optional list of columns that should be
      single-dimensional. If one such column is multi-dimensional, raise an
      error.
    not_unrolled_multi_dim_columns: Optional list of columns that should be
      multi-dimensional and not unrolled.
    expected_unrolled_columns: Information from the data spec about columns
      expected to be unrolled.

  Returns:
    The normalized features, and information about unrolled features.
  """

  unroll_dict_kwargs = {
      "single_dim_columns": single_dim_columns or [],
      "not_unrolled_multi_dim_columns": not_unrolled_multi_dim_columns or [],
      "expected_unrolled_columns": expected_unrolled_columns or [],
  }

  if pandas_io.is_pandas_dataframe(data):
    return _unroll_dict(pandas_io.to_dict(data), **unroll_dict_kwargs)
  elif polars_io.is_polars_dataframe(data):
    return _unroll_dict(polars_io.to_dict(data), **unroll_dict_kwargs)
  elif xarray_io.is_xarray_dataset(data):
    return _unroll_dict(xarray_io.to_dict(data), **unroll_dict_kwargs)
  elif tensorflow_io.is_tensorflow_dataset(data):
    return _unroll_dict(tensorflow_io.to_dict(data), **unroll_dict_kwargs)
  elif pygrain_io.is_pygrain(data):
    return _unroll_dict(pygrain_io.to_dict(data), **unroll_dict_kwargs)

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


def build_batched_example_generator(
    data: dataset_io_types.IODataset,
):
  """Converts any support dataset format into a batched example generator.

  Usage example:

  ```python
  generator = build_batched_example_generator({
      "a":np.array([1, 2, 3]),
      "b":np.array(["x", "y", "z"]),
      })
  for batch in generator.generate(batch_size=2, shuffle=False):
    print(batch)
    >> { "a":np.array([1, 2]), "b":np.array(["x", "y"]) }
    >> { "a":np.array([3]), "b":np.array(["z"]) }
  ```

  Args:
    data: Support dataset format.

  Returns:
    Example generator.
  """
  if pandas_io.is_pandas_dataframe(data):
    return pandas_io.PandasBatchedExampleGenerator(data)
  elif isinstance(data, dict):
    return numpy_io.NumpyDictBatchedExampleGenerator(data)
  else:
    # TODO: Add support for other YDF dataset formats.
    raise ValueError(
        f"Unsupported dataset type to train a Deep YDF model: {type(data)}"
    )
