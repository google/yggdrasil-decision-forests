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

"""Connectors for loading data from PyArrow tables (e.g. read from Parquet)."""

import sys
from typing import Any, Dict

from absl import logging
import numpy as np

from ydf.dataset.io import dataset_io_types


def import_pyarrow():
  # pytype: disable=import-error
  # pylint: disable=g-import-not-at-top
  try:
    import pyarrow as pa

    return pa
  except ImportError:
    logging.warning(
        "Importing data from PyArrow tables requires PyArrow to be installed."
        " Install PyArrow with pip using `pip install pyarrow`."
    )
    raise
  # pylint: enable=g-import-not-at-top
  # pytype: enable=import-error


def is_pyarrow_table(data: dataset_io_types.IODataset) -> bool:
  if "pyarrow" in sys.modules:
    pa = sys.modules["pyarrow"]
    return isinstance(data, (pa.Table, pa.RecordBatch))
  return False


def _column_to_values(name: str, column: Any) -> dataset_io_types.InputValues:
  """Converts a PyArrow array or chunked array into dict of Numpy arrays."""
  pa = import_pyarrow()

  if isinstance(column, pa.ChunkedArray):
    # Note: `combine_chunks` also handles chunked arrays without any chunk,
    # which is how an empty table is represented.
    column = column.combine_chunks()
  column_type = column.type

  if pa.types.is_null(column_type):
    # A column without type only contains missing values. It is fed as an
    # array of empty strings, which YDF interprets as missing values.
    return np.full(len(column), "", dtype=np.object_)

  if pa.types.is_temporal(column_type):
    raise ValueError(
        f"The column {name!r} has the temporal type {column_type!s} which is"
        " not supported by YDF. Convert the column to a numerical or string"
        " value before feeding it to YDF, e.g. with"
        f' `table.set_column(table.column_names.index({name!r}), {name!r},'
        f' table[{name!r}].cast("int64"))`. Note that a date is often better'
        " represented by its components (e.g. the day of the week, the month)"
        " than by a single number."
    )

  if (
      pa.types.is_struct(column_type)
      or pa.types.is_map(column_type)
      or pa.types.is_union(column_type)
      or pa.types.is_run_end_encoded(column_type)
  ):
    raise ValueError(
        f"The column {name!r} has the nested type {column_type!s} which is not"
        " supported by YDF. Flatten the column before feeding it to YDF, e.g."
        " with `table.flatten()` for a column of structs."
    )

  if pa.types.is_decimal(column_type):
    column = column.cast(pa.float32())
  elif (
      pa.types.is_string(column_type)
      or pa.types.is_large_string(column_type)
      or pa.types.is_string_view(column_type)
  ):
    # Empty strings are missing values for categorical features.
    column = column.fill_null("")
  elif (
      pa.types.is_binary(column_type)
      or pa.types.is_large_binary(column_type)
      or pa.types.is_binary_view(column_type)
      or pa.types.is_fixed_size_binary(column_type)
  ):
    column = column.fill_null(b"")

  # Note: `zero_copy_only=False` is required for columns containing missing
  # values and for non-primitive columns. This conversion decodes dictionary
  # columns, converts missing numerical values to NaN, and converts columns of
  # lists into arrays of arrays.
  return column.to_numpy(zero_copy_only=False)


def to_dict(
    data: dataset_io_types.IODataset,
) -> Dict[str, dataset_io_types.InputValues]:
  """Converts a PyArrow Table or RecordBatch to a dict of numpy arrays."""
  pa = import_pyarrow()

  assert isinstance(data, (pa.Table, pa.RecordBatch))

  column_names = data.schema.names
  if len(set(column_names)) != len(column_names):
    duplicates = sorted({c for c in column_names if column_names.count(c) > 1})
    raise ValueError(
        "The PyArrow table must not contain duplicated column names. The"
        f" following column names are duplicated: {duplicates!r}."
    )

  return {
      name: _column_to_values(name, column)
      for name, column in zip(column_names, data.columns)
  }
