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

"""Dataset implementations of PYDF."""

import abc
import collections
import copy
import math
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from yggdrasil_decision_forests.dataset import data_spec_pb2
from ydf.cc import ydf
from ydf.dataset import dataspec as dataspec_lib
from ydf.dataset.io import dataset_io
from ydf.dataset.io import dataset_io_types
from ydf.dataset.io import generator as generator_lib
from ydf.utils import log
from ydf.utils import paths


InputDataset = Union[dataset_io_types.IODataset, "VerticalDataset"]


class VerticalDataset:
  """Dataset for fast column-wise iteration."""

  def __init__(self):
    self._dataset = ydf.VerticalDataset()

  def memory_usage(self) -> int:
    """Memory usage of the dataset in bytes."""
    return self._dataset.MemoryUsage()

  def data_spec(self) -> data_spec_pb2.DataSpecification:
    """Data spec of the dataset.

    The dataspec is the protobuffer containing the dataset schema.

    Returns:
      A dataspec proto.
    """
    return self._dataset.data_spec()

  def _normalize_categorical_string_values(
      self,
      column: dataspec_lib.Column,
      values: npt.ArrayLike,
      original_column_data: Any,
  ) -> npt.NDArray[np.bytes_]:
    """Normalizes a sequence of categorical string values into an array of bytes."""

    def normalize_categorical_string_value(value: Any) -> bytes:
      """Normalizes a categorical string value into a bytes literal."""
      if isinstance(value, str):
        return value.encode("utf-8")
      if isinstance(value, (bytes, np.bytes_)):
        return value
      if isinstance(value, (bool, np.bool_)):
        return b"true" if value else b"false"
      if isinstance(value, (int, np.integer)):
        return str(value).encode("utf-8")
      if isinstance(value, (float, np.floating)):
        raise ValueError(
            f"Cannot import column {column.name!r} with"
            f" semantic={column.semantic} as it contains floating point"
            " values.\nNote: If the column is a label, make sure the correct"
            " task is selected. For example, you cannot train a classification"
            " model (task=ydf.Task.CLASSIFICATION) with floating point labels."
        )
      if isinstance(value, list):
        raise ValueError(
            f"Cannot import column {column.name!r} with"
            f" semantic={column.semantic} as it contains lists.\nNote:"
            " Unrolling multi-dimensional columns is only supported for numpy"
            " arrays"
        )
      raise ValueError(
          f"Cannot import column {column.name!r} with"
          f" semantic={column.semantic} and"
          f" type={_type(original_column_data)}.\nNote: If the column is a"
          " label, the semantic was selected based on the task. For example,"
          " task=ydf.Task.CLASSIFICATION requires a CATEGORICAL compatible"
          " label column, and task=ydf.Task.REGRESSION requires a NUMERICAL"
          " compatible label column."
      )

    normalized_values = [normalize_categorical_string_value(v) for v in values]
    return np.array(normalized_values, dtype=np.bytes_)

  def _add_column(
      self,
      column: dataspec_lib.Column,
      column_data: Any,
      inference_args: Optional[dataspec_lib.DataSpecInferenceArgs],
      column_idx: Optional[int],
      is_label: bool,
  ):
    """Adds a column to the dataset and computes the column statistics."""
    assert (column_idx is None) != (inference_args is None)

    original_column_data = column_data

    if (
        column.semantic == dataspec_lib.Semantic.NUMERICAL
        or column.semantic == dataspec_lib.Semantic.DISCRETIZED_NUMERICAL
    ):
      assert column.semantic is not None  # Appease pylint.
      if not isinstance(column_data, np.ndarray):
        column_data = np.array(column_data, np.float32)
      ydf_dtype = dataspec_lib.np_dtype_to_ydf_dtype(column_data.dtype)

      if column_data.dtype != np.float32:
        log.warning(
            "Column '%s' with %s semantic has dtype %s. Casting value"
            " to float32.",
            column.name,
            column.semantic.name,
            column_data.dtype.name,
            message_id=log.WarningMessage.CAST_NUMERICAL_TO_FLOAT32,
            is_strict=True,
        )

        try:
          column_data = column_data.astype(np.float32)
        except ValueError as e:
          raise ValueError(
              f"Cannot convert {column.semantic.name} column {column.name!r} of"
              f" type {_type(column_data)} and with content={column_data!r} to"
              " np.float32 values.\nNote: If the column is a label, make sure"
              " the training task is compatible. For example, you cannot train"
              " a regression model (task=ydf.Task.REGRESSION) on a string"
              " column."
          ) from e
      if column_data.ndim != 1:
        raise ValueError(
            f"Cannot convert {column.semantic.name} column {column.name!r} "
            f" with content={column_data!r} to a 1-dimensional array of"
            " np.float32 values. Note: Unrolling multi-dimensional columns is"
            " only supported for numpy arrays"
        )

      if column.semantic == dataspec_lib.Semantic.NUMERICAL:
        self._dataset.PopulateColumnNumericalNPFloat32(
            column.name,
            column_data,
            ydf_dtype=ydf_dtype,
            column_idx=column_idx,  # `column_idx` may be None
        )
      elif column.semantic == dataspec_lib.Semantic.DISCRETIZED_NUMERICAL:
        if (
            column.num_discretized_numerical_bins is None
            and inference_args is not None
        ):
          column.num_discretized_numerical_bins = (
              inference_args.num_discretized_numerical_bins
          )
        self._dataset.PopulateColumnDiscretizedNumericalNPFloat32(
            column.name,
            column_data,
            ydf_dtype=ydf_dtype,
            maximum_num_bins=column.num_discretized_numerical_bins,
            column_idx=column_idx,  # `column_idx` may be None
        )
      else:
        raise ValueError("Not reached")
      return

    elif column.semantic == dataspec_lib.Semantic.BOOLEAN:
      if not isinstance(column_data, np.ndarray):
        column_data = np.array(column_data, np.bool_)
      ydf_dtype = dataspec_lib.np_dtype_to_ydf_dtype(column_data.dtype)
      if column_data.dtype != np.bool_:
        message = (
            f"Cannot import column {column.name!r} with"
            f" semantic={column.semantic} as it does not contain boolean"
            f" values. Got {original_column_data!r}."
        )
        raise ValueError(message)
      if column_data.ndim != 1:
        raise ValueError(
            f"Cannot convert BOOLEAN column {column.name!r}"
            f" with content={column_data!r} to a 1-dimensional array of"
            " np.float32 values. Note: Unrolling multi-dimensional columns is"
            " only supported for numpy arrays"
        )

      self._dataset.PopulateColumnBooleanNPBool(
          column.name,
          column_data,
          ydf_dtype=ydf_dtype,
          column_idx=column_idx,  # `column_idx` may be None
      )
      return

    elif column.semantic == dataspec_lib.Semantic.CATEGORICAL:
      force_dictionary = None
      if not isinstance(column_data, np.ndarray):
        column_data = self._normalize_categorical_string_values(
            column, column_data, original_column_data
        )
      ydf_dtype = dataspec_lib.np_dtype_to_ydf_dtype(column_data.dtype)

      if column_data.dtype.type in [np.bool_]:
        bool_column_data = column_data
        column_data = np.full_like(bool_column_data, b"false", "|S5")
        column_data[bool_column_data] = b"true"
        force_dictionary = [dataspec_lib.YDF_OOD_BYTES, b"false", b"true"]
      elif column_data.dtype.type in dataspec_lib.NP_SUPPORTED_INT_DTYPE:
        if is_label:
          # Sort increasing.
          dictionary = np.unique(column_data)
          column_data = column_data.astype(np.bytes_)
          force_dictionary = [dataspec_lib.YDF_OOD_BYTES, *dictionary]
        else:
          column_data = column_data.astype(np.bytes_)
      elif column_data.dtype.type in [np.object_, np.str_]:
        column_data = self._normalize_categorical_string_values(
            column, column_data, original_column_data
        )
        if is_label:
          # Sort lexicographically (as opposed to by frequency as for features).
          dictionary = np.unique(column_data)
          force_dictionary = [dataspec_lib.YDF_OOD_BYTES, *dictionary]
      elif np.issubdtype(column_data.dtype, np.floating):
        message = (
            f"Cannot import column {column.name!r} with"
            f" semantic={column.semantic} as it contains floating point values."
        )
        if is_label:
          message += (
              "\nNote: This is a label column. Try one of the following"
              " solutions: (1) To train a classification model, cast the label"
              " values as integers. (2) To train a regression or a ranking"
              " model, configure the learner with `task=ydf.Task.REGRESSION`)."
          )
        message += f"\nGot {original_column_data!r}."
        raise ValueError(message)
      assert column_data.ndim == 1, "Categorical columns must be 1-dimensional"

      if column_data.dtype.type == np.bytes_:
        if inference_args is not None:
          guide = dataspec_lib.categorical_column_guide(column, inference_args)
          if force_dictionary:
            guide["dictionary"] = np.array(force_dictionary, dtype=np.bytes_)
          self._dataset.PopulateColumnCategoricalNPBytes(
              column.name, column_data, **guide, ydf_dtype=ydf_dtype
          )
        else:
          self._dataset.PopulateColumnCategoricalNPBytes(
              column.name,
              column_data,
              ydf_dtype=ydf_dtype,
              column_idx=column_idx,
          )
        return

    elif column.semantic == dataspec_lib.Semantic.CATEGORICAL_SET:
      if (
          not isinstance(column_data, list)
          and column_data.dtype.type != np.object_
      ):
        raise ValueError("Categorical Set columns must be a list of lists.")
      column_data = np.empty(len(original_column_data), dtype=np.object_)
      column_data_are_bytes = True
      force_dictionary = None
      for i, row in enumerate(original_column_data):
        if isinstance(row, list):
          column_data[i] = self._normalize_categorical_string_values(
              column, row, original_column_data
          )
        elif isinstance(row, np.ndarray):
          if row.dtype.type in [np.bool_]:
            bool_row = row
            column_data[i] = np.full_like(bool_row, b"false", "|S5")
            column_data[i][bool_row] = b"true"
            force_dictionary = [dataspec_lib.YDF_OOD_BYTES, b"false", b"true"]
          elif row.dtype.type in [np.object_, np.str_]:
            column_data[i] = self._normalize_categorical_string_values(
                column, row, original_column_data
            )
          elif row.dtype.type in dataspec_lib.NP_SUPPORTED_INT_DTYPE:
            column_data[i] = row.astype(np.bytes_)
          elif np.issubdtype(row.dtype, np.floating):
            raise ValueError(
                f"Cannot import column {column.name!r} with"
                f" semantic={column.semantic} as it contains floating point"
                " values.\nNote: If the column is a label, make sure the"
                " correct task is selected. For example, you cannot train a"
                " classification model (task=ydf.Task.CLASSIFICATION) with"
                " floating point labels."
            )
          elif row.dtype.type == np.bytes_:
            column_data[i] = row
          else:
            column_data_are_bytes = False
            break
        elif not row:
          column_data[i] = np.array([b""], dtype=np.bytes_)
        else:
          raise ValueError(
              f"Cannot import column {column.name!r} with"
              f" semantic={column.semantic} as it contains non-list values."
              f" Got {original_column_data!r}."
          )
      ydf_dtype = dataspec_lib.np_dtype_to_ydf_dtype(column_data.dtype)

      if column_data_are_bytes:
        if inference_args is not None:
          guide = dataspec_lib.categorical_column_guide(column, inference_args)
          if force_dictionary:
            guide["dictionary"] = np.array(force_dictionary, dtype=np.bytes_)
          self._dataset.PopulateColumnCategoricalSetNPBytes(
              column.name, column_data, **guide, ydf_dtype=ydf_dtype
          )
        else:
          self._dataset.PopulateColumnCategoricalSetNPBytes(
              column.name,
              column_data,
              ydf_dtype=ydf_dtype,
              column_idx=column_idx,
          )
        return

    elif column.semantic == dataspec_lib.Semantic.HASH:
      if not isinstance(column_data, np.ndarray):
        column_data = np.array(column_data, dtype=np.bytes_)
      ydf_dtype = dataspec_lib.np_dtype_to_ydf_dtype(column_data.dtype)

      if column_data.dtype.type in [
          np.object_,
          np.bytes_,
          np.bool_,
      ] or np.issubdtype(column_data.dtype, np.integer):
        column_data = column_data.astype(np.bytes_)
      elif np.issubdtype(column_data.dtype, np.floating):
        raise ValueError(
            f"Cannot import column {column.name!r} with"
            f" semantic={column.semantic} as it contains floating point values."
            f" Got {original_column_data!r}."
        )

      if column_data.dtype.type == np.bytes_:
        self._dataset.PopulateColumnHashNPBytes(
            column.name,
            column_data,
            ydf_dtype=ydf_dtype,
            column_idx=column_idx,
        )
        return

    elif column.semantic == dataspec_lib.Semantic.NUMERICAL_VECTOR_SEQUENCE:
      if not isinstance(column_data, list):
        raise ValueError(
            "A numerical vector sequence should be a list. Got"
            f" {original_column_data!r}"
        )
      if not (
          isinstance(column_data[0], np.ndarray)
          and column_data[0].ndim == 2
          and (
              column_data[0].dtype.type in dataspec_lib.NP_SUPPORTED_INT_DTYPE
              or column_data[0].dtype.type
              in dataspec_lib.NP_SUPPORTED_FLOAT_DTYPE
          )
      ):
        raise ValueError(
            "Each value of a numerical vector sequence should be numerical"
            f" numpy array with two dimension. Got {original_column_data!r}"
        )
      ydf_dtype = dataspec_lib.np_dtype_to_ydf_dtype(column_data[0].dtype)
      self._dataset.PopulateColumnNumericalVectorSequence(
          column.name,
          column_data,
          ydf_dtype=ydf_dtype,
          column_idx=column_idx,
      )
      return

    raise ValueError(
        f"Cannot import column {column.name!r} with semantic={column.semantic},"
        f" type={_type(original_column_data)} and"
        f" content={original_column_data!r}.\nNote: If the column is a label,"
        " the semantic was selected based on the task. For example,"
        " task=ydf.Task.CLASSIFICATION requires a CATEGORICAL compatible label"
        " column, and task=ydf.Task.REGRESSION requires a NUMERICAL compatible"
        " label column."
    )

  def _initialize_from_data_spec(
      self, data_spec: data_spec_pb2.DataSpecification
  ):
    self._dataset.CreateColumnsFromDataSpec(data_spec)

  def _finalize(self, set_num_rows_in_data_spec: bool):
    self._dataset.SetAndCheckNumRowsAndFillMissing(set_num_rows_in_data_spec)


def create_vertical_dataset(
    data: InputDataset,
    columns: dataspec_lib.ColumnDefs = None,
    include_all_columns: bool = False,
    max_vocab_count: int = 2000,
    min_vocab_frequency: int = 5,
    discretize_numerical_columns: bool = False,
    num_discretized_numerical_bins: int = 255,
    max_num_scanned_rows_to_infer_semantic: int = 100_000,
    max_num_scanned_rows_to_compute_statistics: int = 100_000,
    data_spec: Optional[data_spec_pb2.DataSpecification] = None,
    required_columns: Optional[Sequence[str]] = None,
    dont_unroll_columns: Optional[Sequence[str]] = None,
    label: Optional[str] = None,
) -> VerticalDataset:
  """Creates a VerticalDataset from various sources of data.

  The feature semantics are automatically determined and can be explicitly
  set with the `columns` argument. The semantics of a dataset (or model) are
  available its data_spec.

  Note that the CATEGORICAL_SET semantic is not automatically inferred when
  reading from file. When reading from CSV files, setting the CATEGORICAL_SET
  semantic for a feature will have YDF tokenize the feature. When reading from
  in-memory datasets (e.g. pandas), YDF only accepts lists of lists for
  CATEGORICAL_SET features.

  Usage example:

  ```python
  import pandas as pd
  import ydf

  df = pd.read_csv("my_dataset.csv")

  # Loads all the columns
  ds = ydf.create_vertical_dataset(df)

  # Only load columns "a" and "b". Ensure "b" is interpreted as a categorical
  # feature.
  ds = ydf.create_vertical_dataset(df,
    columns=[
      "a",
      ("b", ydf.semantic.categorical),
    ])
  ```

  Args:
    data: Source dataset. Supported formats: VerticalDataset, (typed) path, list
      of (typed) paths, Pandas DataFrame, Xarray Dataset, TensorFlow Dataset,
      PyGrain DataLoader and Dataset (experimental, Linux only), dictionary of
      string to NumPy array or lists. If the data is already a VerticalDataset,
      it is returned unchanged.
    columns: If None, all columns are imported. The semantic of the columns is
      determined automatically. Otherwise, if include_all_columns=False
      (default) only the column listed in `columns` are imported. If
      include_all_columns=True, all the columns are imported and only the
      semantic of the columns NOT in `columns` is determined automatically. If
      specified, "columns" defines the order of the columns - any non-listed
      columns are appended in-order after the specified columns (if
      include_all_columns=True).
    include_all_columns: See `columns`.
    max_vocab_count: Maximum size of the vocabulary of CATEGORICAL and
      CATEGORICAL_SET columns stored as strings. If more unique values exist,
      only the most frequent values are kept, and the remaining values are
      considered as out-of-vocabulary.  If max_vocab_count = -1, the number of
      values in the column is not limited (not recommended).
    min_vocab_frequency: Minimum number of occurrence of a value for CATEGORICAL
      and CATEGORICAL_SET columns. Value observed less than
      `min_vocab_frequency` are considered as out-of-vocabulary.
    discretize_numerical_columns: If true, discretize all the numerical columns
      before training. Discretized numerical columns are faster to train with,
      but they can have a negative impact on the model quality. Using
      `discretize_numerical_columns=True` is equivalent as setting the column
      semantic DISCRETIZED_NUMERICAL in the `column` argument. See the
      definition of DISCRETIZED_NUMERICAL for more details.
    num_discretized_numerical_bins: Number of bins used when disretizing
      numerical columns.
    max_num_scanned_rows_to_infer_semantic: Number of rows to scan when
      inferring the column's semantic if it is not explicitly specified. Only
      used when reading from file, in-memory datasets are always read in full.
      Setting this to a lower number will speed up dataset reading, but might
      result in incorrect column semantics. Set to -1 to scan the entire
      dataset.
    max_num_scanned_rows_to_compute_statistics: Number of rows to scan when
      computing a column's statistics. Only used when reading from file,
      in-memory datasets are always read in full. A column's statistics include
      the dictionary for categorical features and the mean / min / max for
      numerical features. Setting this to a lower number will speed up dataset
      reading, but skew statistics in the dataspec, which can hurt model quality
      (e.g. if an important category of a categorical feature is considered
      OOV). Set to -1 to scan the entire dataset.
    data_spec: Dataspec to be used for this dataset. If a data spec is given,
      all other arguments except `data` and `required_columns` should not be
      provided.
    required_columns: List of columns required in the data. If None, all columns
      mentioned in the data spec or `columns` are required.
    dont_unroll_columns: List of columns that cannot be unrolled. If one such
      column needs to be unrolled, raise an error.
    label: Name of the label column, if any.

  Returns:
    Dataset to be ingested by the learner algorithms.

  Raises:
    ValueError: If the dataset has an unsupported type.
  """
  if isinstance(data, VerticalDataset):
    return data

  if data_spec is not None:
    if columns is not None:
      raise ValueError(
          "When passing a data spec at dataset creation, `columns` must be"
          " None and all arguments to guide data spec inference are ignored."
      )
    return create_vertical_dataset_with_spec_or_args(
        data,
        required_columns,
        data_spec=data_spec,
        inference_args=None,
        single_dim_columns=dont_unroll_columns,
        label=label,
    )
  else:
    inference_args = dataspec_lib.DataSpecInferenceArgs(
        columns=dataspec_lib.normalize_column_defs(columns),
        include_all_columns=include_all_columns,
        max_vocab_count=max_vocab_count,
        min_vocab_frequency=min_vocab_frequency,
        discretize_numerical_columns=discretize_numerical_columns,
        num_discretized_numerical_bins=num_discretized_numerical_bins,
        max_num_scanned_rows_to_infer_semantic=max_num_scanned_rows_to_infer_semantic,
        max_num_scanned_rows_to_compute_statistics=max_num_scanned_rows_to_compute_statistics,
    )
    return create_vertical_dataset_with_spec_or_args(
        data,
        required_columns,
        inference_args=inference_args,
        data_spec=None,
        single_dim_columns=dont_unroll_columns,
        label=label,
    )


def create_vertical_dataset_with_spec_or_args(
    data: dataset_io_types.IODataset,
    required_columns: Optional[Sequence[str]],
    inference_args: Optional[dataspec_lib.DataSpecInferenceArgs],
    data_spec: Optional[data_spec_pb2.DataSpecification],
    single_dim_columns: Optional[Sequence[str]] = None,
    label: Optional[str] = None,
) -> VerticalDataset:
  """Returns a vertical dataset from inference args or data spec (not both!)."""
  assert (data_spec is None) != (inference_args is None)
  # If `data` is a path, try to import from the path directly from C++.
  # Everything else we try to transform into a dictionary with Python.
  if isinstance(data, str) or (
      isinstance(data, Sequence)
      and data
      and all(isinstance(s, str) for s in data)
  ):
    return create_vertical_dataset_from_path(
        data, required_columns, inference_args, data_spec
    )
  else:
    # Ignore unrolling for list or set features.
    if inference_args is None or inference_args.columns is None:
      not_unrolled_multi_dim_columns = None
    else:
      not_unrolled_multi_dim_columns = [
          c.name
          for c in inference_args.columns
          if c.semantic == dataspec_lib.Semantic.CATEGORICAL_SET
      ]

    # Convert the data to an in-memory dictionary of numpy array.
    # Also unroll multi-dimensional features.
    data_dict, unroll_feature_info = dataset_io.cast_input_dataset_to_dict(
        data,
        single_dim_columns=single_dim_columns,
        not_unrolled_multi_dim_columns=not_unrolled_multi_dim_columns,
    )
    return create_vertical_dataset_from_dict_of_values(
        data_dict,
        unroll_feature_info,
        required_columns,
        inference_args=inference_args,
        data_spec=data_spec,
        label=label,
    )


def create_vertical_dataset_from_path(
    path: Union[str, List[str]],
    required_columns: Optional[Sequence[str]],
    inference_args: Optional[dataspec_lib.DataSpecInferenceArgs],
    data_spec: Optional[data_spec_pb2.DataSpecification],
) -> VerticalDataset:
  """Returns a VerticalDataset from (list of) path using YDF dataset reading."""
  assert (data_spec is None) != (inference_args is None)
  if not isinstance(path, str):
    path = paths.normalize_list_of_paths(path)
  dataset = VerticalDataset()
  if data_spec is not None:
    dataset._dataset.CreateFromPathWithDataSpec(  # pylint: disable=protected-access
        path, data_spec, required_columns
    )
  if inference_args is not None:
    dataset._dataset.CreateFromPathWithDataSpecGuide(  # pylint: disable=protected-access
        path, inference_args.to_proto_guide(), required_columns
    )
  return dataset


def _create_missing_feature_error_message(
    data: Dict[str, dataset_io_types.InputValues],
    column_spec: data_spec_pb2.Column,
    shapes_of_given_columns: Dict[str, int],
) -> Optional[str]:
  """Builds an error message explaining why a feature/column is be missing."""

  if column_spec.is_unstacked:
    # The missing feature is multi-dimensional.

    # Find the baseline of the requested feature.
    feature_components = dataset_io.parse_unrolled_feature_name(
        column_spec.name
    )
    # Note: feature_components is guaranteed to be set since column_spec
    # is_unstacked=True (i.e. is multi-dimensional).
    assert feature_components is not None
    expected_shape = feature_components[2]

    if feature_components[0] in shapes_of_given_columns:
      # There is a miss-match of shape.
      provided_shape = shapes_of_given_columns[feature_components[0]]
      return (
          "Unexpected shape for multi-dimensional column"
          f" {feature_components[0]!r}. Column has shape"
          f" {provided_shape} but is expected to have shape"
          f" {expected_shape}."
      )
    else:
      if feature_components[0] in data:
        # The base-name of the missing (multi-dimensional) feature is equal to
        # the name of a single-dimensional feature.
        return (
            f"Column {feature_components[0]!r} is expected to be"
            f" multi-dimensional with shape {expected_shape} but it is"
            " single-dimensional. If you use Numpy arrays, the column is"
            " expected to be an array of shape [num_examples,"
            f" {expected_shape}]."
        )
  else:
    # The missing feature is single-dimensional.

    if column_spec.name in shapes_of_given_columns:
      provided_shape = shapes_of_given_columns[column_spec.name]
      # The name of the missing (single-dimensional) feature is equal to the
      # base name of a multi-dimensional feature.
      return (
          f"Column {column_spec.name!r} is expected to single-dimensional but"
          f" it is multi-dimensional with shape {provided_shape}."
      )

  # The feature is simply missing.
  return None


def create_vertical_dataset_from_dict_of_values(
    data: Dict[str, dataset_io_types.InputValues],
    unroll_feature_info: dataset_io_types.UnrolledFeaturesInfo,
    required_columns: Optional[Sequence[str]],
    inference_args: Optional[dataspec_lib.DataSpecInferenceArgs],
    data_spec: Optional[data_spec_pb2.DataSpecification],
    label: Optional[str] = None,
) -> VerticalDataset:
  """Specialization of create_vertical_dataset to dictionary of values.

  The data spec is either inferred using inference_args or uses the given
  data_spec

  Args:
    data: Data to copy to the Vertical Dataset.
    unroll_feature_info: Information about feature unrolling.
    required_columns: Names of columns that are required in the data.
    inference_args: Arguments for data spec inference. Must be None if data_spec
      is set.
    data_spec: Data spec of the given data. Must be None if inference_args is
      set.
    label: Name of the label column, if any.

  Returns:
    A Vertical dataset with the given properties.
  """

  def dataspec_to_normalized_columns(
      data: Dict[str, dataset_io_types.InputValues],
      data_spec: data_spec_pb2.DataSpecification,
      required_columns: Sequence[str],
  ):
    # Index the multi-dim feature provided by the user.
    shapes_of_given_columns = {}
    for name in data:
      components = dataset_io.parse_unrolled_feature_name(name)
      if components is not None:
        # The provided value is part of a multi-dim feature.
        shapes_of_given_columns[components[0]] = components[2]

    normalized_columns = []
    for column_spec in data_spec.columns:
      if column_spec.name not in data and column_spec.name in required_columns:

        # Try to generate a helpful message about the missing feature.
        error_prefix = _create_missing_feature_error_message(
            data,
            column_spec,
            shapes_of_given_columns,
        )

        if error_prefix is not None:
          error_prefix = f"{error_prefix}\n\nDetails: "

        raise ValueError(
            f"{error_prefix}Missing required column {column_spec.name!r}.\nThe"
            f" available unrolled columns are: {list(data)}.\nThe required"
            f" unrolled columns are: {required_columns}"
        )

      if (
          column_spec.type == data_spec_pb2.CATEGORICAL_SET
          and column_spec.HasField("tokenizer")
          and column_spec.tokenizer.splitter
          != data_spec_pb2.Tokenizer.NO_SPLITTING
      ):
        log.warning(
            f"The dataspec for columns {column_spec.name} specifies a"
            " tokenizer, but it is ignored when reading in-memory datasets."
        )
      else:
        normalized_columns.append(
            dataspec_lib.Column(
                name=column_spec.name,
                semantic=dataspec_lib.Semantic.from_proto_type(
                    column_spec.type
                ),
            )
        )
    return normalized_columns

  assert (data_spec is None) != (inference_args is None)
  dataset = VerticalDataset()
  if data_spec is None:
    # If `required_columns` is None, only check if the columns mentioned in the
    # `inference_args` are required. This is checked by
    # dataspec.get_all_columns()
    normalized_columns, effective_unroll_feature_info = (
        dataspec_lib.get_all_columns(
            available_columns=list(data.keys()),
            inference_args=inference_args,
            required_columns=required_columns,
            unroll_feature_info=unroll_feature_info,
        )
    )
  else:
    effective_unroll_feature_info = None  # To please linter
    dataset._initialize_from_data_spec(data_spec)  # pylint: disable=protected-access
    required_columns = (
        required_columns
        if required_columns is not None
        else [c.name for c in data_spec.columns]
    )
    normalized_columns = dataspec_to_normalized_columns(
        data, data_spec, required_columns
    )

  columns_to_check = []
  for column_idx, column in enumerate(normalized_columns):
    effective_column = copy.deepcopy(column)
    if column.name not in data:
      # The column is missing, so no data is passed but the column will still be
      # created.
      column_data = []
    else:
      column_data = data[column.name]

    if column.semantic is None:
      discretize_numerical = (
          inference_args is None
      ) or inference_args.discretize_numerical_columns
      infered_semantic = infer_semantic(
          column.name, column_data, discretize_numerical
      )
      effective_column.semantic = infered_semantic
      columns_to_check.append(column_idx)

    dataset._add_column(  # pylint: disable=protected-access
        effective_column,
        column_data,
        inference_args=inference_args,  # Might be None
        column_idx=column_idx if data_spec is not None else None,
        is_label=label == column.name,
    )

  if data_spec is None:
    assert effective_unroll_feature_info is not None
    dataset._dataset.SetMultiDimDataspec(effective_unroll_feature_info)  # pylint: disable=protected-access

    warnings = validate_dataspec(dataset.data_spec(), columns_to_check)
    if warnings:
      log.warning(
          "%s",
          "\n".join(warnings),
          message_id=log.WarningMessage.CATEGORICAL_LOOK_LIKE_NUMERICAL,
      )

  dataset._finalize(set_num_rows_in_data_spec=(data_spec is None))  # pylint: disable=protected-access
  return dataset


def validate_dataspec(
    data_spec: data_spec_pb2.DataSpecification,
    to_check: Sequence[int],
) -> List[str]:
  """Validates a dataspec.

  Can raise an error or return a warning (as list of strings). If return None,
  the dataspec is correctly.

  Args:
    data_spec: A dataspec to check.
    to_check: List of columns to check.

  Returns:
    List of warnings.
  """
  warnings = []
  for column_idx in to_check:
    column = data_spec.columns[column_idx]
    if column.is_manual_type:
      continue
    if column.type != data_spec_pb2.CATEGORICAL:
      continue
    if len(column.categorical.items) < 3:
      continue

    count_look_numerical = 0
    count_total = 0
    examples_of_value = []
    for k, v in dataspec_lib.categorical_vocab_iterator(column.categorical):
      count_total += v.count
      if look_numerical(k):
        count_look_numerical += v.count
        if len(examples_of_value) < 3:
          examples_of_value.append(k)

    if count_look_numerical >= 0.8 * count_total:
      examples_for_warning = b", ".join(examples_of_value)
      warnings.append(
          f"Column {column.name!r} is detected as CATEGORICAL but its values"
          f" look like numbers (e.g., {examples_for_warning}). Should"
          " the column not be NUMERICAL instead? If so, feed numerical values"
          " instead of strings or objects."
      )
  return warnings


def look_numerical(v: Union[str, bytes]) -> bool:
  """Tests if a string look like a numerical value."""
  try:
    float(v)
    return True
  except ValueError:
    return False


def infer_semantic(
    name: str,
    data: Any,
    discretize_numerical: bool,
) -> dataspec_lib.Semantic:
  """Infers the semantic of a column from its data."""

  # If a column has no data, we assume it only contains missing values.
  if len(data) == 0:  # pylint: disable=g-explicit-length-test
    raise ValueError(
        f"Cannot infer automatically the semantic of column {name!r} since no"
        " data for this column was provided. Make sure this column exists in"
        " the dataset, or exclude the column from the list of required"
        " columns. If the dataset should contain missing values for the all"
        " examples of this column, specify the semantic of the column manually"
        f" using the `features` argument e.g. `features=[({name!r},"
        " ydf.Semantic.NUMERICAL)]` if the feature is numerical."
    )

  if isinstance(data, np.ndarray):
    # We finely control the supported types.
    if (
        data.dtype.type in dataspec_lib.NP_SUPPORTED_INT_DTYPE
        or data.dtype.type in dataspec_lib.NP_SUPPORTED_FLOAT_DTYPE
    ):
      if discretize_numerical:
        return dataspec_lib.Semantic.DISCRETIZED_NUMERICAL
      else:
        return dataspec_lib.Semantic.NUMERICAL

    if data.dtype.type in [np.bytes_, np.str_]:
      return dataspec_lib.Semantic.CATEGORICAL

    if data.dtype.type in [np.object_]:
      # For performance reasons, only check the type on the first and last item
      # of the column if it is tokenized.
      if (
          len(data) > 0  # pylint: disable=g-explicit-length-test
          and isinstance(data[0], (list, np.ndarray))
          and isinstance(data[-1], (list, np.ndarray))
      ):
        return dataspec_lib.Semantic.CATEGORICAL_SET
      return dataspec_lib.Semantic.CATEGORICAL

    if data.dtype.type in [np.bool_]:
      return dataspec_lib.Semantic.BOOLEAN
    type_str = f"numpy.array of {data.dtype}"
  else:
    if (
        isinstance(data, list)
        and isinstance(data[0], np.ndarray)
        and data[0].ndim == 2
        and (
            data[0].dtype.type in dataspec_lib.NP_SUPPORTED_INT_DTYPE
            or data[0].dtype.type in dataspec_lib.NP_SUPPORTED_FLOAT_DTYPE
        )
    ):
      return dataspec_lib.Semantic.NUMERICAL_VECTOR_SEQUENCE
    type_str = str(type(data))

  raise ValueError(
      f"Cannot infer automatically the semantic of column {name!r} with"
      f" type={type_str}, and content={data}. Convert the column to a supported"
      " type, or specify the semantic of the column manually using the"
      f" `features` argument e.g. `features=[({name!r},"
      " ydf.Semantic.NUMERICAL)]` if the feature is numerical."
  )


def _type(value: Any) -> str:
  """Returns a string representation of the type of value."""

  if isinstance(value, np.ndarray):
    return f"numpy's array of '{value.dtype.name}'"
  else:
    return str(type(value))


def dense_integer_dictionary_size(values: np.ndarray) -> Optional[int]:
  """Gets the number of items in a dense and zero-indexed array of integers.

  If the array is not dense or not zero-indexed, returns None.

  Args:
    values: Numpy array of integer values.

  Returns:
    Number of unique dense values, or None.
  """
  unique_values = np.unique(values).tolist()
  if (
      unique_values
      and unique_values[0] == 0
      and unique_values[-1] + 1 == len(unique_values)
  ):
    return len(unique_values)
  return None


def infer_dataspec_types(
    batch: generator_lib.NumpyExampleBatch,
    dataspec: data_spec_pb2.DataSpecification,
    data_spec_args: dataspec_lib.DataSpecInferenceArgs,
):
  """Infers the semantic of columns from a single generator batch."""

  # Generate the columns
  columns_to_check = []
  columns, _ = dataspec_lib.get_all_columns(
      available_columns=list(batch.keys()),
      inference_args=data_spec_args,
      required_columns=None,
  )
  for column_idx, py_column in enumerate(columns):
    if py_column.semantic is None:
      columns_to_check.append(column_idx)
      py_semantic = infer_semantic(py_column.name, batch[py_column.name], False)
    else:
      py_semantic = py_column.semantic
    column = data_spec_pb2.Column(
        name=py_column.name,
        type=py_semantic.to_proto_type(),
    )

    # Dtyping
    column_data = batch[py_column.name]
    column.dtype = dataspec_lib.np_dtype_to_ydf_dtype(column_data.dtype)

    # Copy filter configs
    if column.type == data_spec_pb2.ColumnType.CATEGORICAL:
      # Note: We add +1 to take into account the OOD item.
      if py_column.max_vocab_count is None:
        column.categorical.max_number_of_unique_values = (
            data_spec_args.max_vocab_count
        ) + 1
      else:
        column.categorical.max_number_of_unique_values = (
            py_column.max_vocab_count
        ) + 1
      if py_column.min_vocab_frequency is None:
        column.categorical.min_value_count = data_spec_args.min_vocab_frequency
      else:
        column.categorical.min_value_count = py_column.min_vocab_frequency

    dataspec.columns.append(column)
  warnings = validate_dataspec(dataspec, columns_to_check)
  if warnings:
    log.warning(
        "%s",
        "\n".join(warnings),
        message_id=log.WarningMessage.CATEGORICAL_LOOK_LIKE_NUMERICAL,
    )


def infer_dataspec(
    generator: generator_lib.BatchedExampleGenerator,
    data_spec_args: dataspec_lib.DataSpecInferenceArgs,
    seed: int = 1234,
) -> data_spec_pb2.DataSpecification:
  """Infers a dataspec from an example generator."""

  dataspec = data_spec_pb2.DataSpecification()
  rng = np.random.default_rng(seed)

  num_examples = 0
  accumulators = []
  for batch_idx, batch in enumerate(
      generator.generate(
          batch_size=1000, shuffle=True, seed=rng.integers(sys.maxsize)
      )
  ):
    if batch_idx == 0:
      # Generate the columns
      infer_dataspec_types(batch, dataspec, data_spec_args)
      for column in dataspec.columns:
        if column.type == data_spec_pb2.ColumnType.NUMERICAL:
          accumulators.append(
              NumericalDataSpecAccumulator(
                  column.name,
                  rng,
                  num_quantiles=data_spec_args.num_discretized_numerical_bins
                  if data_spec_args.discretize_numerical_columns
                  else None,
              )
          )
        elif column.type == data_spec_pb2.ColumnType.CATEGORICAL:
          accumulators.append(
              CategoricalDataSpecAccumulator(
                  column.name,
                  max_vocab_count=column.categorical.max_number_of_unique_values,
                  min_vocab_frequency=column.categorical.min_value_count,
              )
          )
        else:
          accumulators.append(None)

    for accumulator, column in zip(accumulators, dataspec.columns):
      if accumulator is not None:
        value = batch[column.name]
        try:
          accumulator.visit(value)
        except Exception as e:
          raise ValueError(
              f"While processing value {value} of feature {column.name}"
          ) from e

    num_examples += generator_lib.get_num_examples(batch)
    if (
        num_examples
        >= data_spec_args.max_num_scanned_rows_to_compute_statistics
    ):
      break

  dataspec.created_num_rows = num_examples
  for accumulator, column in zip(accumulators, dataspec.columns):
    if accumulator is not None:
      accumulator.finalize(column)
  return dataspec


class DataSpecAccumulator(abc.ABC):
  """Collects data to create the dataspec of a numerical column."""

  def __init__(self, name: str):
    self._name = name

  @abc.abstractmethod
  def visit(self, value: np.ndarray):
    raise NotImplementedError

  @abc.abstractmethod
  def finalize(self, column: data_spec_pb2.Column):
    raise NotImplementedError


class NumericalDataSpecAccumulator(DataSpecAccumulator):
  """Specialization for numerical columns."""

  def __init__(
      self,
      name: str,
      rng: np.random.Generator,
      num_quantiles: Optional[int],
  ):
    super().__init__(name)
    self._sum_values = 0.0
    self._sum_square_values = 0.0
    self._count_values = 0
    self._max_value = None
    self._min_value = None
    self._count_missing_values = 0
    self._num_quantiles = num_quantiles
    if self._num_quantiles is not None:
      self._reservoir = BatchReservoirSampling(rng=rng)
    else:
      self._reservoir = None

  def visit(self, value: np.ndarray):
    num_values = np.size(value)
    num_missing = np.count_nonzero(np.isnan(value))
    self._count_values += num_values - num_missing
    self._count_missing_values += num_missing

    if num_values == num_missing:
      return

    self._sum_values += np.nansum(value, dtype=np.float64)
    self._sum_square_values += np.nansum(value**2, dtype=np.float64)
    self._max_value = np.nanmax(value, initial=self._max_value)
    self._min_value = np.nanmin(value, initial=self._min_value)
    if self._reservoir is not None:
      self._reservoir.add(value.ravel())

  def finalize(self, column: data_spec_pb2.Column):
    if self._count_values == 0:
      log.warning(f"Feature {column.name} has no values")
      column.numerical.CopyFrom(
          data_spec_pb2.NumericalSpec(
              mean=0.5,
              standard_deviation=1,
              min_value=0,
              max_value=1,
          ),
      )
      if self._reservoir is not None:
        column.discretized_numerical.CopyFrom(
            data_spec_pb2.DiscretizedNumericalSpec(
                boundaries=[0, 1],
            )
        )
      return

    column.count_nas = self._count_missing_values
    mean = self._sum_values / self._count_values
    var = self._sum_square_values / self._count_values - mean * mean
    if var < 0:
      # Possible rounding error
      var = 0
    column.numerical.CopyFrom(
        data_spec_pb2.NumericalSpec(
            mean=mean,
            standard_deviation=math.sqrt(var),
            min_value=self._min_value,
            max_value=self._max_value,
        ),
    )
    if self._reservoir is not None:
      assert self._num_quantiles is not None
      quantiles, _ = self._reservoir.get_quantiles(self._num_quantiles)
      boundaries = np.unique(quantiles).tolist()
      if len(boundaries) < 2:
        if self._min_value == self._max_value:
          boundaries = [self._min_value - 1, self._min_value + 1]
        else:
          boundaries = [self._min_value, self._max_value]
      column.discretized_numerical.CopyFrom(
          data_spec_pb2.DiscretizedNumericalSpec(
              boundaries=boundaries,
          )
      )


class CategoricalDataSpecAccumulator(DataSpecAccumulator):
  """Specialization for categorical columns."""

  def __init__(self, name: str, max_vocab_count: int, min_vocab_frequency: int):
    super().__init__(name)
    self._items = collections.defaultdict(int)
    self._max_vocab_count = max_vocab_count
    self._min_vocab_frequency = min_vocab_frequency
    self._count_nan = 0

  def visit(self, value: np.ndarray):
    value = value.astype(np.bytes_)
    unique_values, counts = np.unique(value, return_counts=True)
    for key, count in zip(unique_values, counts):
      self._items[key.item()] += count.item()

  def finalize(self, column: data_spec_pb2.Column):
    column.count_nas = self._items.get(b"nan", 0)
    num_ood = self._items.get(dataspec_lib.YDF_OOD_BYTES, 0)

    # Sort values by count
    items = sorted(
        [
            (count, key)
            for key, count in self._items.items()
            if key != b"nan" and key != dataspec_lib.YDF_OOD_BYTES
        ],
        key=lambda x: (-x[0], x[1]),
    )

    # Prune items
    while items and (
        (self._max_vocab_count > 1 and len(items) + 1 > self._max_vocab_count)
        or items[-1][0] < self._min_vocab_frequency
    ):
      last = items.pop()
      num_ood += last[0]
    items.insert(0, (num_ood, dataspec_lib.YDF_OOD_BYTES))

    # TODO: Sort integer looking categorical strings
    column.categorical.CopyFrom(
        data_spec_pb2.CategoricalSpec(
            items={
                key: data_spec_pb2.CategoricalSpec.VocabValue(
                    index=item_idx, count=count
                )
                for item_idx, (count, key) in enumerate(items)
            },
            number_of_unique_values=len(items),
        )
    )


class BatchReservoirSampling:
  """Computes the quantiles of a stream of batches."""

  def __init__(
      self,
      rng: np.random.Generator,
      cache_size: int = 1_000_000,
  ):
    """Initializes the reservoir sampling.

    Args:
      rng: Optional numpy bit generator.
      cache_size: Cache size. The larger, the most accurate the estimates.
    """
    self._cache_size = cache_size
    self._samples = np.empty(cache_size, dtype=np.float32)
    self._num_in_cache = 0
    self._num_seen = 0
    self._rng = rng

  def add(self, data: np.ndarray) -> None:
    """Adds a batch of observations."""
    assert data.ndim == 1
    batch_size = data.shape[0]
    new_num_in_cache = self._num_in_cache + batch_size

    if new_num_in_cache <= self._cache_size:
      # Add the entire batch to the cache
      self._samples[self._num_in_cache : new_num_in_cache] = data
      self._num_in_cache = new_num_in_cache
      self._num_seen += batch_size
      return

    if self._num_in_cache + 1 <= self._cache_size:
      # Add the first elements to the cache until it's full
      num_consumed = self._cache_size - self._num_in_cache
      self._samples[self._num_in_cache : self._cache_size] = data[:num_consumed]
      data = data[num_consumed:]
      batch_size -= num_consumed
      self._num_in_cache += num_consumed
      self._num_seen += num_consumed
      assert self._num_in_cache == self._cache_size

    # Reservoir sampling
    rnd01 = self._rng.uniform(0.0, 1.0, batch_size)
    sample_idx = np.arange(self._num_seen + 1, self._num_seen + batch_size + 1)
    indexes = (rnd01 * sample_idx).astype(np.int32)
    is_selected = (indexes < self._cache_size).nonzero()
    self._samples[indexes[is_selected]] = data[is_selected]
    self._num_seen += batch_size

  def get_quantiles(self, num_quantiles: int) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the quantiles and corresponding quantile arguments."""
    # TODO: This is not optimal when a few values have a lot of
    # weights: The functions might return less than "num_quantiles" quantiles
    # when returning "num_quantiles" quantiles is possible.
    thresholds = np.linspace(0, 1, num_quantiles)
    quantiles = np.nanquantile(self._samples[: self._num_in_cache], thresholds)
    return quantiles.tolist(), thresholds.tolist()
