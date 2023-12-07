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

import copy
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from yggdrasil_decision_forests.dataset import data_spec_pb2
from ydf.cc import ydf
from ydf.dataset import dataspec
from ydf.dataset.io import dataset_io
from ydf.dataset.io import dataset_io_types
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

  def _add_column(
      self,
      column: dataspec.Column,
      column_data: Any,
      inference_args: Optional[dataspec.DataSpecInferenceArgs],
      column_idx: Optional[int],
  ):
    """Adds a column to the dataset and computes the column statistics."""
    assert (column_idx is None) != (inference_args is None)

    original_column_data = column_data

    if column.semantic == dataspec.Semantic.NUMERICAL:
      if not isinstance(column_data, np.ndarray):
        column_data = np.array(column_data, np.float32)

      if column_data.dtype != np.float32:
        log.warning(
            "Column '%s' with NUMERICAL semantic has dtype %s. Casting value"
            " to float32.",
            column.name,
            column_data.dtype.name,
            message_id=log.WarningMessage.CAST_NUMERICAL_TO_FLOAT32,
            is_strict=True,
        )

        try:
          column_data = column_data.astype(np.float32)
        except ValueError as e:
          raise ValueError(
              f"Cannot convert NUMERICAL column {column.name!r} of type"
              f" {_type(column_data)} and with content={column_data!r} to"
              " np.float32 values.\nNote: If the column is a label, make sure"
              " the training task is compatible. For example, you cannot train"
              " a regression model (task=ydf.Task.REGRESSION) on a string"
              " column."
          ) from e

      self._dataset.PopulateColumnNumericalNPFloat32(
          column.name, column_data, column_idx  # `column_idx` may be None
      )
      return

    elif column.semantic == dataspec.Semantic.BOOLEAN:
      if not isinstance(column_data, np.ndarray):
        column_data = np.array(column_data, np.bool_)

      self._dataset.PopulateColumnBooleanNPBool(
          column.name, column_data, column_idx  # `column_idx` may be None
      )
      return

    elif column.semantic == dataspec.Semantic.CATEGORICAL:
      from_boolean = False
      if not isinstance(column_data, np.ndarray):
        column_data = np.array(column_data, dtype=np.bytes_)
      elif column_data.dtype.type in [np.bool_]:
        bool_column_data = column_data
        column_data = np.full_like(bool_column_data, b"false", "|S5")
        column_data[bool_column_data] = b"true"
        from_boolean = True
      elif column_data.dtype.type in [
          np.object_,
          np.string_,
          np.int8,
          np.int16,
          np.int32,
          np.int64,
          np.uint8,
          np.uint16,
          np.uint32,
          np.uint64,
      ]:
        column_data = column_data.astype(np.bytes_)
      elif column_data.dtype.type in [
          np.float16,
          np.float32,
          np.float64,
      ]:
        raise ValueError(
            f"Cannot import column {column.name!r} with"
            f" semantic={column.semantic} as it contains floating point values."
            f" Got {original_column_data!r}.\nNote: If the column is a label,"
            " make sure the correct task is selected. For example, you cannot"
            " train a classification model (task=ydf.Task.CLASSIFICATION) with"
            " floating point labels."
        )

      if column_data.dtype.type == np.bytes_:
        if inference_args is not None:
          guide = dataspec.categorical_column_guide(column, inference_args)
          if from_boolean:
            guide["dictionary"] = np.array(
                [b"<OOV>", b"false", b"true"], dtype=np.bytes_
            )
          self._dataset.PopulateColumnCategoricalNPBytes(
              column.name, column_data, **guide
          )
        else:
          self._dataset.PopulateColumnCategoricalNPBytes(
              column.name, column_data, column_idx=column_idx
          )
        return

    elif column.semantic == dataspec.Semantic.CATEGORICAL_SET:
      if column_data.dtype.type != np.object_:
        raise ValueError("Categorical Set columns must be a list of lists.")
      if len(column_data) > 0:
        if column_data[0] and not isinstance(
            column_data[0], (list, np.ndarray)
        ):
          raise ValueError("Categorical Set columns must be a list of lists.")
      # TODO: b/313414785 - Consider speeding this up by moving logic to C++.
      # np.unique also sorts the unique elements, which is expected by YDF.
      column_data = [np.unique(row).astype(np.bytes_) for row in column_data]
      boundaries = np.cumsum(
          [0] + [len(row) for row in column_data[:-1]], dtype=np.int64
      )
      bank = np.concatenate(column_data)

      if inference_args is not None:
        guide = dataspec.categorical_column_guide(column, inference_args)
        self._dataset.PopulateColumnCategoricalSetNPBytes(
            column.name, bank, boundaries, **guide
        )
      else:
        self._dataset.PopulateColumnCategoricalSetNPBytes(
            column.name, bank, boundaries, column_idx=column_idx
        )
      return

    elif column.semantic == dataspec.Semantic.HASH:
      if not isinstance(column_data, np.ndarray):
        column_data = np.array(column_data, dtype=np.bytes_)
      elif column_data.dtype.type in [
          np.object_,
          np.string_,
          np.bool_,
          np.int8,
          np.int16,
          np.int32,
          np.int64,
          np.uint8,
          np.uint16,
          np.uint32,
          np.uint64,
      ]:
        column_data = column_data.astype(np.bytes_)
      elif column_data.dtype.type in [
          np.float16,
          np.float32,
          np.float64,
      ]:
        raise ValueError(
            f"Cannot import column {column.name!r} with"
            f" semantic={column.semantic} as it contains floating point values."
            f" Got {original_column_data!r}."
        )

      if column_data.dtype.type == np.bytes_:
        self._dataset.PopulateColumnHashNPBytes(
            column.name, column_data, column_idx=column_idx
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
    columns: dataspec.ColumnDefs = None,
    include_all_columns: bool = False,
    max_vocab_count: int = 2000,
    min_vocab_frequency: int = 5,
    discretize_numerical_columns: bool = False,
    num_discretized_numerical_bins: int = 255,
    max_num_scanned_rows_to_infer_semantic: int = 10000,
    max_num_scanned_rows_to_compute_statistics: int = 10000,
    data_spec: Optional[data_spec_pb2.DataSpecification] = None,
    required_columns: Optional[Sequence[str]] = None,
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
    data: Source dataset. Supported formats: VerticalDataset, (typed) path,
      Pandas Dataframe, dictionary of string to Numpy array or lists. If the
      data is already a VerticalDataset, it is returned unchanged.
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
        data, required_columns, data_spec=data_spec, inference_args=None
    )
  else:
    inference_args = dataspec.DataSpecInferenceArgs(
        columns=dataspec.normalize_column_defs(columns),
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
    )


def create_vertical_dataset_with_spec_or_args(
    data: dataset_io_types.IODataset,
    required_columns: Optional[Sequence[str]],
    inference_args: Optional[dataspec.DataSpecInferenceArgs],
    data_spec: Optional[data_spec_pb2.DataSpecification],
) -> VerticalDataset:
  """Returns a vertical dataset from inference args or data spec (not both!)."""
  assert (data_spec is None) != (inference_args is None)
  # If `data` is a path, try to import from the path directly from C++.
  # Everything else we try to transform into a dictionary with Python.
  if isinstance(data, str) or (
      isinstance(data, list) and data and all(isinstance(s, str) for s in data)
  ):
    return create_vertical_dataset_from_path(
        data, required_columns, inference_args, data_spec
    )
  else:
    data_dict = dataset_io.cast_input_dataset_to_dict(data)
    return create_vertical_dataset_from_dict_of_values(
        data_dict,
        required_columns,
        inference_args=inference_args,
        data_spec=data_spec,
    )


def create_vertical_dataset_from_path(
    path: Union[str, List[str]],
    required_columns: Optional[Sequence[str]],
    inference_args: Optional[dataspec.DataSpecInferenceArgs],
    data_spec: Optional[data_spec_pb2.DataSpecification],
) -> VerticalDataset:
  """Returns a VerticalDataset from (list of) path using YDF dataset reading."""
  assert (data_spec is None) != (inference_args is None)
  if not isinstance(path, str):
    path = paths.normalize_list_of_paths(path)
  dataset = VerticalDataset()
  if data_spec is not None:
    dataset._dataset.CreateFromPathWithDataSpec(
        path, data_spec, required_columns
    )
  if inference_args is not None:
    dataset._dataset.CreateFromPathWithDataSpecGuide(
        path, inference_args.to_proto_guide(), required_columns
    )
  return dataset


def create_vertical_dataset_from_dict_of_values(
    data: Dict[str, dataset_io_types.InputValues],
    required_columns: Optional[Sequence[str]],
    inference_args: Optional[dataspec.DataSpecInferenceArgs],
    data_spec: Optional[data_spec_pb2.DataSpecification],
) -> VerticalDataset:
  """Specialization of create_vertical_dataset to dictionary of values.

  The data spec is either inferred using inference_args or uses the given
  data_spec

  Args:
    data: Data to copy to the Vertical Dataset.
    required_columns: Names of columns that are required in the data.
    inference_args: Arguments for data spec inference. Must be None if data_spec
      is set.
    data_spec: Data spec of the given data. Must be None if inference_args is
      set.

  Returns:
    A Vertical dataset with the given properties.
  """

  def dataspec_to_normalized_columns(
      data: Dict[str, dataset_io_types.InputValues],
      columns,
      required_columns: Sequence[str],
  ):
    normalized_columns = []
    for column_spec in columns:
      if column_spec.name not in data and column_spec.name in required_columns:
        raise ValueError(
            f"The data spec expects columns {column_spec.name} which was not"
            f" found in the data. Available columns: {list(data)}. Required"
            f" columns: {required_columns}"
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
            dataspec.Column(
                name=column_spec.name,
                semantic=dataspec.Semantic.from_proto_type(column_spec.type),
            )
        )
    return normalized_columns

  assert (data_spec is None) != (inference_args is None)
  dataset = VerticalDataset()
  if data_spec is None:
    # If `required_columns` is None, only check if the columns mentioned in the
    # `inference_args` are required. This is checked by
    # dataspec.get_all_columns()
    normalized_columns = dataspec.get_all_columns(
        available_columns=list(data.keys()),
        inference_args=inference_args,
        required_columns=required_columns,
    )
  else:
    dataset._initialize_from_data_spec(data_spec)  # pylint: disable=protected-access
    required_columns = (
        required_columns
        if required_columns is not None
        else [c.name for c in data_spec.columns]
    )
    normalized_columns = dataspec_to_normalized_columns(
        data, data_spec.columns, required_columns
    )

  for column_idx, column in enumerate(normalized_columns):
    effective_column = copy.deepcopy(column)
    if column.name not in data:
      # The column is missing, so no data is passed but the column will still be
      # created.
      column_data = []
    else:
      column_data = data[column.name]
    if column.semantic is None:
      infered_semantic = infer_semantic(column.name, column_data)
      effective_column.semantic = infered_semantic

    dataset._add_column(  # pylint: disable=protected-access
        effective_column,
        column_data,
        inference_args=inference_args,  # Might be None
        column_idx=column_idx if data_spec is not None else None,
    )

  dataset._finalize(set_num_rows_in_data_spec=(data_spec is None))  # pylint: disable=protected-access
  return dataset


def infer_semantic(name: str, data: Any) -> dataspec.Semantic:
  """Infers the semantic of a column from its data."""

  # If a column has no data, we assume it only contains missing values.
  if len(data) == 0:
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
    if data.dtype.type in [
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ]:
      return dataspec.Semantic.NUMERICAL

    if data.dtype.type in [np.string_, np.bytes_]:
      return dataspec.Semantic.CATEGORICAL

    if data.dtype.type in [np.object_]:
      # For performance reasons, only check the type on the first and last item
      # of the column if it is tokenized.
      if (
          len(data) > 0
          and (isinstance(data[0], list) or isinstance(data[0], np.ndarray))
          and (isinstance(data[-1], list) or isinstance(data[-1], np.ndarray))
      ):
        return dataspec.Semantic.CATEGORICAL_SET
      return dataspec.Semantic.CATEGORICAL

    if data.dtype.type in [np.bool_]:
      return dataspec.Semantic.BOOLEAN
    type_str = f"numpy.array of {data.dtype}"
  else:
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
