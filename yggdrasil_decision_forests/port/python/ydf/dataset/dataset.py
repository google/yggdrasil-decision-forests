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
from dataclasses import dataclass  # pylint: disable=g-importing-member
import enum
import sys
import typing
from typing import Any, Dict, List, Optional, Tuple, Union

from absl import logging
import numpy as np

from yggdrasil_decision_forests.dataset import data_spec_pb2
from ydf.cc import ydf


if typing.TYPE_CHECKING:
  import pandas as pd  # pylint: disable=unused-import,g-bad-import-order

# Supported type of column input values.
InputValues = Union[np.ndarray, List[Any]]


# Supported types of datasets (including a YDF Dataset).
InputDataset = Union[Dict[str, InputValues], "pd.DataFrame", "VerticalDataset"]

SUPPORTED_INPUT_DATA_DESCRIPTION = """\
A dataset can be one of the following:
- A Pandas DataFrame
- A dictionary of column name (str) to values. Values can be lists of int, float, bool, str or bytes. Values can also be Numpy arrays.
- A YDF VerticalDataset
- A TensorFlow Batched Dataset
"""


# The different ways a user can specify the columns of VerticalDataset.
ColumnDefs = Optional[List[Union["Column", str, Tuple[str, "Semantic"]]]]


class Semantic(enum.Enum):
  """Semantic (e.g. numerical, categorical) of a column.

  Determines how a column is interpreted by the model.
  Similar to the "ColumnType" of YDF's DataSpecification.

  Attributes:
    NUMERICAL: Numerical value. Generally for quantities or counts with full
      ordering. For example, the age of a person, or the number of items in a
      bag. Can be a float or an integer.  Missing values are represented by
      math.nan.
    CATEGORICAL: A categorical value. Generally for a type/class in finite set
      of possible values without ordering. For example, the color RED in the set
      {RED, BLUE, GREEN}. Can be a string or an integer.  Missing values are
      represented by "" (empty sting) or value -2. An out-of-vocabulary value
      (i.e. a value that was never seen in training) is represented by any new
      string value or the value -1. Integer categorical values: (1) The training
      logic and model representation is optimized with the assumption that
      values are dense. (2) Internally, the value is stored as int32. The values
      should be <~2B. (3) The number of possible values is computed
      automatically from the training dataset. During inference, integer values
      greater than any value seen during training will be treated as
      out-of-vocabulary. (4) Minimum frequency and maximum vocabulary size
      constraints do not apply.
    HASH: The hash of a string value. Used when only the equality between values
      is important (not the value itself). Currently, only used for groups in
      ranking problems e.g. the query in a query/document problem. The hashing
      is computed with Google's farmhash and stored as an uint64.
    CATEGORICAL_SET: Set of categorical values. Great to represent tokenized
      texts. Can be a string. Unlike CATEGORICAL, the number of items in a
      CATEGORICAL_SET can change and the order/index of each item doesn"t
      matter.
    BOOLEAN: Boolean value. Can be a float or an integer. Missing values are
      represented by math.nan.  If a numerical tensor contains multiple values,
      its size should be constant, and each dimension isthreaded independently
      (and each dimension should always have the same "meaning").
    DISCRETIZED_NUMERICAL: Numerical values automatically discretized into bins.
      Discretized numerical columns are faster to train than (non-discretized)
      numerical columns. If the number of unique values of these columns is
      lower than the number of bins, the discretization is lossless from the
      point of view of the model. If the number of unique values of this columns
      is greater than the number of bins, the discretization is lossy from the
      point of view of the model. Lossy discretization can reduce and sometime
      increase (due to regularization) the quality of the model.
  """

  NUMERICAL = 1
  CATEGORICAL = 2
  HASH = 3
  CATEGORICAL_SET = 4
  BOOLEAN = 5
  DISCRETIZED_NUMERICAL = 6

  def to_proto_type(self) -> data_spec_pb2.ColumnType:
    if self in SEMANTIC_TO_PROTO:
      return SEMANTIC_TO_PROTO[self]
    else:
      raise NotImplementedError(f"Unsupported semantic {self}")

  @classmethod
  def from_proto_type(cls, column_type: data_spec_pb2.ColumnType):
    if column_type in PROTO_TO_SEMANTIC:
      return PROTO_TO_SEMANTIC[column_type]
    else:
      raise NotImplementedError(f"Unsupported semantic {column_type}")


# Mappings between semantic enum in python and in protobuffer and vice versa.
SEMANTIC_TO_PROTO = {
    Semantic.NUMERICAL: data_spec_pb2.NUMERICAL,
    Semantic.CATEGORICAL: data_spec_pb2.CATEGORICAL,
    Semantic.HASH: data_spec_pb2.HASH,
    Semantic.CATEGORICAL_SET: data_spec_pb2.CATEGORICAL_SET,
    Semantic.BOOLEAN: data_spec_pb2.BOOLEAN,
    Semantic.DISCRETIZED_NUMERICAL: data_spec_pb2.DISCRETIZED_NUMERICAL,
}
PROTO_TO_SEMANTIC = {v: k for k, v in SEMANTIC_TO_PROTO.items()}


@dataclass
class Column(object):
  """Semantic and parameters for a single column.

  This class allows to:
    1. Limit the input features of the model.
    2. Manually specify the semantic of a feature.
    3. Specify feature specific hyper-parameters.

  Attributes:
    name: The name of the column or feature.
    semantic: Semantic of the column. If None, the semantic is automatically
      determined. The semantic controls how a column is interpreted by a model.
      Using the wrong semantic (e.g. numerical instead of categorical) will hurt
      your model"s quality.
    max_vocab_count: For CATEGORICAL and CATEGORICAL_SET columns only. Number of
      unique categorical values stored as string. If more categorical values are
      present, the least frequent values are grouped into a Out-of-vocabulary
      item. Reducing the value can improve or hurt the model. If max_vocab_count
      = -1, the number of values in the column is not limited.
    min_vocab_frequency: For CATEGORICAL and CATEGORICAL_SET columns only.
      Minimum number of occurrence of a categorical value. Values present less
      than "min_vocab_frequency" times in the training dataset are treated as
      "Out-of-vocabulary".
    num_discretized_numerical_bins: For DISCRETIZED_NUMERICAL columns only.
      Number of bins used to discretize DISCRETIZED_NUMERICAL columns.
  """

  name: str
  semantic: Optional[Semantic] = None
  max_vocab_count: Optional[int] = None
  min_vocab_frequency: Optional[int] = None
  num_discretized_numerical_bins: Optional[int] = None

  def __post_init__(self):
    # Check matching between hyper-parameters and semantic.
    if self.semantic != Semantic.DISCRETIZED_NUMERICAL:
      if self.num_discretized_numerical_bins is not None:
        raise ValueError(
            "Argument num_discretized_numerical_bins requires"
            " semantic=DISCRETIZED_NUMERICAL."
        )

    if self.semantic not in [
        Semantic.CATEGORICAL,
        Semantic.CATEGORICAL_SET,
    ]:
      if self.max_vocab_count is not None:
        raise ValueError(
            "Argment max_vocab_count requires semantic=CATEGORICAL "
            " or semantic=CATEGORICAL_SET."
        )
      if self.min_vocab_frequency is not None:
        raise ValueError(
            "Argmentmin_vocab_frequency requires semantic=CATEGORICAL "
            " or semantic=CATEGORICAL_SET."
        )


@dataclass
class DataSpecInferenceArgs:
  """Arguments for the construction of a dataset."""

  columns: ColumnDefs
  include_all_columns: bool
  max_vocab_count: int
  min_vocab_frequency: int
  discretize_numerical_columns: bool
  num_discretized_numerical_bins: int


def is_pandas_dataframe(data: InputDataset) -> bool:
  if "pandas" in sys.modules:
    return isinstance(data, sys.modules["pandas"].DataFrame)
  return False


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
      column: Column,
      column_data: Any,
      inference_args: Optional[DataSpecInferenceArgs],
      column_idx: Optional[int],
  ):
    """Adds a column to the dataset and computes the column statistics."""
    original_column_data = column_data

    assert (column_idx is None) != (inference_args is None)
    if column.semantic == Semantic.NUMERICAL:
      if not isinstance(column_data, np.ndarray):
        column_data = np.array(column_data, np.float32)

      if column_data.dtype != np.float32:
        # TODO: Add control for warning (flag or count).
        logging.info(
            "Column '%s' with numerical semantic has dtype %s. Casting value to"
            " float32.",
            column.name,
            column_data.dtype.name,
        )

        try:
          column_data = column_data.astype(np.float32)
        except ValueError as e:
          raise ValueError(
              f"Cannot convert NUMERICAL column {column.name!r} with"
              f" content={column_data!r} to np.float32 values. If"
              " the column is a label, make sure the training task is"
              " compatible. For example, you cannot train a regression model"
              " (task=ydf.Task.REGRESSION) on a string column."
          ) from e

      self._dataset.PopulateColumnNumericalNPFloat32(
          column.name, column_data, column_idx  # `column_idx` may be None
      )
      return

    elif column.semantic == Semantic.BOOLEAN:
      if not isinstance(column_data, np.ndarray):
        column_data = np.array(column_data, np.bool_)

      self._dataset.PopulateColumnBooleanNPBool(
          column.name, column_data, column_idx  # `column_idx` may be None
      )
      return

    elif column.semantic == Semantic.CATEGORICAL:
      if not isinstance(column_data, np.ndarray):
        column_data = np.array(column_data, dtype=np.bytes_)
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
            f"Column {column.name!r} with semantic={column.semantic} should not"
            f" contain floating point values. Got {original_column_data!r}. If"
            " the column is a label, make sure the correct task is selected."
            " For example, you cannot train a classification model"
            " (task=ydf.Task.CLASSIFICATION) with floating point labels."
        )

      if column_data.dtype.type == np.bytes_:
        if inference_args is not None:
          guide = categorical_column_guide(column, inference_args)
          self._dataset.PopulateColumnCategoricalNPBytes(
              column.name, column_data, **guide
          )
        else:
          self._dataset.PopulateColumnCategoricalNPBytes(
              column.name, column_data, column_idx=column_idx
          )
        return

    raise ValueError(
        f"Column {column.name!r} with semantic={column.semantic} and"
        f" content={original_column_data!r} is not supported"
    )

  def _initialize_from_data_spec(
      self, data_spec: data_spec_pb2.DataSpecification
  ):
    self._dataset.CreateColumnsFromDataSpec(data_spec)

  def _finalize(self, set_num_rows_in_data_spec: bool):
    self._dataset.SetAndCheckNumRows(set_num_rows_in_data_spec)


def create_vertical_dataset(
    data: InputDataset,
    columns: ColumnDefs = None,
    include_all_columns: bool = False,
    max_vocab_count: int = 2000,
    min_vocab_frequency: int = 5,
    discretize_numerical_columns: bool = False,
    num_discretized_numerical_bins: int = 255,
    data_spec: Optional[data_spec_pb2.DataSpecification] = None,
) -> VerticalDataset:
  """Creates a VerticalDataset from various sources of data.

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
    data: Source dataset. Supported formats: VerticalDataset, Pandas Dataframe,
      dictionary of string to Numpy array or lists. If the data is already a
      VerticalDataset, it is returned unchanged.
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
    data_spec: Dataspec to be used for this dataset. If a data spec is given,
      all other arguments except `data` should not be provided.

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
        data, data_spec=data_spec, inference_args=None
    )
  else:
    inference_args = DataSpecInferenceArgs(
        columns=columns,
        include_all_columns=include_all_columns,
        max_vocab_count=max_vocab_count,
        min_vocab_frequency=min_vocab_frequency,
        discretize_numerical_columns=discretize_numerical_columns,
        num_discretized_numerical_bins=num_discretized_numerical_bins,
    )
    return create_vertical_dataset_with_spec_or_args(
        data, inference_args=inference_args, data_spec=None
    )


def create_vertical_dataset_with_spec_or_args(
    data: Dict[str, InputValues],
    inference_args: Optional[DataSpecInferenceArgs],
    data_spec: Optional[data_spec_pb2.DataSpecification],
):
  """Creates a vertical dataset with inference args or data spec (not both!)."""
  assert (data_spec is None) != (inference_args is None)
  data_dict = cast_input_dataset_to_dict(data)
  return create_vertical_dataset_from_dict_of_values(
      data_dict, inference_args=inference_args, data_spec=data_spec
  )


def cast_input_dataset_to_dict(data: InputDataset) -> Dict[str, InputValues]:
  """Transforms the input dataset into a dictionary of values."""
  if is_pandas_dataframe(data):
    import pandas as pd  # pylint: disable=g-import-not-at-top

    assert isinstance(data, pd.DataFrame)
    # Pandas dataframe
    if data.ndim != 2:
      raise ValueError("The pandas dataframe must be two-dimensional.")
    data_dict = data.to_dict("series")

    def clean(values):
      if values.dtype == "object":
        return values.to_numpy(copy=False, na_value="")
      else:
        return values.to_numpy(copy=False)

    data_dict = {k: clean(v) for k, v in data_dict.items()}

    return data_dict

  elif isinstance(data, dict):
    # Dictionary of values
    return data

  # TensorFlow dataset.
  # Note: We only test if the dataset is a TensorFlow dataset if the object name
  # look like a TensorFlow object. This way, we avoid importing TF is not
  # necessary.
  if (
      "tensorflow" in str(type(data))
      and data.__class__.__name__ == "_BatchDataset"
      and hasattr(data, "rebatch")
  ):
    # Create a single batch with all the data
    full_batch = next(iter(data.rebatch(sys.maxsize)))
    return {k: v.numpy() for k, v in full_batch.items()}

  raise ValueError(
      "Cannot import dataset from"
      f" {type(data)}.\n{SUPPORTED_INPUT_DATA_DESCRIPTION}"
  )


def create_vertical_dataset_from_dict_of_values(
    data: Dict[str, InputValues],
    inference_args: Optional[DataSpecInferenceArgs],
    data_spec: Optional[data_spec_pb2.DataSpecification],
) -> VerticalDataset:
  """Specialization of create_vertical_dataset to dictionary of values.

  The data spec is either inferred using inference_args or uses the given
  data_spec

  Args:
    data: Data to copy to the Vertical Dataset.
    inference_args: Arguments for data spec inference. Must be None if data_spec
      is set.
    data_spec: Data spec of the given data. Must be None if inference_args is
      set.

  Returns:
    A Vertical dataset with the given properties.
  """

  def dataspec_to_normalized_columns(data, columns):
    normalized_columns = []
    for column_spec in columns:
      if column_spec.name not in data:
        raise ValueError(
            f"The data spec expects columns {column_spec.name} which was not"
            f" found in the data. Available columns: {data.keys()}"
        )
      normalized_columns.append(
          Column(
              name=column_spec.name,
              semantic=Semantic.from_proto_type(column_spec.type),
          )
      )
    return normalized_columns

  assert (data_spec is None) != (inference_args is None)
  dataset = VerticalDataset()
  if data_spec is None:
    normalized_columns = normalize_column_defs(
        available_columns=list(data.keys()),
        inference_args=inference_args,
    )
  else:
    dataset._initialize_from_data_spec(data_spec)  # pylint: disable=protected-access
    normalized_columns = dataspec_to_normalized_columns(data, data_spec.columns)

  for column_idx, column in enumerate(normalized_columns):
    column_data = data[column.name]
    effective_column = column
    if column.semantic is None:
      effective_column = copy.deepcopy(column)
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


def infer_semantic(name: str, data: Any) -> Semantic:
  """Infers the semantic of a column from its data."""

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
      return Semantic.NUMERICAL

    if data.dtype.type in [np.string_, np.bytes_, np.object_]:
      return Semantic.CATEGORICAL

    if data.dtype.type in [np.bool_]:
      return Semantic.BOOLEAN

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


def normalize_column_defs(
    available_columns: List[str], inference_args: DataSpecInferenceArgs
) -> List[Column]:
  """Converts a user column set into a list of columns."""

  columns = inference_args.columns
  if columns is None:
    # Select all the available columns with an unknown semantic.
    normalized_columns = [Column(f) for f in available_columns]

  elif isinstance(columns, list):

    def normalized_item(item) -> Column:
      if isinstance(item, str):
        # Raw column names
        return Column(item)
      elif isinstance(item, tuple):
        # Tuples of column names and semantics
        if (
            len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], Semantic)
        ):
          raise ValueError(
              "Column definition tuple should be a (name:str,"
              f" semantic:Semantic). Instead, got {item}"
          )
        return Column(item[0], item[1])
      elif isinstance(item, Column):
        # An already normalized column
        return item
      else:
        raise ValueError(f"Unsupported column item with type: {type(columns)}")

    normalized_columns = [normalized_item(f) for f in columns]

    # Check that the user specified columns exist
    available_columns_set = set(available_columns)
    for f in normalized_columns:
      if f.name not in available_columns_set:
        raise ValueError(
            f"Column {f.name} no found. The available columns are:"
            f" {available_columns}"
        )

    if inference_args.include_all_columns:
      # Add the remaining columns with an unknown semantic.
      existing_columns = {f.name for f in normalized_columns}
      for f in available_columns:
        if f in existing_columns:
          # Skip columns already specified by the user
          continue
        normalized_columns.append(Column(f))

  else:
    raise ValueError(
        f"Unsupported column definition with type: {type(columns)}"
    )

  return normalized_columns


def priority(a: Any, b: Any) -> Any:
  """Merge arguments with priority.

  If "a" is not None, return "a".
  If "a" is None, return "b".

  Args:
    a: High priority argument.
    b: Low priority argument.

  Returns:
    Selected argument.
  """
  return a if a is not None else b


def categorical_column_guide(
    column: Column, inference_args: DataSpecInferenceArgs
) -> Dict[str, Any]:
  return {
      "max_vocab_count": priority(
          column.max_vocab_count, inference_args.max_vocab_count
      ),
      "min_vocab_frequency": priority(
          column.min_vocab_frequency, inference_args.min_vocab_frequency
      ),
  }


def column_defs_contains_column(column_name: str, columns: ColumnDefs) -> bool:
  """Checks if the given ColumnDefs contain a column of the given name."""
  if columns is None:
    return False

  elif isinstance(columns, list):
    for item in columns:
      if isinstance(item, str):
        if item == column_name:
          return True
      elif isinstance(item, tuple):
        if (
            len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], Semantic)
        ):
          raise ValueError(
              "Column definition tuple should be a (name:str,"
              f" semantic:Semantic). Instead, got {item}"
          )
        if item[0] == column_name:
          return True
      elif isinstance(item, Column):
        if item.name == column_name:
          return True
    return False
  else:
    raise ValueError(
        f"Unsupported column definition with type: {type(columns)}"
    )
