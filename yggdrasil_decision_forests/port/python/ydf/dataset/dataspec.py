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

"""Dataspec utilities."""

import dataclasses
import enum
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb


# The different ways a user can specify the columns of VerticalDataset.
ColumnDef = Union["Column", str, Tuple[str, "Semantic"]]
ColumnDefs = Optional[List[ColumnDef]]

# Normalized version of "ColumnDefs".
NormalizedColumnDefs = Optional[List["Column"]]

# The key of the "out of vocabulary" (OOV) used by YDF.
# Must match kOutOfDictionaryItemKey in
# yggdrasil_decision_forests/dataset/data_spec.h
YDF_OOD = "<OOD>"
YDF_OOD_BYTES = b"<OOD>"

# Mapping between Numpy dtypes and YDF dtypes.
_NP_DTYPE_TO_YDF_DTYPE = {
    np.int8: ds_pb.DType.DTYPE_INT8,
    np.int16: ds_pb.DType.DTYPE_INT16,
    np.int32: ds_pb.DType.DTYPE_INT32,
    np.int64: ds_pb.DType.DTYPE_INT64,
    np.uint8: ds_pb.DType.DTYPE_UINT8,
    np.uint16: ds_pb.DType.DTYPE_UINT16,
    np.uint32: ds_pb.DType.DTYPE_UINT32,
    np.uint64: ds_pb.DType.DTYPE_UINT64,
    np.float16: ds_pb.DType.DTYPE_FLOAT16,
    np.float32: ds_pb.DType.DTYPE_FLOAT32,
    np.float64: ds_pb.DType.DTYPE_FLOAT64,
    np.bool_: ds_pb.DType.DTYPE_BOOL,
    np.str_: ds_pb.DType.DTYPE_BYTES,
    np.bytes_: ds_pb.DType.DTYPE_BYTES,
    np.object_: ds_pb.DType.DTYPE_BYTES,
}

NP_SUPPORTED_INT_DTYPE = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

NP_SUPPORTED_FLOAT_DTYPE = [
    np.float16,
    np.float32,
    np.float64,
]


def np_dtype_to_ydf_dtype(np_dtype: np.dtype) -> Optional["ds_pb.DType"]:
  """Converts a Numpy dtype to a YDF dtype.

  If the numpy dtype has no matching ydf dtype, returns None.

  Args:
    np_dtype: Numpy dtype.

  Returns:
    YDF dtype.
  """

  if hasattr(np_dtype, "type"):
    ydf_dtype = _NP_DTYPE_TO_YDF_DTYPE.get(np_dtype.type)
  else:
    ydf_dtype = _NP_DTYPE_TO_YDF_DTYPE.get(np_dtype)

  if ydf_dtype is None:
    raise ValueError(f"ydf_dtype: {ydf_dtype!r} {np_dtype!r}")

  return ydf_dtype


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
      CATEGORICAL_SET can change between examples. The order of values inside a
      feature values does not matter.
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

  def to_proto_type(self) -> ds_pb.ColumnType:
    if self in SEMANTIC_TO_PROTO:
      return SEMANTIC_TO_PROTO[self]
    else:
      raise NotImplementedError(f"Unsupported semantic {self}")

  @classmethod
  def from_proto_type(cls, column_type: ds_pb.ColumnType):
    if column_type in PROTO_TO_SEMANTIC:
      return PROTO_TO_SEMANTIC[column_type]
    else:
      raise NotImplementedError(f"Unsupported semantic {column_type}")


# Mappings between semantic enum in python and in protobuffer and vice versa.
SEMANTIC_TO_PROTO = {
    Semantic.NUMERICAL: ds_pb.NUMERICAL,
    Semantic.CATEGORICAL: ds_pb.CATEGORICAL,
    Semantic.HASH: ds_pb.HASH,
    Semantic.CATEGORICAL_SET: ds_pb.CATEGORICAL_SET,
    Semantic.BOOLEAN: ds_pb.BOOLEAN,
    Semantic.DISCRETIZED_NUMERICAL: ds_pb.DISCRETIZED_NUMERICAL,
}
PROTO_TO_SEMANTIC = {v: k for k, v in SEMANTIC_TO_PROTO.items()}


class Monotonic(enum.Enum):
  """Monotonic constraint between a feature and the model output."""

  INCREASING = 1
  DECREASING = 2


# Map between integer monotonic constraints (as commonly used by decision
# forests libraries) and Monotonic enum value.
_INTEGER_MONOTONIC_TUPLES = (
    (0, None),
    (1, Monotonic.INCREASING),
    (-1, Monotonic.DECREASING),
)


def _build_integer_monotonic_map() -> Dict[int, Optional[Monotonic]]:
  """Returns a mapping between integer monotonic constraints and enum value.

  The returned value is always the same. So, when possible, create and reuse
  the result instead of calling "_build_integer_monotonic_map" multiple times.
  """

  return {key: value for key, value in _INTEGER_MONOTONIC_TUPLES}


# Various ways for a user to specify a monotonic constraint.
MonotonicConstraint = Optional[Union[Monotonic, Literal[-1, 0, +1]]]


def _normalize_monotonic_constraint(
    constraint: MonotonicConstraint,
) -> Optional[Monotonic]:
  """Normalizes monotonic constraints provided by the user.

  Args:
    constraint: User monotonic constraints.

  Returns:
    Normalized monotonic constraint.

  Raises:
    ValueError: If the user input is not a valid monotonic constraint.
  """

  if isinstance(constraint, int):
    monotonic_map = _build_integer_monotonic_map()
    if constraint not in monotonic_map:
      raise ValueError(
          "monotonic argument provided as integer should be one of"
          f" {list(monotonic_map)!r}. Got {constraint!r} instead"
      )
    constraint = monotonic_map[constraint]

  if constraint is None or isinstance(constraint, Monotonic):
    return constraint

  raise ValueError(
      "Unexpected monotonic value. monotonic value can be 0, +1, -1, None,"
      " Monotonic.INCREASING, or Monotonic.DECREASING. Got"
      f" {constraint!r} instead"
  )


@dataclasses.dataclass
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
    monotonic: Monotonic constraints between the feature and the model output.
      Use `None` (default; or 0) for an unconstrained feature. Use
      `Monotonic.INCREASING` (or +1) to ensure the model is monotonically
      increasing with the features. Use `Monotonic.DECREASING` (or -1) to ensure
      the model is monotonically decreasing with the features.
  """

  name: str
  semantic: Optional[Semantic] = None
  max_vocab_count: Optional[int] = None
  min_vocab_frequency: Optional[int] = None
  num_discretized_numerical_bins: Optional[int] = None
  monotonic: MonotonicConstraint = None

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
            "Argment min_vocab_frequency requires semantic=CATEGORICAL "
            " or semantic=CATEGORICAL_SET."
        )

    self._normalized_monotonic = _normalize_monotonic_constraint(self.monotonic)

    if self.monotonic and self.semantic and self.semantic != Semantic.NUMERICAL:
      raise ValueError(
          f"Feature {self.name!r} with monotonic constraint is expected to have"
          " semantic=NUMERICAL or semantic=None (default). Got"
          f" semantic={self.semantic!r} instead."
      )

  @property
  def normalized_monotonic(self) -> Optional[Monotonic]:
    """Returns the normalized version of the "monotonic" attribute."""

    return self._normalized_monotonic

  def to_proto_column_guide(self) -> ds_pb.ColumnGuide:
    """Creates a proto ColumnGuide from the given specification."""
    guide = ds_pb.ColumnGuide(
        # Only match the exact name
        column_name_pattern=f"^{self.name}$",
        categorial=ds_pb.CategoricalGuide(
            max_vocab_count=self.max_vocab_count,
            min_vocab_frequency=self.min_vocab_frequency,
        ),
        discretized_numerical=ds_pb.DiscretizedNumericalGuide(
            maximum_num_bins=self.num_discretized_numerical_bins
        ),
    )
    column_semantic = self.semantic
    if column_semantic is not None:
      guide.type = column_semantic.to_proto_type()
    return guide

  @classmethod
  def from_column_def(cls, column_def: ColumnDef):
    """Converts a ColumnDef to a Column."""
    if isinstance(column_def, cls):
      return column_def
    if isinstance(column_def, str):
      return Column(name=column_def)
    if isinstance(column_def, tuple):
      if (
          len(column_def) == 2
          and isinstance(column_def[0], str)
          and isinstance(column_def[1], Semantic)
      ):
        return Column(name=column_def[0], semantic=column_def[1])
    raise ValueError(
        f"Unsupported column definition: {column_def}. Supported definitions:"
        f" {ColumnDefs}"
    )


def normalize_column_defs(
    columns: ColumnDefs,
) -> NormalizedColumnDefs:
  """Converts a user column set into a normalized list of columns.

  Args:
    columns: Columns as defined by the user.

  Returns:
    Normalized column definitions.
  """

  if columns is None:
    return None

  elif isinstance(columns, list):
    return [_normalized_column(column) for column in columns]

  else:
    raise ValueError(
        f"Unsupported column definition with type: {type(columns)}"
    )


def _normalized_column(
    column: Union[Column, str, Tuple[str, Semantic]],
) -> Column:
  """Normalizes a single column."""

  if isinstance(column, str):
    # Raw column names
    return Column(column)
  elif isinstance(column, tuple):
    # Tuples of column names and semantics
    if (
        len(column) != 2
        or not isinstance(column[0], str)
        or not isinstance(column[1], Semantic)
    ):
      raise ValueError(
          "Column definition tuple should be a (name:str,"
          f" semantic:Semantic). Instead, got {column}"
      )
    return Column(column[0], column[1])
  elif isinstance(column, Column):
    # An already normalized column
    return column
  else:
    raise ValueError(f"Unsupported column item with type: {type(column)}")


@dataclasses.dataclass
class DataSpecInferenceArgs:
  """Arguments for the construction of a dataset."""

  columns: NormalizedColumnDefs
  include_all_columns: bool
  max_vocab_count: int
  min_vocab_frequency: int
  discretize_numerical_columns: bool
  num_discretized_numerical_bins: int
  max_num_scanned_rows_to_infer_semantic: int
  max_num_scanned_rows_to_compute_statistics: int

  def to_proto_guide(self) -> ds_pb.DataSpecificationGuide:
    """Creates a proto DataSpecGuide for these arguments.

    For consistency with in-memory datasets, the proto guide deactivates YDF's
    tokenizer.

    Returns:
      A guide to be used for dataspec inference by C++ YDF.
    """
    ignore_columns_without_guides = (
        False if self.columns is None else not self.include_all_columns
    )
    guide = ds_pb.DataSpecificationGuide(
        ignore_columns_without_guides=ignore_columns_without_guides,
        detect_numerical_as_discretized_numerical=self.discretize_numerical_columns,
        default_column_guide=ds_pb.ColumnGuide(
            categorial=ds_pb.CategoricalGuide(
                max_vocab_count=self.max_vocab_count,
                min_vocab_frequency=self.min_vocab_frequency,
            ),
            discretized_numerical=ds_pb.DiscretizedNumericalGuide(
                maximum_num_bins=self.num_discretized_numerical_bins
            ),
        ),
        allow_tokenization_for_inference_as_categorical_set=False,
    )
    if self.columns is not None:
      for column_def in self.columns:
        column = Column.from_column_def(column_def)
        guide.column_guides.append(column.to_proto_column_guide())

    guide.max_num_scanned_rows_to_guess_type = (
        self.max_num_scanned_rows_to_infer_semantic
    )
    guide.max_num_scanned_rows_to_accumulate_statistics = (
        self.max_num_scanned_rows_to_compute_statistics
    )
    return guide


def categorical_column_dictionary_to_list(
    column_spec: ds_pb.Column,
) -> List[str]:
  """Returns a list of string representation of dictionary items in a column.

  If the categorical column is integerized (i.e., it does not contain a
  dictionary), returns the string representation of the indices e.g. ["0", "1",
  "2"].

  Args:
    column_spec: Dataspec column.
  """

  if column_spec.categorical.is_already_integerized:
    return [
        str(i) for i in range(column_spec.categorical.number_of_unique_values)
    ]

  items = [None] * column_spec.categorical.number_of_unique_values

  for key, value in column_spec.categorical.items.items():
    if items[value.index] is not None:
      raise ValueError(
          f"Invalid dictionary. Duplicated index {value.index} in dictionary"
      )
    items[value.index] = key

  for index, value in enumerate(items):
    if value is None:
      raise ValueError(
          f"Invalid dictionary. No value for index {index} "
          f"in column {column_spec}"
      )

  return items  # pytype: disable=bad-return-type


def get_all_columns(
    available_columns: Sequence[str],
    inference_args: DataSpecInferenceArgs,
    required_columns: Optional[Sequence[str]],
    unroll_feature_info: Dict[str, List[str]] = {},
) -> Tuple[Sequence[Column], Dict[str, List[str]]]:
  """Gets all the columns to use by the model / learner.

  Args:
    available_columns: All the available column names in the dataset.
    inference_args: User configurations for the consumption of columns.
    required_columns: List of columns that must be used by the model. If None,
      defaults the columns specified in `inference_args`.  The order of the
      columns is important. First should come the columns in
      `inference_args.columns` in unchanged order, then any remaining columns in
      the order they appear at in the data.
    unroll_feature_info: Information about feature unrolling.

  Returns:
    The list of model input columns (This includes the required columns plus the
    specified columns in the inference_args that are available), and the
    actually used unrolled features.

  Raises:
    ValueError: One of the required columns is not available
  """
  available_columns_as_set = frozenset(available_columns)
  required_columns_as_set = frozenset(required_columns or [])

  if not required_columns_as_set.issubset(available_columns_as_set):
    raise ValueError(
        "One of the required columns was not found in the data. Required"
        f" columns: {required_columns}, available columns: {available_columns}"
    )

  used_unroll_feature_info = {}

  if inference_args.columns is None:
    # The user did not filter on the column names, so we use all the columns.
    specified_columns = [Column(col) for col in available_columns]
    used_unroll_feature_info.update(unroll_feature_info)
  else:
    specified_columns = []
    # Add the specified columns in order, but ignore those that do not exist.
    for col in inference_args.columns:
      if col.name in available_columns_as_set:
        if col.name in unroll_feature_info:
          raise ValueError(
              f"Column {col.name!r} is available both natively and unrolled."
          )
        specified_columns.append(col)
      elif col.name in unroll_feature_info:
        if col.name in unroll_feature_info:
          used_unroll_feature_info[col.name] = unroll_feature_info[col.name]
        specified_columns.extend(
            [Column(col) for col in unroll_feature_info[col.name]]
        )
      else:
        if required_columns is None or col.name in required_columns_as_set:
          raise ValueError(
              f"Column {col.name!r} is required but was not found in the data."
              f" Available columns: {available_columns}"
          )

  specified_columns_names = set(col.name for col in specified_columns)
  if inference_args.include_all_columns and inference_args.columns is not None:
    # The user specified the type of some of the columns, but asked for all the
    # columns to be used.
    used_unroll_feature_info.update(unroll_feature_info)
    for col_name in available_columns:
      if col_name not in specified_columns_names:
        specified_columns.append(Column(col_name))
        specified_columns_names.add(col_name)

  # Add remaining available columns.
  for col_name in available_columns:
    if (
        col_name in required_columns_as_set
        and col_name not in specified_columns_names
    ):
      # Note: Required columns that are not already included, are not unrolled.
      specified_columns.append(Column(col_name))
      specified_columns_names.add(col_name)
  return specified_columns, used_unroll_feature_info


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
