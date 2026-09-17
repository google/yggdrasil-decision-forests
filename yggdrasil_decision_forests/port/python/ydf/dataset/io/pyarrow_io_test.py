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

"""Tests for the PyArrow dataset connector."""

import datetime
from typing import Any, Dict

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from numpy import testing as npt
import pandas as pd
import pyarrow as pa

from ydf.dataset.io import pyarrow_io


def _column(data: Dict[str, Any], name: str) -> np.ndarray:
  """Returns a column of `data`, checking that it is a numpy array."""
  values = data[name]
  assert isinstance(values, np.ndarray), f"Got {type(values)} for {name!r}"
  return values


class IsPyArrowTableTest(absltest.TestCase):

  def test_table(self):
    self.assertTrue(pyarrow_io.is_pyarrow_table(pa.table({"a": [1, 2]})))

  def test_record_batch(self):
    self.assertTrue(pyarrow_io.is_pyarrow_table(pa.record_batch({"a": [1, 2]})))

  def test_empty_table(self):
    self.assertTrue(pyarrow_io.is_pyarrow_table(pa.table({})))

  def test_array_is_not_a_table(self):
    self.assertFalse(pyarrow_io.is_pyarrow_table(pa.array([1, 2])))

  def test_dict_is_not_a_table(self):
    self.assertFalse(pyarrow_io.is_pyarrow_table({"a": np.array([1, 2])}))

  def test_pandas_is_not_a_table(self):
    self.assertFalse(pyarrow_io.is_pyarrow_table(pd.DataFrame()))


class ToDictTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("int8", pa.int8(), np.int8),
      ("int16", pa.int16(), np.int16),
      ("int32", pa.int32(), np.int32),
      ("int64", pa.int64(), np.int64),
      ("uint8", pa.uint8(), np.uint8),
      ("uint16", pa.uint16(), np.uint16),
      ("uint32", pa.uint32(), np.uint32),
      ("uint64", pa.uint64(), np.uint64),
      ("float32", pa.float32(), np.float32),
      ("float64", pa.float64(), np.float64),
  )
  def test_numerical(self, arrow_type, expected_np_type):
    table = pa.table({"a": pa.array([1, 2, 3], type=arrow_type)})
    data = pyarrow_io.to_dict(table)
    self.assertEqual(list(data), ["a"])
    self.assertEqual(_column(data, "a").dtype.type, expected_np_type)
    npt.assert_array_equal(_column(data, "a"), np.array([1, 2, 3]))

  def test_float_with_missing_values(self):
    table = pa.table({"a": pa.array([1.0, None, 3.0], type=pa.float64())})
    data = pyarrow_io.to_dict(table)
    self.assertEqual(_column(data, "a").dtype.type, np.float64)
    npt.assert_array_equal(_column(data, "a"), np.array([1.0, np.nan, 3.0]))

  def test_integer_with_missing_values_becomes_float(self):
    # Numpy integers cannot represent missing values, so the column is
    # converted to floating point values, with NaN for the missing values.
    table = pa.table({"a": pa.array([1, None, 3], type=pa.int64())})
    data = pyarrow_io.to_dict(table)
    self.assertEqual(_column(data, "a").dtype.type, np.float64)
    npt.assert_array_equal(_column(data, "a"), np.array([1.0, np.nan, 3.0]))

  def test_boolean(self):
    table = pa.table({"a": pa.array([True, False, True])})
    data = pyarrow_io.to_dict(table)
    self.assertEqual(_column(data, "a").dtype.type, np.bool_)
    npt.assert_array_equal(_column(data, "a"), np.array([True, False, True]))

  def test_boolean_with_missing_values(self):
    # Numpy booleans cannot represent missing values, so the column is fed as
    # an array of objects. YDF then treats it as a categorical column.
    table = pa.table({"a": pa.array([True, None, False])})
    data = pyarrow_io.to_dict(table)
    self.assertEqual(_column(data, "a").dtype.type, np.object_)
    self.assertEqual(_column(data, "a").tolist(), [True, None, False])

  @parameterized.named_parameters(
      ("string", pa.string()),
      ("large_string", pa.large_string()),
  )
  def test_string(self, arrow_type):
    table = pa.table({"a": pa.array(["x", "y", "z"], type=arrow_type)})
    data = pyarrow_io.to_dict(table)
    self.assertEqual(_column(data, "a").dtype.type, np.object_)
    self.assertEqual(_column(data, "a").tolist(), ["x", "y", "z"])

  def test_string_with_missing_values(self):
    # YDF interprets an empty string as a missing value.
    table = pa.table({"a": pa.array(["x", None, "z"])})
    data = pyarrow_io.to_dict(table)
    self.assertEqual(_column(data, "a").tolist(), ["x", "", "z"])

  def test_binary_with_missing_values(self):
    table = pa.table({"a": pa.array([b"x", None, b"z"], type=pa.binary())})
    data = pyarrow_io.to_dict(table)
    self.assertEqual(_column(data, "a").tolist(), [b"x", b"", b"z"])

  def test_dictionary_is_decoded(self):
    table = pa.table({"a": pa.array(["x", "y", "x"]).dictionary_encode()})
    data = pyarrow_io.to_dict(table)
    self.assertEqual(_column(data, "a").dtype.type, np.object_)
    self.assertEqual(_column(data, "a").tolist(), ["x", "y", "x"])

  def test_decimal_becomes_float32(self):
    table = pa.table({"a": pa.array(["1.5", "2.5"]).cast(pa.decimal128(5, 2))})
    data = pyarrow_io.to_dict(table)
    self.assertEqual(_column(data, "a").dtype.type, np.float32)
    npt.assert_array_equal(_column(data, "a"), np.array([1.5, 2.5]))

  def test_null_column_is_fully_missing(self):
    table = pa.table({"a": pa.array([None, None], type=pa.null())})
    data = pyarrow_io.to_dict(table)
    self.assertEqual(_column(data, "a").dtype.type, np.object_)
    self.assertEqual(_column(data, "a").tolist(), ["", ""])

  @parameterized.named_parameters(
      ("list", pa.list_(pa.string())),
      ("large_list", pa.large_list(pa.string())),
  )
  def test_list_of_strings(self, arrow_type):
    # Variable-length lists are fed as an array of arrays, which YDF detects
    # as a categorical-set column.
    table = pa.table({"a": pa.array([["x"], ["x", "y"], []], type=arrow_type)})
    data = pyarrow_io.to_dict(table)
    self.assertEqual(_column(data, "a").dtype.type, np.object_)
    self.assertEqual(
        [list(v) for v in _column(data, "a")], [["x"], ["x", "y"], []]
    )

  def test_list_of_numbers(self):
    table = pa.table(
        {"a": pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float32()))}
    )
    data = pyarrow_io.to_dict(table)
    self.assertEqual(_column(data, "a").dtype.type, np.object_)
    npt.assert_array_equal(
        np.vstack(_column(data, "a").tolist()), [[1, 2], [3, 4]]
    )

  def test_fixed_size_list(self):
    table = pa.table({
        "a": pa.array(
            [[1.0, 2.0], [3.0, 4.0]],
            type=pa.list_(pa.float32(), list_size=2),
        )
    })
    data = pyarrow_io.to_dict(table)
    self.assertEqual(_column(data, "a").dtype.type, np.object_)
    npt.assert_array_equal(
        np.vstack(_column(data, "a").tolist()), [[1, 2], [3, 4]]
    )

  def test_multiple_columns(self):
    table = pa.table({
        "a": [1, 2],
        "b": ["x", "y"],
        "c": [True, False],
    })
    data = pyarrow_io.to_dict(table)
    self.assertEqual(list(data), ["a", "b", "c"])

  def test_chunked_table(self):
    table = pa.concat_tables([
        pa.table({"a": [1, 2], "b": ["x", "y"]}),
        pa.table({"a": [3], "b": ["z"]}),
    ])
    self.assertGreater(table.column("a").num_chunks, 1)
    data = pyarrow_io.to_dict(table)
    npt.assert_array_equal(_column(data, "a"), np.array([1, 2, 3]))
    self.assertEqual(_column(data, "b").tolist(), ["x", "y", "z"])

  def test_record_batch_equals_table(self):
    batch = pa.record_batch({"a": [1, 2], "b": ["x", "y"]})
    from_batch = pyarrow_io.to_dict(batch)
    from_table = pyarrow_io.to_dict(pa.Table.from_batches([batch]))
    self.assertEqual(list(from_batch), list(from_table))
    for key in from_batch:
      npt.assert_array_equal(_column(from_batch, key), _column(from_table, key))

  def test_empty_table(self):
    data = pyarrow_io.to_dict(pa.table({}))
    self.assertEmpty(data)

  def test_table_without_rows(self):
    table = pa.table({"a": pa.array([], type=pa.int64())})
    data = pyarrow_io.to_dict(table)
    self.assertEmpty(_column(data, "a"))

  def test_duplicated_column_names(self):
    table = pa.Table.from_arrays(
        [pa.array([1]), pa.array([2])], names=["a", "a"]
    )
    with self.assertRaisesRegex(ValueError, "duplicated"):
      pyarrow_io.to_dict(table)

  @parameterized.named_parameters(
      ("timestamp", [datetime.datetime(2024, 1, 2, 3, 4)], pa.timestamp("us")),
      ("date32", [datetime.date(2024, 1, 2)], pa.date32()),
      ("time64", [datetime.time(1, 2, 3)], pa.time64("us")),
      ("duration", [datetime.timedelta(seconds=1)], pa.duration("s")),
  )
  def test_temporal_columns_are_not_supported(self, values, arrow_type):
    table = pa.table({"a": pa.array(values, type=arrow_type)})
    with self.assertRaisesRegex(ValueError, "temporal type"):
      pyarrow_io.to_dict(table)

  def test_struct_columns_are_not_supported(self):
    table = pa.table({"a": pa.array([{"x": 1}, {"x": 2}])})
    with self.assertRaisesRegex(ValueError, "nested type"):
      pyarrow_io.to_dict(table)

  def test_map_columns_are_not_supported(self):
    table = pa.table(
        {"a": pa.array([[("x", 1)]], type=pa.map_(pa.string(), pa.int64()))}
    )
    with self.assertRaisesRegex(ValueError, "nested type"):
      pyarrow_io.to_dict(table)


if __name__ == "__main__":
  absltest.main()
