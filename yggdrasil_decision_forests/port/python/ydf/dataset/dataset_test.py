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

import enum
import os
from typing import Optional
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import fastavro
import numpy as np
import pandas as pd

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from ydf.dataset import dataset as dataset_lib
from ydf.dataset import dataspec as dataspec_lib
from ydf.dataset.io import dataset_io as dataset_io_lib
from ydf.utils import test_utils

# Make "assertEqual" print more details.
unittest.util._MAX_LENGTH = 10000

Semantic = dataspec_lib.Semantic
VocabValue = ds_pb.CategoricalSpec.VocabValue
Column = dataspec_lib.Column


class GenericDatasetTest(parameterized.TestCase):

  @parameterized.parameters(
      (np.array([1], np.int8), Semantic.NUMERICAL),
      (np.array([1], np.int16), Semantic.NUMERICAL),
      (np.array([1], np.int32), Semantic.NUMERICAL),
      (np.array([1], np.int64), Semantic.NUMERICAL),
      (np.array([1], np.uint8), Semantic.NUMERICAL),
      (np.array([1], np.uint16), Semantic.NUMERICAL),
      (np.array([1], np.uint32), Semantic.NUMERICAL),
      (np.array([1], np.uint64), Semantic.NUMERICAL),
      (np.array([1], np.float32), Semantic.NUMERICAL),
      (np.array([1], np.float64), Semantic.NUMERICAL),
      (np.array([1], np.bool_), Semantic.BOOLEAN),
      (np.array(["a"], np.bytes_), Semantic.CATEGORICAL),
      (np.array(["a", np.nan], np.object_), Semantic.CATEGORICAL),
  )
  def test_infer_semantic(self, value, expected_semantic):
    self.assertEqual(
        dataset_lib.infer_semantic("", value, discretize_numerical=False),
        expected_semantic,
    )

  @parameterized.parameters(
      (np.array([1], np.int8), Semantic.DISCRETIZED_NUMERICAL),
      (np.array([1], np.int16), Semantic.DISCRETIZED_NUMERICAL),
      (np.array([1], np.int32), Semantic.DISCRETIZED_NUMERICAL),
      (np.array([1], np.int64), Semantic.DISCRETIZED_NUMERICAL),
      (np.array([1], np.uint8), Semantic.DISCRETIZED_NUMERICAL),
      (np.array([1], np.uint16), Semantic.DISCRETIZED_NUMERICAL),
      (np.array([1], np.uint32), Semantic.DISCRETIZED_NUMERICAL),
      (np.array([1], np.uint64), Semantic.DISCRETIZED_NUMERICAL),
      (np.array([1], np.float32), Semantic.DISCRETIZED_NUMERICAL),
      (np.array([1], np.float64), Semantic.DISCRETIZED_NUMERICAL),
      (np.array([1], np.bool_), Semantic.BOOLEAN),
      (np.array(["a"], np.bytes_), Semantic.CATEGORICAL),
      (np.array(["a", np.nan], np.object_), Semantic.CATEGORICAL),
  )
  def test_infer_semantic_discretized(self, value, expected_semantic):
    self.assertEqual(
        dataset_lib.infer_semantic("", value, discretize_numerical=True),
        expected_semantic,
    )

  @parameterized.parameters(
      np.float16,
      np.float32,
      np.float64,
      np.int8,
      np.int16,
      np.int32,
      np.int64,
  )
  def test_create_vds_pd_numerical(self, dtype):
    df = pd.DataFrame(
        {
            "col_pos": [1, 2, 3],
            "col_neg": [-1, -2, -3],
            "col_zero": [0, 0, 0],
        },
        dtype=dtype,
    )
    ds = dataset_lib.create_vertical_dataset(df)
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="col_pos",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=dataspec_lib.np_dtype_to_ydf_dtype(dtype),
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=2,
                    standard_deviation=0.8164965809277263,  # ~math.sqrt(2 / 3)
                    min_value=1,
                    max_value=3,
                ),
            ),
            ds_pb.Column(
                name="col_neg",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=dataspec_lib.np_dtype_to_ydf_dtype(dtype),
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=-2,
                    standard_deviation=0.8164965809277263,  # ~math.sqrt(2 / 3)
                    min_value=-3,
                    max_value=-1,
                ),
            ),
            ds_pb.Column(
                name="col_zero",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=dataspec_lib.np_dtype_to_ydf_dtype(dtype),
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=0,
                    standard_deviation=0,
                    min_value=0,
                    max_value=0,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  @parameterized.parameters(np.float16, np.float32, np.float64)
  def test_create_vds_pd_numerical_with_nan(self, dtype):
    df = pd.DataFrame(
        {
            "col_single_nan": [1, 2, np.nan],
            "col_nan_only": [np.nan, np.nan, np.nan],
        },
        dtype=dtype,
    )
    ds = dataset_lib.create_vertical_dataset(df)
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="col_single_nan",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=dataspec_lib.np_dtype_to_ydf_dtype(dtype),
                count_nas=1,
                numerical=ds_pb.NumericalSpec(
                    mean=1.5,
                    standard_deviation=0.5,
                    min_value=1,
                    max_value=2,
                ),
            ),
            ds_pb.Column(
                name="col_nan_only",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=dataspec_lib.np_dtype_to_ydf_dtype(dtype),
                count_nas=3,
                numerical=ds_pb.NumericalSpec(),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_create_vds_pd_categorical_string(self):
    df = pd.DataFrame({
        "col1": ["A", "A", "B", "B", "C"],
        "col2": ["", "A", "B", "C", "D"],
    })

    ds = dataset_lib.create_vertical_dataset(
        df,
        min_vocab_frequency=2,
        columns=[
            Column(
                "col2",
                Semantic.CATEGORICAL,
                min_vocab_frequency=1,
                max_vocab_count=3,
            )
        ],
        include_all_columns=True,
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=5,
        columns=(
            ds_pb.Column(
                name="col2",
                type=ds_pb.ColumnType.CATEGORICAL,
                dtype=ds_pb.DType.DTYPE_BYTES,
                count_nas=1,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=1),
                        "A": VocabValue(index=1, count=1),
                        "B": VocabValue(index=2, count=1),
                        "C": VocabValue(index=3, count=1),
                    },
                    number_of_unique_values=4,
                ),
            ),
            ds_pb.Column(
                name="col1",
                type=ds_pb.ColumnType.CATEGORICAL,
                dtype=ds_pb.DType.DTYPE_BYTES,
                count_nas=0,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=1),
                        "A": VocabValue(index=1, count=2),
                        "B": VocabValue(index=2, count=2),
                    },
                    number_of_unique_values=3,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_create_vds_pd_categorical_int(self):
    df = pd.DataFrame({
        "col1": [1, 1, 2, 2, 3],
    })

    ds = dataset_lib.create_vertical_dataset(
        df,
        min_vocab_frequency=1,
        columns=[
            Column(
                "col1",
                Semantic.CATEGORICAL,
            )
        ],
        include_all_columns=True,
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=5,
        columns=(
            ds_pb.Column(
                name="col1",
                type=ds_pb.ColumnType.CATEGORICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "1": VocabValue(index=1, count=2),
                        "2": VocabValue(index=2, count=2),
                        "3": VocabValue(index=3, count=1),
                    },
                    number_of_unique_values=4,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_create_vds_pd_boolean(self):
    df = pd.DataFrame(
        {"col_boolean": [True, False, False]},
    )

    ds = dataset_lib.create_vertical_dataset(df)
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="col_boolean",
                type=ds_pb.ColumnType.BOOLEAN,
                dtype=ds_pb.DType.DTYPE_BOOL,
                count_nas=0,
                boolean=ds_pb.BooleanSpec(
                    count_true=1,
                    count_false=2,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_create_vds_pd_hash(self):
    df = pd.DataFrame(
        {"col_hash": ["a", "b", "abc"]},
    )

    ds = dataset_lib.create_vertical_dataset(
        df, columns=[("col_hash", Semantic.HASH)]
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="col_hash",
                type=ds_pb.ColumnType.HASH,
                dtype=ds_pb.DType.DTYPE_BYTES,
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  @parameterized.parameters(
      (["col_numerical"],),
      ([Column("col_numerical")],),
      ([("col_numerical", Semantic.NUMERICAL)],),
  )
  def test_create_vds_exclude_columns(self, column_definition):
    df = pd.DataFrame({
        "col_numerical": [1, 2, 3],
        "col_str": ["A", "B", "C"],
    })
    ds = dataset_lib.create_vertical_dataset(df, columns=column_definition)
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="col_numerical",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=2,
                    standard_deviation=0.8164965809277263,  # ~math.sqrt(2 / 3)
                    min_value=1,
                    max_value=3,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_create_vds_dict_of_values(self):
    ds_dict = {
        "a": np.array([1, 2, 3]),
        "b": np.array([-1, -2, -3]),
    }
    ds = dataset_lib.create_vertical_dataset(ds_dict)
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="a",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=2,
                    standard_deviation=0.8164965809277263,  # ~math.sqrt(2 / 3)
                    min_value=1,
                    max_value=3,
                ),
            ),
            ds_pb.Column(
                name="b",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=-2,
                    standard_deviation=0.8164965809277263,  # ~math.sqrt(2 / 3)
                    min_value=-3,
                    max_value=-1,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_memory_usage(self):
    df = pd.DataFrame({
        "col1": [1.1, 21.1, np.nan],
        "col2": [1, 2, 3],
    })
    ds = dataset_lib.create_vertical_dataset(df)
    # 2 columns * 3 rows * 4 bytes per values
    self.assertEqual(ds.memory_usage(), 2 * 3 * 4)

  def test_create_vds_pd_with_spec(self):
    data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="col_num",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=2,
                    standard_deviation=0.8164965809277263,  # ~math.sqrt(2 / 3)
                    min_value=1,
                    max_value=3,
                ),
            ),
            ds_pb.Column(
                name="col_cat",
                type=ds_pb.ColumnType.CATEGORICAL,
                dtype=ds_pb.DType.DTYPE_BYTES,
                count_nas=0,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=1),
                        "A": VocabValue(index=1, count=2),
                        "B": VocabValue(index=2, count=2),
                    },
                    number_of_unique_values=3,
                ),
            ),
            ds_pb.Column(
                name="col_bool",
                type=ds_pb.ColumnType.BOOLEAN,
                dtype=ds_pb.DType.DTYPE_BOOL,
                count_nas=0,
                boolean=ds_pb.BooleanSpec(
                    count_true=55,
                    count_false=123,
                ),
            ),
        ),
    )
    df = pd.DataFrame({
        "col_num": [1, 2, 3],
        "col_cat": ["A", "B", "C"],
        "col_bool": [True, True, False],
    })
    ds = dataset_lib.create_vertical_dataset(df, data_spec=data_spec)
    test_utils.assertProto2Equal(self, ds.data_spec(), data_spec)

  def test_create_vds_dict_of_values_with_spec(self):
    data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="col_num",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=2,
                    standard_deviation=0.8164965809277263,  # ~math.sqrt(2 / 3)
                    min_value=1,
                    max_value=3,
                ),
            ),
            ds_pb.Column(
                name="col_cat",
                type=ds_pb.ColumnType.CATEGORICAL,
                dtype=ds_pb.DType.DTYPE_BYTES,
                count_nas=0,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=1),
                        "A": VocabValue(index=1, count=2),
                        "B": VocabValue(index=2, count=2),
                    },
                    number_of_unique_values=3,
                ),
            ),
            ds_pb.Column(
                name="col_bool",
                type=ds_pb.ColumnType.BOOLEAN,
                dtype=ds_pb.DType.DTYPE_BOOL,
                count_nas=0,
                boolean=ds_pb.BooleanSpec(
                    count_true=55,
                    count_false=123,
                ),
            ),
        ),
    )
    ds_dict = {
        "col_num": [1, 2, 3],
        "col_cat": ["A", "B", "C"],
        "col_bool": [True, False, True],
    }
    ds = dataset_lib.create_vertical_dataset(ds_dict, data_spec=data_spec)
    test_utils.assertProto2Equal(self, ds.data_spec(), data_spec)
    # 2 columns * 3 rows * 4 bytes per value + 1 col * 3 rows * 1 byte p.v.
    self.assertEqual(ds.memory_usage(), 2 * 3 * 4 + 1 * 3 * 1)

  def test_create_vds_pd_check_contents(self):
    df = pd.DataFrame({
        "col_str": ["A", "string", "column with", "four entries"],
        "col_int": [5, 6, 7, 8],
        "col_int_cat": [1, 2, 3, 4],
        "col_float": [1.1, 2.2, 3.3, 4.4],
        "col_bool": [True, True, False, False],
    })
    feature_definitions = [
        Column("col_str", Semantic.CATEGORICAL, min_vocab_frequency=1),
        Column("col_int_cat", Semantic.CATEGORICAL, min_vocab_frequency=1),
    ]
    ds = dataset_lib.create_vertical_dataset(
        df, columns=feature_definitions, include_all_columns=True
    )
    expected_dataset_content = """col_str,col_int_cat,col_int,col_float,col_bool
A,1,5,1.1,1
string,2,6,2.2,1
column with,3,7,3.3,0
four entries,4,8,4.4,0
"""
    self.assertEqual(expected_dataset_content, ds._dataset.DebugString())

  def test_create_vds_pd_check_categorical(self):
    df = pd.DataFrame({"col_str": ["A", "B"] * 5})
    ds = dataset_lib.create_vertical_dataset(df)
    expected_dataset_content = "col_str\n" + "A\nB\n" * 5
    self.assertEqual(expected_dataset_content, ds._dataset.DebugString())

  def test_max_vocab_count_minus_1(self):
    df = pd.DataFrame({
        "col1": ["A", "B", "C", "D", "D"],
    })
    ds = dataset_lib.create_vertical_dataset(
        df,
        columns=[
            Column(
                "col1",
                Semantic.CATEGORICAL,
                min_vocab_frequency=1,
                max_vocab_count=-1,
            )
        ],
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=5,
        columns=(
            ds_pb.Column(
                name="col1",
                type=ds_pb.ColumnType.CATEGORICAL,
                dtype=ds_pb.DType.DTYPE_BYTES,
                count_nas=0,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "A": VocabValue(index=2, count=1),
                        "B": VocabValue(index=3, count=1),
                        "C": VocabValue(index=4, count=1),
                        "D": VocabValue(index=1, count=2),
                    },
                    number_of_unique_values=5,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_invalid_max_vocab_count(self):
    df = pd.DataFrame({
        "col1": ["A", "B", "C", "D", "D"],
    })
    with self.assertRaisesRegex(
        test_utils.AbslInvalidArgumentError, "max_vocab_count"
    ):
      dataset_lib.create_vertical_dataset(
          df,
          columns=[
              Column(
                  "col1",
                  Semantic.CATEGORICAL,
                  max_vocab_count=-2,
              )
          ],
      )

  @parameterized.parameters(
      ([True, True, True], (0, 0, 3), 0),
      ([False, False, False], (0, 3, 0), 0),
      ([True, False, False], (0, 2, 1), 0),
  )
  def test_order_boolean(self, values, expected_counts, count_nas):
    ds = dataset_lib.create_vertical_dataset(
        {"col": np.array(values)},
        columns=[Column("col", Semantic.CATEGORICAL)],
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="col",
                type=ds_pb.ColumnType.CATEGORICAL,
                dtype=ds_pb.DType.DTYPE_BOOL,
                count_nas=count_nas,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=expected_counts[0]),
                        "false": VocabValue(index=1, count=expected_counts[1]),
                        "true": VocabValue(index=2, count=expected_counts[2]),
                    },
                    number_of_unique_values=3,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_order_integers(self):
    ds = dataset_lib.create_vertical_dataset(
        {"col": np.array([0, 1, 4, 3, 1, 2, 3, 4, 12, 11, 10, 9, 8, 7, 6, 5])},
        columns=[Column("col", Semantic.CATEGORICAL)],
        label="col",
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=16,
        columns=(
            ds_pb.Column(
                name="col",
                type=ds_pb.ColumnType.CATEGORICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "0": VocabValue(index=1, count=1),
                        "1": VocabValue(index=2, count=2),
                        "2": VocabValue(index=3, count=1),
                        "3": VocabValue(index=4, count=2),
                        "4": VocabValue(index=5, count=2),
                        "5": VocabValue(index=6, count=1),
                        "6": VocabValue(index=7, count=1),
                        "7": VocabValue(index=8, count=1),
                        "8": VocabValue(index=9, count=1),
                        "9": VocabValue(index=10, count=1),
                        "10": VocabValue(index=11, count=1),
                        "11": VocabValue(index=12, count=1),
                        "12": VocabValue(index=13, count=1),
                    },
                    number_of_unique_values=14,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  @parameterized.parameters("", "csv:")
  def test_read_csv(self, path_prefix):
    tmp_dir = self.create_tempdir()
    csv_file = self.create_tempfile(
        content="""col_cat,col_num
A,1
B,2
B,3""",
        file_path=os.path.join(tmp_dir.full_path, "file.csv"),
    )
    ds = dataset_lib.create_vertical_dataset(
        path_prefix + csv_file.full_path, min_vocab_frequency=1
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="col_cat",
                type=ds_pb.ColumnType.CATEGORICAL,
                is_manual_type=False,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "A": VocabValue(index=2, count=1),
                        "B": VocabValue(index=1, count=2),
                    },
                    number_of_unique_values=3,
                    most_frequent_value=1,
                    min_value_count=1,
                    max_number_of_unique_values=2000,
                    is_already_integerized=False,
                ),
            ),
            ds_pb.Column(
                name="col_num",
                type=ds_pb.ColumnType.NUMERICAL,
                is_manual_type=False,
                numerical=ds_pb.NumericalSpec(
                    mean=2,
                    standard_deviation=0.8164965809277263,  # ~math.sqrt(2 / 3)
                    min_value=1,
                    max_value=3,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_max_num_scanned_rows_to_compute_statistics(self):
    tmp_dir = self.create_tempdir()
    csv_file = self.create_tempfile(
        content="""col_cat,col_num
A,1
A,2
B,3""",
        file_path=os.path.join(tmp_dir.full_path, "file.csv"),
    )
    ds = dataset_lib.create_vertical_dataset(
        "csv:" + csv_file.full_path,
        min_vocab_frequency=1,
        max_num_scanned_rows_to_compute_statistics=2,
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="col_cat",
                type=ds_pb.ColumnType.CATEGORICAL,
                is_manual_type=False,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "A": VocabValue(index=1, count=2),
                    },
                    number_of_unique_values=2,
                    most_frequent_value=1,
                    min_value_count=1,
                    max_number_of_unique_values=2000,
                    is_already_integerized=False,
                ),
            ),
            ds_pb.Column(
                name="col_num",
                type=ds_pb.ColumnType.NUMERICAL,
                is_manual_type=False,
                numerical=ds_pb.NumericalSpec(
                    mean=1.5,
                    standard_deviation=0.5,
                    min_value=1,
                    max_value=2,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_max_num_scanned_rows_to_infer_semantic(self):
    tmp_dir = self.create_tempdir()
    csv_file = self.create_tempfile(
        content="""col_cat,col_num
A,1
B,2
B,3""",
        file_path=os.path.join(tmp_dir.full_path, "file.csv"),
    )
    ds = dataset_lib.create_vertical_dataset(
        "csv:" + csv_file.full_path,
        min_vocab_frequency=1,
        max_num_scanned_rows_to_infer_semantic=1,
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="col_cat",
                type=ds_pb.ColumnType.CATEGORICAL,
                is_manual_type=False,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=3,
                    most_frequent_value=1,
                    min_value_count=1,
                    max_number_of_unique_values=2000,
                    is_already_integerized=False,
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "A": VocabValue(index=2, count=1),
                        "B": VocabValue(index=1, count=2),
                    },
                ),
            ),
            ds_pb.Column(
                name="col_num",
                type=ds_pb.ColumnType.BOOLEAN,
                is_manual_type=False,
                boolean=ds_pb.BooleanSpec(count_true=3),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_read_from_path(self):
    csv_file = self.create_tempfile(content="""col_cat,col_num
A,1
B,2
B,3""")
    ds = dataset_lib.create_vertical_dataset(
        "csv:" + csv_file.full_path, min_vocab_frequency=1
    )
    df_pd = pd.read_csv(csv_file)
    ds_pd = dataset_lib.create_vertical_dataset(df_pd, data_spec=ds.data_spec())
    self.assertEqual(ds._dataset.DebugString(), ds_pd._dataset.DebugString())

  @unittest.skip("Requires building YDF with tensorflow io")
  def test_read_from_sharded_tfe(self):
    sharded_path = "tfrecord:" + os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "toy.tfe-tfrecord@2"
    )
    ds = dataset_lib.create_vertical_dataset(
        sharded_path,
        min_vocab_frequency=1,
        columns=["Bool_1", "Cat_2", "Num_1"],
    )
    expected_data_spec = ds_pb.DataSpecification(
        columns=(
            ds_pb.Column(
                name="Bool_1",
                type=ds_pb.BOOLEAN,
                dtype=ds_pb.DType.DTYPE_INT64,
                is_manual_type=False,
                boolean=ds_pb.BooleanSpec(count_true=2, count_false=2),
            ),
            ds_pb.Column(
                name="Cat_2",
                type=ds_pb.CATEGORICAL,
                dtype=ds_pb.DType.DTYPE_BYTES,
                is_manual_type=False,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=3,
                    most_frequent_value=1,
                    min_value_count=1,
                    max_number_of_unique_values=2000,
                    is_already_integerized=False,
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "A": VocabValue(index=2, count=1),
                        "B": VocabValue(index=1, count=1),
                    },
                ),
                count_nas=2,
            ),
            ds_pb.Column(
                name="Num_1",
                type=ds_pb.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_FLOAT32,
                is_manual_type=False,
                numerical=ds_pb.NumericalSpec(
                    mean=2.5,
                    min_value=1.0,
                    max_value=4.0,
                    standard_deviation=1.118033988749895,
                ),
            ),
        ),
        created_num_rows=4,
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_read_from_sharded_tfe_nocompress(self):
    sharded_path = "tfrecordv2+tfe:" + os.path.join(
        test_utils.ydf_test_data_path(),
        "dataset",
        "toy.nocompress-tfe-tfrecord@2",
    )
    ds = dataset_lib.create_vertical_dataset(
        sharded_path,
        min_vocab_frequency=1,
        columns=["Bool_1", "Cat_2", "Num_1"],
    )
    expected_data_spec = ds_pb.DataSpecification(
        columns=(
            ds_pb.Column(
                name="Bool_1",
                type=ds_pb.BOOLEAN,
                dtype=ds_pb.DType.DTYPE_INT64,
                is_manual_type=False,
                boolean=ds_pb.BooleanSpec(count_true=2, count_false=2),
            ),
            ds_pb.Column(
                name="Cat_2",
                type=ds_pb.CATEGORICAL,
                dtype=ds_pb.DType.DTYPE_BYTES,
                is_manual_type=False,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=3,
                    most_frequent_value=1,
                    min_value_count=1,
                    max_number_of_unique_values=2000,
                    is_already_integerized=False,
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "A": VocabValue(index=2, count=1),
                        "B": VocabValue(index=1, count=1),
                    },
                ),
                count_nas=2,
            ),
            ds_pb.Column(
                name="Num_1",
                type=ds_pb.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_FLOAT32,
                is_manual_type=False,
                numerical=ds_pb.NumericalSpec(
                    mean=2.5,
                    min_value=1.0,
                    max_value=4.0,
                    standard_deviation=1.118033988749895,
                ),
            ),
        ),
        created_num_rows=4,
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_numerical_float(self):
    ds = dataset_lib.create_vertical_dataset(
        {"feature": np.array([1.0, 2.0, 3.0], np.float32)}
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_FLOAT32,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=2,
                    standard_deviation=0.8164965809277263,  # ~math.sqrt(2 / 3)
                    min_value=1,
                    max_value=3,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_categorical_invalid_type_float(self):
    with self.assertRaisesRegex(
        ValueError,
        "Cannot import column 'feature' with semantic=Semantic.CATEGORICAL as"
        " it contains floating point values.",
    ):
      _ = dataset_lib.create_vertical_dataset(
          {"feature": np.array([1.0, 2.0, 3.0])},
          min_vocab_frequency=1,
          columns=[Column("feature", Semantic.CATEGORICAL)],
      )

  def test_categorical_invalid_type_none(self):
    with self.assertRaisesRegex(
        ValueError,
        "Cannot import column 'feature' with semantic=Semantic.CATEGORICAL",
    ):
      _ = dataset_lib.create_vertical_dataset(
          {"feature": np.array(["x", "y", None])},
          min_vocab_frequency=1,
          columns=[Column("feature", Semantic.CATEGORICAL)],
      )

  def test_catset_invalid_type_float(self):
    with self.assertRaisesRegex(
        ValueError,
        "Cannot import column 'feature' with semantic=Semantic.CATEGORICAL_SET"
        " as it contains floating point values.",
    ):
      _ = dataset_lib.create_vertical_dataset(
          {"feature": np.array([[1.0, 2.0, 3.0]], np.object_)},
          min_vocab_frequency=1,
          columns=[Column("feature", Semantic.CATEGORICAL_SET)],
      )

  def test_catset_invalid_type_none(self):
    with self.assertRaisesRegex(
        ValueError,
        "Cannot import column 'feature' with semantic=Semantic.CATEGORICAL_SET",
    ):
      _ = dataset_lib.create_vertical_dataset(
          {"feature": np.array([["x", "y", None]], np.object_)},
          min_vocab_frequency=1,
          columns=[Column("feature", Semantic.CATEGORICAL_SET)],
      )

  def test_catset_string(self):
    ds = dataset_lib.create_vertical_dataset(
        {"feature": np.array([["x", "y"], ["z", "w"]], np.object_)},
        min_vocab_frequency=1,
        columns=[Column("feature", Semantic.CATEGORICAL_SET)],
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.CATEGORICAL_SET,
                dtype=ds_pb.DType.DTYPE_BYTES,
                count_nas=0,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=5,
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "w": VocabValue(index=1, count=1),
                        "x": VocabValue(index=2, count=1),
                        "y": VocabValue(index=3, count=1),
                        "z": VocabValue(index=4, count=1),
                    },
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_catset_bytes(self):
    ds = dataset_lib.create_vertical_dataset(
        {"feature": np.array([[b"x", b"y"], [b"z", b"w"]], np.object_)},
        min_vocab_frequency=1,
        columns=[Column("feature", Semantic.CATEGORICAL_SET)],
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.CATEGORICAL_SET,
                dtype=ds_pb.DType.DTYPE_BYTES,
                count_nas=0,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=5,
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "w": VocabValue(index=1, count=1),
                        "x": VocabValue(index=2, count=1),
                        "y": VocabValue(index=3, count=1),
                        "z": VocabValue(index=4, count=1),
                    },
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_catset_bool(self):
    ds = dataset_lib.create_vertical_dataset(
        {"feature": np.array([[True, False], [False, True]], np.object_)},
        min_vocab_frequency=1,
        columns=[Column("feature", Semantic.CATEGORICAL_SET)],
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.CATEGORICAL_SET,
                dtype=ds_pb.DType.DTYPE_BYTES,
                count_nas=0,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=3,
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "false": VocabValue(index=1, count=2),
                        "true": VocabValue(index=2, count=2),
                    },
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_catset_int(self):
    ds = dataset_lib.create_vertical_dataset(
        {"feature": np.array([[0, 1, 2], [4, 5, 6]], np.object_)},
        min_vocab_frequency=1,
        columns=[Column("feature", Semantic.CATEGORICAL_SET)],
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.CATEGORICAL_SET,
                dtype=ds_pb.DType.DTYPE_BYTES,
                count_nas=0,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=7,
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "0": VocabValue(index=1, count=1),
                        "1": VocabValue(index=2, count=1),
                        "2": VocabValue(index=3, count=1),
                        "4": VocabValue(index=4, count=1),
                        "5": VocabValue(index=5, count=1),
                        "6": VocabValue(index=6, count=1),
                    },
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_multidimensional_numerical(self):
    ds = dataset_lib.create_vertical_dataset(
        {"feature": np.array([[0, 1, 2], [4, 5, 6]])}
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="feature.0_of_3",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=2,
                    standard_deviation=2,
                    min_value=0,
                    max_value=4,
                ),
                is_unstacked=True,
            ),
            ds_pb.Column(
                name="feature.1_of_3",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=3,
                    standard_deviation=2,
                    min_value=1,
                    max_value=5,
                ),
                is_unstacked=True,
            ),
            ds_pb.Column(
                name="feature.2_of_3",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=4,
                    standard_deviation=2,
                    min_value=2,
                    max_value=6,
                ),
                is_unstacked=True,
            ),
        ),
        unstackeds=(
            ds_pb.Unstacked(
                original_name="feature",
                begin_column_idx=0,
                size=3,
                type=ds_pb.ColumnType.NUMERICAL,
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_multidimensional_categorical_int(self):
    ds = dataset_lib.create_vertical_dataset(
        {"feature": np.array([[0, 1, 2], [4, 5, 6]])},
        columns=[
            Column("feature", Semantic.CATEGORICAL, min_vocab_frequency=1)
        ],
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="feature.0_of_3",
                type=ds_pb.ColumnType.CATEGORICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=3,
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "0": VocabValue(index=1, count=1),
                        "4": VocabValue(index=2, count=1),
                    },
                ),
                is_unstacked=True,
            ),
            ds_pb.Column(
                name="feature.1_of_3",
                type=ds_pb.ColumnType.CATEGORICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=3,
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "1": VocabValue(index=1, count=1),
                        "5": VocabValue(index=2, count=1),
                    },
                ),
                is_unstacked=True,
            ),
            ds_pb.Column(
                name="feature.2_of_3",
                type=ds_pb.ColumnType.CATEGORICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=3,
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "2": VocabValue(index=1, count=1),
                        "6": VocabValue(index=2, count=1),
                    },
                ),
                is_unstacked=True,
            ),
        ),
        unstackeds=(
            ds_pb.Unstacked(
                original_name="feature",
                begin_column_idx=0,
                size=3,
                type=ds_pb.ColumnType.CATEGORICAL,
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_multidimensional_catset_object(self):
    ds = dataset_lib.create_vertical_dataset(
        {"feature": np.array([[0, 1, 2], [4, 5, 6]], dtype=np.object_)},
        min_vocab_frequency=1,
        columns=[Column("feature", Semantic.CATEGORICAL_SET)],
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.CATEGORICAL_SET,
                dtype=ds_pb.DType.DTYPE_BYTES,
                count_nas=0,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=7,
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "0": VocabValue(index=1, count=1),
                        "1": VocabValue(index=2, count=1),
                        "2": VocabValue(index=3, count=1),
                        "4": VocabValue(index=4, count=1),
                        "5": VocabValue(index=5, count=1),
                        "6": VocabValue(index=6, count=1),
                    },
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_multidimensional_ragged_catset_int(self):
    ds = dataset_lib.create_vertical_dataset(
        {
            "feature": np.array(
                [np.array([0, 1]), np.array([0, 1, 2, 3])], dtype=np.object_
            )
        },
        min_vocab_frequency=2,
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.CATEGORICAL_SET,
                dtype=ds_pb.DType.DTYPE_BYTES,
                count_nas=0,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=3,
                    items={
                        "<OOD>": VocabValue(index=0, count=2),
                        "0": VocabValue(index=1, count=2),
                        "1": VocabValue(index=2, count=2),
                    },
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_multidimensional_equal_sized_list_unroll(self):
    # Constructing the array directly will not trigger the same shape as first
    # constructing and then modifying it.
    non_uniform_array = np.array(
        [np.array([1, 2]), np.array([3])], dtype=object
    )
    non_uniform_array[1] = np.array([3, 4])
    ds = dataset_lib.create_vertical_dataset(
        {"f": non_uniform_array}, min_vocab_frequency=1
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="f.0_of_2",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=2.0,
                    min_value=1.0,
                    max_value=3.0,
                    standard_deviation=1.0,
                ),
                is_unstacked=True,
            ),
            ds_pb.Column(
                name="f.1_of_2",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=3.0,
                    min_value=2.0,
                    max_value=4.0,
                    standard_deviation=1.0,
                ),
                is_unstacked=True,
            ),
        ),
        unstackeds=(
            ds_pb.Unstacked(
                original_name="f",
                begin_column_idx=0,
                size=2,
                type=ds_pb.ColumnType.NUMERICAL,
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  @parameterized.parameters(
      (1, "feature.0_of_1"),
      (9, "feature.0_of_9"),
      (10, "feature.00_of_10"),
      (9999, "feature.0000_of_9999"),
      (10000, "feature.00000_of_10000"),
  )
  def test_multidimensional_feature_name(
      self, num_dims: int, expected_first_feature: str
  ):
    ds = dataset_lib.create_vertical_dataset(
        {"feature": np.zeros((3, num_dims))}
    )
    self.assertLen(ds.data_spec().columns, num_dims)
    self.assertEqual(ds.data_spec().columns[0].name, expected_first_feature)

  def test_multi_dimensions_error_too_many_dims(self):
    with self.assertRaisesRegex(
        ValueError, "Input features can only be one or two dimensional"
    ):
      _ = dataset_lib.create_vertical_dataset({"feature": np.zeros((3, 3, 3))})

  def test_list_of_csv_datasets(self):
    df = pd.DataFrame({
        "col_str": ["A", "string", "column with", "four entries"],
        "col_int": [5, 6, 7, 8],
        "col_int_cat": [1, 2, 3, 4],
        "col_float": [1.1, 2.2, 3.3, 4.4],
    })
    feature_definitions = [
        Column("col_str", Semantic.CATEGORICAL, min_vocab_frequency=1),
        Column("col_int_cat", Semantic.CATEGORICAL, min_vocab_frequency=1),
    ]
    dataset_directory = self.create_tempdir()
    path1 = os.path.join(dataset_directory.full_path, "ds1")
    path2 = os.path.join(dataset_directory.full_path, "ds2")
    df.head(3).to_csv(path1, index=False)
    df.tail(1).to_csv(path2, index=False)

    ds = dataset_lib.create_vertical_dataset(
        ["csv:" + path1, "csv:" + path2],
        columns=feature_definitions,
        include_all_columns=True,
    )

    expected_dataset_content = """\
col_str,col_int,col_int_cat,col_float
A,5,1,1.1
string,6,2,2.2
column with,7,3,3.3
four entries,8,4,4.4
"""
    self.assertEqual(expected_dataset_content, ds._dataset.DebugString())

  def test_singledimensional_strided_float32(self):
    data = np.array([[0, 1], [4, 5]], np.float32)
    feature = data[:, 0]  # "feature" shares the same memory as "data".
    self.assertEqual(data.strides, (8, 4))
    self.assertEqual(feature.strides, (8,))

    ds = dataset_lib.create_vertical_dataset({"feature": feature})
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_FLOAT32,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=2,
                    standard_deviation=2,
                    min_value=0,
                    max_value=4,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)
    self.assertEqual("feature\n0\n4\n", ds._dataset.DebugString())

  def test_singledimensional_strided_boolean(self):
    data = np.array([[True, False], [False, True]])
    feature = data[:, 0]  # "feature" shares the same memory as "data".
    self.assertEqual(data.strides, (2, 1))
    self.assertEqual(feature.strides, (2,))

    ds = dataset_lib.create_vertical_dataset({"feature": feature})
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.BOOLEAN,
                dtype=ds_pb.DType.DTYPE_BOOL,
                count_nas=0,
                boolean=ds_pb.BooleanSpec(
                    count_true=1,
                    count_false=1,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)
    self.assertEqual("feature\n1\n0\n", ds._dataset.DebugString())

  def test_multidimensional_strided(self):
    # Note: multidimensional features are unrolled into singledimensional
    # strided features.
    ds = dataset_lib.create_vertical_dataset(
        {"feature": np.array([[0, 1, 2], [4, 5, 6]], np.float32)}
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="feature.0_of_3",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_FLOAT32,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=2,
                    standard_deviation=2,
                    min_value=0,
                    max_value=4,
                ),
                is_unstacked=True,
            ),
            ds_pb.Column(
                name="feature.1_of_3",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_FLOAT32,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=3,
                    standard_deviation=2,
                    min_value=1,
                    max_value=5,
                ),
                is_unstacked=True,
            ),
            ds_pb.Column(
                name="feature.2_of_3",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_FLOAT32,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=4,
                    standard_deviation=2,
                    min_value=2,
                    max_value=6,
                ),
                is_unstacked=True,
            ),
        ),
        unstackeds=(
            ds_pb.Unstacked(
                original_name="feature",
                begin_column_idx=0,
                size=3,
                type=ds_pb.ColumnType.NUMERICAL,
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

    expected_dataset_content = """\
feature.0_of_3,feature.1_of_3,feature.2_of_3
0,1,2
4,5,6
"""
    self.assertEqual(expected_dataset_content, ds._dataset.DebugString())

  def test_warnings(self):
    # No issues.
    self.assertEmpty(
        dataset_lib.validate_dataspec(
            ds_pb.DataSpecification(
                columns=(
                    ds_pb.Column(
                        name="f",
                        type=ds_pb.ColumnType.CATEGORICAL,
                        categorical=ds_pb.CategoricalSpec(
                            items={
                                "<OOD>": VocabValue(index=0, count=0),
                                "A": VocabValue(index=1, count=1),
                                "B": VocabValue(index=2, count=1),
                                "C.": VocabValue(index=2, count=1),
                                "1.": VocabValue(index=2, count=1),
                            },
                        ),
                    ),
                ),
            ),
            [0],
        ),
    )

    # Look like a number
    bad_dataspec = ds_pb.DataSpecification(
        columns=(
            ds_pb.Column(
                name="f",
                type=ds_pb.ColumnType.CATEGORICAL,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "0": VocabValue(index=1, count=1),
                        "1": VocabValue(index=2, count=1),
                        "3.": VocabValue(index=2, count=1),
                        "4.": VocabValue(index=2, count=1),
                        "": VocabValue(index=3, count=1),
                    },
                ),
            ),
        ),
    )
    self.assertStartsWith(
        dataset_lib.validate_dataspec(bad_dataspec, [0])[0],
        "Column 'f' is detected as CATEGORICAL",
    )

    # Bad column not selected
    self.assertEmpty(dataset_lib.validate_dataspec(bad_dataspec, []))

  @parameterized.parameters("1", "1.", "1.0")
  def test_look_numerical(self, value: str):
    self.assertTrue(dataset_lib.look_numerical(value))

  @parameterized.parameters("", "a", "hello")
  def test_does_not_look_numerical(self, value: str):
    self.assertFalse(dataset_lib.look_numerical(value))

  def test_from_numpy(self):
    with self.assertRaisesRegex(
        ValueError, "YDF does not consume Numpy arrays directly"
    ):
      dataset_lib.create_vertical_dataset(np.array([1, 2, 3]))

  def test_from_column_less_pandas(self):
    with self.assertRaisesRegex(
        ValueError, "The pandas DataFrame must have string column names"
    ):
      dataset_lib.create_vertical_dataset(pd.DataFrame([[1, 2, 3], [4, 5, 6]]))

  def test_boolean_column(self):
    data = {
        "f1": np.array([True, True, True, False, False, False, False]),
    }
    ds = dataset_lib.create_vertical_dataset(
        data,
        columns=[("f1", dataspec_lib.Semantic.BOOLEAN)],
    )
    self.assertEqual(
        ds.data_spec(),
        ds_pb.DataSpecification(
            created_num_rows=7,
            columns=[
                ds_pb.Column(
                    name="f1",
                    type=ds_pb.ColumnType.BOOLEAN,
                    count_nas=0,
                    boolean=ds_pb.BooleanSpec(count_true=3, count_false=4),
                    dtype=ds_pb.DType.DTYPE_BOOL,
                )
            ],
        ),
    )
    self.assertEqual(ds._dataset.DebugString(), "f1\n1\n1\n1\n0\n0\n0\n0\n")

  def test_fail_gracefully_for_incorrect_boolean_type(self):
    data = {
        "f1": np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    }
    with self.assertRaisesRegex(
        test_utils.AbslInvalidArgumentError,
        "Cannot import column 'f1' with semantic=Semantic.BOOLEAN as it does"
        " not contain boolean values.*",
    ):
      dataset_lib.create_vertical_dataset(
          data,
          columns=[("f1", dataspec_lib.Semantic.BOOLEAN)],
      )

  def test_multidim_numerical_list(self):
    data = {
        "f1": [[1, 2], [3, 4], [5, 6]],
    }
    with self.assertRaisesRegex(
        ValueError,
        ".*Unrolling multi-dimensional columns is only supported for numpy"
        " arrays.*",
    ):
      _ = dataset_lib.create_vertical_dataset(
          data, columns=[("f1", dataspec_lib.Semantic.NUMERICAL)]
      )

  def test_multidim_boolean_list(self):
    data = {
        "f1": [[True, False], [True, False], [True, False]],
    }
    with self.assertRaisesRegex(
        ValueError,
        ".*Unrolling multi-dimensional columns is only supported for numpy"
        " arrays.*",
    ):
      _ = dataset_lib.create_vertical_dataset(
          data, columns=[("f1", dataspec_lib.Semantic.BOOLEAN)]
      )

  def test_multidim_categorical_list(self):
    data = {
        "f1": [[1, 2], [3, 4], [5, 6]],
    }
    with self.assertRaisesRegex(
        ValueError,
        ".*Unrolling multi-dimensional columns is only supported for numpy"
        " arrays.*",
    ):
      _ = dataset_lib.create_vertical_dataset(
          data, columns=[("f1", dataspec_lib.Semantic.CATEGORICAL)]
      )


class CategoricalSetTest(absltest.TestCase):

  def create_toy_csv(self) -> str:
    """Returns the path to a small csv file with sentences."""
    tmp_dir = self.create_tempdir()
    csv_file = self.create_tempfile(
        content="""\
col_cat_set
first sentence foo bar foo bar
second sentence foo bar foo foo foo""",
        file_path=os.path.join(tmp_dir.full_path, "file.csv"),
    )
    return csv_file.full_path

  def toy_csv_dataspec_categorical(self) -> ds_pb.DataSpecification:
    """Returns a dataspec for the toy CSV example with a categorical feature."""
    return ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="col_cat_set",
                type=ds_pb.ColumnType.CATEGORICAL,
                is_manual_type=False,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "first sentence foo bar foo bar": VocabValue(
                            index=2, count=1
                        ),
                        "second sentence foo bar foo foo foo": VocabValue(
                            index=1, count=1
                        ),
                    },
                    number_of_unique_values=3,
                    most_frequent_value=1,
                    min_value_count=1,
                    max_number_of_unique_values=2000,
                    is_already_integerized=False,
                ),
            ),
        ),
    )

  def toy_csv_dataspec_catset(self) -> ds_pb.DataSpecification:
    """Returns a dataspec for the toy CSV example with a catset feature."""
    return ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="col_cat_set",
                type=ds_pb.ColumnType.CATEGORICAL_SET,
                is_manual_type=True,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "foo": VocabValue(index=1, count=6),
                        "bar": VocabValue(index=2, count=3),
                        "sentence": VocabValue(index=3, count=2),
                        "second": VocabValue(index=4, count=1),
                        "first": VocabValue(index=5, count=1),
                    },
                    number_of_unique_values=6,
                    most_frequent_value=1,
                    min_value_count=1,
                    max_number_of_unique_values=2000,
                    is_already_integerized=False,
                ),
            ),
        ),
    )

  def test_csv_file_no_automatic_tokenization(self):
    path_to_csv = self.create_toy_csv()
    ds = dataset_lib.create_vertical_dataset(
        "csv:" + path_to_csv, min_vocab_frequency=1
    )
    expected_data_spec = self.toy_csv_dataspec_categorical()
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_csv_tokenization_when_semantic_specified(self):
    path_to_csv = self.create_toy_csv()
    ds = dataset_lib.create_vertical_dataset(
        "csv:" + path_to_csv,
        min_vocab_frequency=1,
        columns=[("col_cat_set", Semantic.CATEGORICAL_SET)],
    )
    expected_data_spec = self.toy_csv_dataspec_catset()
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_csv_file_reading_respects_data_spec_categorical(self):
    path_to_csv = self.create_toy_csv()
    data_spec = self.toy_csv_dataspec_categorical()
    ds = dataset_lib.create_vertical_dataset(
        "csv:" + path_to_csv, data_spec=data_spec
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), data_spec)
    self.assertEqual(
        ds._dataset.DebugString(),
        """\
col_cat_set
first sentence foo bar foo bar
second sentence foo bar foo foo foo
""",
    )

  def test_csv_file_reading_respects_data_spec_catset(self):
    path_to_csv = self.create_toy_csv()
    data_spec = self.toy_csv_dataspec_catset()
    ds = dataset_lib.create_vertical_dataset(
        "csv:" + path_to_csv, data_spec=data_spec
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), data_spec)
    self.assertEqual(
        ds._dataset.DebugString(),
        """\
col_cat_set
foo, bar, sentence, first
foo, bar, sentence, second
""",
    )

  def test_pd_ragged_list_of_lists(self):
    df = pd.DataFrame({
        "feature": [
            ["single item"],
            ["two", "words"],
            ["three", "simple", "words", "words"],
            [""],
        ]
    })
    ds = dataset_lib.create_vertical_dataset(
        df,
        min_vocab_frequency=1,
        columns=[("feature", Semantic.CATEGORICAL_SET)],
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=4,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.CATEGORICAL_SET,
                dtype=ds_pb.DType.DTYPE_BYTES,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "words": VocabValue(index=1, count=2),
                        "simple": VocabValue(index=2, count=1),
                        "single item": VocabValue(index=3, count=1),
                        "three": VocabValue(index=4, count=1),
                        "two": VocabValue(index=5, count=1),
                    },
                    number_of_unique_values=6,
                ),
                count_nas=1,
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_pd_invalid_type_text(self):
    df = pd.DataFrame({"feature": ["a", "b c", "d e f g h"]})
    with self.assertRaisesRegex(
        ValueError,
        "Cannot import column 'feature' with semantic=Semantic.CATEGORICAL_SET"
        " as it contains non-list values.",
    ):
      _ = dataset_lib.create_vertical_dataset(
          df,
          min_vocab_frequency=1,
          columns=[("feature", Semantic.CATEGORICAL_SET)],
      )

  def test_pd_np_bytes(self):
    df = pd.DataFrame({
        "feature": [
            np.array(["foo", "bar", "sentence", "first"]),
            np.array(["foo", "bar", "sentence", "second"]),
        ]
    })
    ds = dataset_lib.create_vertical_dataset(
        df,
        min_vocab_frequency=1,
        columns=[("feature", Semantic.CATEGORICAL_SET)],
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.CATEGORICAL_SET,
                dtype=ds_pb.DType.DTYPE_BYTES,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "bar": VocabValue(index=1, count=2),
                        "foo": VocabValue(index=2, count=2),
                        "sentence": VocabValue(index=3, count=2),
                        "first": VocabValue(index=4, count=1),
                        "second": VocabValue(index=5, count=1),
                    },
                    number_of_unique_values=6,
                ),
                count_nas=0,
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_pd_ragged_np_bytes(self):
    df = pd.DataFrame({
        "feature": [
            np.array(["single item"], np.bytes_),
            np.array(["two", "words"], np.bytes_),
            np.array(["three", "simple", "words", "words"], np.bytes_),
            np.array([""], np.bytes_),
        ]
    })
    ds = dataset_lib.create_vertical_dataset(
        df,
        min_vocab_frequency=1,
        columns=[("feature", Semantic.CATEGORICAL_SET)],
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=4,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.CATEGORICAL_SET,
                dtype=ds_pb.DType.DTYPE_BYTES,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "words": VocabValue(index=1, count=2),
                        "simple": VocabValue(index=2, count=1),
                        "single item": VocabValue(index=3, count=1),
                        "three": VocabValue(index=4, count=1),
                        "two": VocabValue(index=5, count=1),
                    },
                    number_of_unique_values=6,
                ),
                count_nas=1,
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_pd_with_na(self):
    df = pd.DataFrame({
        "feature": [
            pd.NA,
            ["single item"],
            ["two", "words"],
            ["three", "simple", "words", "words"],
        ]
    })
    ds = dataset_lib.create_vertical_dataset(
        df,
        min_vocab_frequency=1,
        columns=[("feature", Semantic.CATEGORICAL_SET)],
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=4,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.CATEGORICAL_SET,
                dtype=ds_pb.DType.DTYPE_BYTES,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "words": VocabValue(index=1, count=2),
                        "simple": VocabValue(index=2, count=1),
                        "single item": VocabValue(index=3, count=1),
                        "three": VocabValue(index=4, count=1),
                        "two": VocabValue(index=5, count=1),
                    },
                    number_of_unique_values=6,
                ),
                count_nas=1,
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_pd_with_empty_list(self):
    df = pd.DataFrame({
        "feature": [
            [],
            ["single item"],
            ["two", "words"],
            ["three", "simple", "words", "words"],
        ]
    })
    ds = dataset_lib.create_vertical_dataset(
        df,
        min_vocab_frequency=1,
        columns=[("feature", Semantic.CATEGORICAL_SET)],
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=4,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.CATEGORICAL_SET,
                dtype=ds_pb.DType.DTYPE_BYTES,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "words": VocabValue(index=1, count=2),
                        "simple": VocabValue(index=2, count=1),
                        "single item": VocabValue(index=3, count=1),
                        "three": VocabValue(index=4, count=1),
                        "two": VocabValue(index=5, count=1),
                    },
                    number_of_unique_values=6,
                ),
                count_nas=0,
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_pd_with_unicode_feature_name(self):
    df = pd.DataFrame({"\u0080": [1, 2, 3]})
    ds = dataset_lib.create_vertical_dataset(
        df,
        min_vocab_frequency=1,
        columns=[("\u0080", Semantic.NUMERICAL)],
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="\u0080",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=2,
                    standard_deviation=0.8164965809277263,  # ~math.sqrt(2 / 3)
                    min_value=1,
                    max_value=3,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_pd_with_unicode_values(self):
    df = pd.DataFrame({"feature": [["\u0080"]]})
    ds = dataset_lib.create_vertical_dataset(
        df,
        min_vocab_frequency=1,
        columns=[("feature", Semantic.CATEGORICAL_SET)],
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=1,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.CATEGORICAL_SET,
                dtype=ds_pb.DType.DTYPE_BYTES,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "\u0080": VocabValue(index=1, count=1),
                    },
                    number_of_unique_values=2,
                ),
                count_nas=0,
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_pd_type_inference_lists(self):
    df = pd.DataFrame(
        {
            "feature": [
                ["single item"],
                ["two", "words"],
            ]
        }
    )
    ds = dataset_lib.create_vertical_dataset(
        df,
        min_vocab_frequency=1,
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.CATEGORICAL_SET,
                dtype=ds_pb.DType.DTYPE_BYTES,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "single item": VocabValue(index=1, count=1),
                        "two": VocabValue(index=2, count=1),
                        "words": VocabValue(index=3, count=1),
                    },
                    number_of_unique_values=4,
                ),
                count_nas=0,
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_pd_type_inference_nparrays(self):
    df = pd.DataFrame(
        {
            "feature": [
                np.array(["single item"]),
                np.array(["two", "words"]),
            ]
        }
    )
    ds = dataset_lib.create_vertical_dataset(
        df,
        min_vocab_frequency=1,
    )
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="feature",
                type=ds_pb.ColumnType.CATEGORICAL_SET,
                dtype=ds_pb.DType.DTYPE_BYTES,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOD>": VocabValue(index=0, count=0),
                        "single item": VocabValue(index=1, count=1),
                        "two": VocabValue(index=2, count=1),
                        "words": VocabValue(index=3, count=1),
                    },
                    number_of_unique_values=4,
                ),
                count_nas=0,
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)


@parameterized.parameters(
    ds_pb.ColumnType.NUMERICAL, ds_pb.ColumnType.CATEGORICAL
)
class MissingColumnsTest(parameterized.TestCase):

  def create_data_spec(self, column_type: ds_pb.ColumnType):
    if column_type == ds_pb.ColumnType.NUMERICAL:
      return ds_pb.DataSpecification(
          created_num_rows=3,
          columns=(
              ds_pb.Column(name="f1", type=ds_pb.ColumnType.NUMERICAL),
              ds_pb.Column(name="f2", type=ds_pb.ColumnType.NUMERICAL),
          ),
      )
    if column_type == ds_pb.ColumnType.CATEGORICAL:
      return ds_pb.DataSpecification(
          created_num_rows=3,
          columns=(
              ds_pb.Column(
                  name="f1",
                  type=ds_pb.ColumnType.CATEGORICAL,
                  dtype=ds_pb.DType.DTYPE_BYTES,
                  categorical=ds_pb.CategoricalSpec(
                      items={
                          "<OOD>": VocabValue(index=0, count=0),
                          "1": VocabValue(index=1, count=1),
                          "2": VocabValue(index=2, count=1),
                          "3": VocabValue(index=3, count=1),
                      },
                      number_of_unique_values=4,
                      min_value_count=1,
                  ),
              ),
              ds_pb.Column(
                  name="f2",
                  type=ds_pb.ColumnType.CATEGORICAL,
                  dtype=ds_pb.DType.DTYPE_BYTES,
                  categorical=ds_pb.CategoricalSpec(
                      items={
                          "<OOD>": VocabValue(index=0, count=0),
                          "1": VocabValue(index=1, count=1),
                          "2": VocabValue(index=2, count=1),
                          "3": VocabValue(index=3, count=1),
                      },
                      number_of_unique_values=4,
                      min_value_count=1,
                  ),
              ),
          ),
      )
    raise NotImplementedError(f"Errors for type {column_type} not implemented")

  def get_inferred_dataspec_pd_f1only(self, column_type: ds_pb.ColumnType):
    if column_type == ds_pb.ColumnType.NUMERICAL:
      return ds_pb.DataSpecification(
          created_num_rows=3,
          columns=(
              ds_pb.Column(
                  name="f1",
                  type=ds_pb.ColumnType.NUMERICAL,
                  dtype=ds_pb.DType.DTYPE_INT64,
                  numerical=ds_pb.NumericalSpec(
                      mean=2.0,
                      min_value=1.0,
                      max_value=3.0,
                      standard_deviation=0.8164965809277263,
                  ),
                  count_nas=0,
              ),
          ),
      )
    if column_type == ds_pb.ColumnType.CATEGORICAL:
      return ds_pb.DataSpecification(
          created_num_rows=3,
          columns=(
              ds_pb.Column(
                  name="f1",
                  type=ds_pb.ColumnType.CATEGORICAL,
                  dtype=ds_pb.DType.DTYPE_INT64,
                  categorical=ds_pb.CategoricalSpec(
                      items={
                          "<OOD>": VocabValue(index=0, count=0),
                          "1": VocabValue(index=1, count=1),
                          "2": VocabValue(index=2, count=1),
                          "3": VocabValue(index=3, count=1),
                      },
                      number_of_unique_values=4,
                  ),
                  count_nas=0,
              ),
          ),
      )
    raise NotImplementedError(f"Errors for type {column_type} not implemented")

  def get_inferred_dataspec_file_f1only(self, column_type: ds_pb.ColumnType):
    if column_type == ds_pb.ColumnType.NUMERICAL:
      return ds_pb.DataSpecification(
          created_num_rows=3,
          columns=(
              ds_pb.Column(
                  name="f1",
                  is_manual_type=True,
                  type=ds_pb.ColumnType.NUMERICAL,
                  numerical=ds_pb.NumericalSpec(
                      mean=2.0,
                      min_value=1.0,
                      max_value=3.0,
                      standard_deviation=0.8164965809277263,
                  ),
              ),
          ),
      )
    if column_type == ds_pb.ColumnType.CATEGORICAL:
      return ds_pb.DataSpecification(
          created_num_rows=3,
          columns=(
              ds_pb.Column(
                  name="f1",
                  is_manual_type=True,
                  type=ds_pb.ColumnType.CATEGORICAL,
                  categorical=ds_pb.CategoricalSpec(
                      items={
                          "<OOD>": VocabValue(index=0, count=0),
                          "1": VocabValue(index=3, count=1),
                          "2": VocabValue(index=2, count=1),
                          "3": VocabValue(index=1, count=1),
                      },
                      number_of_unique_values=4,
                      most_frequent_value=1,
                      min_value_count=1,
                      is_already_integerized=False,
                      max_number_of_unique_values=2000,
                  ),
              ),
          ),
      )
    raise NotImplementedError(f"Errors for type {column_type} not implemented")

  def create_csv(self) -> str:
    tmp_dir = self.create_tempdir()
    csv_file = self.create_tempfile(
        content="""f1
1
2
3""",
        file_path=os.path.join(tmp_dir.full_path, "file.csv"),
    )
    return csv_file.full_path

  def get_debug_string(self, column_type: ds_pb.ColumnType):
    if column_type == ds_pb.ColumnType.NUMERICAL:
      return "f1,f2\n1,nan\n2,nan\n3,nan\n"
    if column_type == ds_pb.ColumnType.CATEGORICAL:
      return "f1,f2\n1,NA\n2,NA\n3,NA\n"

    raise NotImplementedError(f"Errors for type {column_type} not implemented")

  def test_required_columns_pd_data_spec_none(
      self, column_type: ds_pb.ColumnType
  ):
    data_spec = self.create_data_spec(column_type)
    df = pd.DataFrame({"f1": [1, 2, 3]})
    with self.assertRaises(ValueError):
      _ = dataset_lib.create_vertical_dataset(df, data_spec=data_spec)

  def test_required_columns_pd_data_spec_empty(
      self, column_type: ds_pb.ColumnType
  ):
    data_spec = self.create_data_spec(column_type)
    df = pd.DataFrame({"f1": [1, 2, 3]})
    ds = dataset_lib.create_vertical_dataset(
        df, data_spec=data_spec, required_columns=[]
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), data_spec)
    self.assertEqual(
        ds._dataset.DebugString(), self.get_debug_string(column_type)
    )

  def test_required_columns_pd_data_spec_explicit_success(
      self, column_type: ds_pb.ColumnType
  ):
    data_spec = self.create_data_spec(column_type)
    df = pd.DataFrame({"f1": [1, 2, 3]})
    ds = dataset_lib.create_vertical_dataset(
        df, data_spec=data_spec, required_columns=["f1"]
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), data_spec)
    self.assertEqual(
        ds._dataset.DebugString(), self.get_debug_string(column_type)
    )

  def test_required_columns_pd_data_spec_explicit_failure(
      self, column_type: ds_pb.ColumnType
  ):
    data_spec = self.create_data_spec(column_type)
    df = pd.DataFrame({"f1": [1, 2, 3]})
    with self.assertRaises(ValueError):
      _ = dataset_lib.create_vertical_dataset(
          df, data_spec=data_spec, required_columns=["f2"]
      )

  def test_required_columns_pd_inference_args_none(
      self, column_type: ds_pb.ColumnType
  ):
    df = pd.DataFrame({"f1": [1, 2, 3]})
    column_semantic = Semantic.from_proto_type(column_type)
    with self.assertRaises(ValueError):
      _ = dataset_lib.create_vertical_dataset(
          df, columns=[("f1", column_semantic), ("f2", column_semantic)]
      )

  def test_required_columns_pd_inference_args_empty(
      self, column_type: ds_pb.ColumnType
  ):
    df = pd.DataFrame({"f1": [1, 2, 3]})
    column_semantic = Semantic.from_proto_type(column_type)
    ds = dataset_lib.create_vertical_dataset(
        df,
        columns=[("f1", column_semantic), ("f2", column_semantic)],
        required_columns=[],
        min_vocab_frequency=1,
    )
    # Note that the dataspec does not contain column f2 since it was not found
    # in the data.
    test_utils.assertProto2Equal(
        self,
        ds._dataset.data_spec(),
        self.get_inferred_dataspec_pd_f1only(column_type),
    )
    self.assertEqual(ds._dataset.DebugString(), "f1\n1\n2\n3\n")

  def test_required_columns_pd_inference_args_explicit_failure(
      self, column_type: ds_pb.ColumnType
  ):
    df = pd.DataFrame({"f1": [1, 2, 3]})
    column_semantic = Semantic.from_proto_type(column_type)
    with self.assertRaises(ValueError):
      _ = dataset_lib.create_vertical_dataset(
          df, columns=[("f2", column_semantic)]
      )

  def test_required_columns_pd_inference_args_explicit_success(
      self, column_type: ds_pb.ColumnType
  ):
    df = pd.DataFrame({"f1": [1, 2, 3]})
    column_semantic = Semantic.from_proto_type(column_type)
    ds = dataset_lib.create_vertical_dataset(
        df,
        columns=[("f1", column_semantic), ("f2", column_semantic)],
        min_vocab_frequency=1,
        required_columns=["f1"],
    )
    # Note that the dataspec does not contain column f2 since it was not found
    # in the data.
    test_utils.assertProto2Equal(
        self,
        ds._dataset.data_spec(),
        self.get_inferred_dataspec_pd_f1only(column_type),
    )
    self.assertEqual(ds._dataset.DebugString(), "f1\n1\n2\n3\n")

  def test_required_columns_file_data_spec_none(
      self, column_type: ds_pb.ColumnType
  ):
    data_spec = self.create_data_spec(column_type)
    file_path = self.create_csv()
    with self.assertRaises(ValueError):
      _ = dataset_lib.create_vertical_dataset(file_path, data_spec=data_spec)

  def test_required_columns_file_data_spec_empty(
      self, column_type: ds_pb.ColumnType
  ):
    data_spec = self.create_data_spec(column_type)
    file_path = self.create_csv()
    ds = dataset_lib.create_vertical_dataset(
        file_path, data_spec=data_spec, required_columns=[]
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), data_spec)
    self.assertEqual(
        ds._dataset.DebugString(), self.get_debug_string(column_type)
    )

  def test_required_columns_file_data_spec_explicit_success(
      self, column_type: ds_pb.ColumnType
  ):
    data_spec = self.create_data_spec(column_type)
    file_path = self.create_csv()
    ds = dataset_lib.create_vertical_dataset(
        file_path, data_spec=data_spec, required_columns=["f1"]
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), data_spec)
    self.assertEqual(
        ds._dataset.DebugString(), self.get_debug_string(column_type)
    )

  def test_required_columns_file_data_spec_explicit_failure(
      self, column_type: ds_pb.ColumnType
  ):
    data_spec = self.create_data_spec(column_type)
    file_path = self.create_csv()
    with self.assertRaises(ValueError):
      _ = dataset_lib.create_vertical_dataset(
          file_path, data_spec=data_spec, required_columns=["f2"]
      )

  def test_required_columns_file_inference_args_none(
      self, column_type: ds_pb.ColumnType
  ):
    file_path = self.create_csv()
    column_semantic = Semantic.from_proto_type(column_type)
    with self.assertRaises(ValueError):
      _ = dataset_lib.create_vertical_dataset(
          file_path, columns=[("f1", column_semantic), ("f2", column_semantic)]
      )

  def test_required_columns_file_inference_args_empty(
      self, column_type: ds_pb.ColumnType
  ):
    file_path = self.create_csv()
    column_semantic = Semantic.from_proto_type(column_type)
    ds = dataset_lib.create_vertical_dataset(
        file_path,
        columns=[("f1", column_semantic), ("f2", column_semantic)],
        required_columns=[],
        min_vocab_frequency=1,
    )
    # Note that the dataspec does not contain column f2 since it was not found
    # in the data.
    test_utils.assertProto2Equal(
        self,
        ds._dataset.data_spec(),
        self.get_inferred_dataspec_file_f1only(column_type),
    )
    self.assertEqual(ds._dataset.DebugString(), "f1\n1\n2\n3\n")

  def test_required_columns_file_inference_args_explicit_failure(
      self, column_type: ds_pb.ColumnType
  ):
    file_path = self.create_csv()
    column_semantic = Semantic.from_proto_type(column_type)
    with self.assertRaises(ValueError):
      _ = dataset_lib.create_vertical_dataset(
          file_path, columns=[("f2", column_semantic)]
      )

  def test_required_columns_file_inference_args_explicit_success(
      self, column_type: ds_pb.ColumnType
  ):
    file_path = self.create_csv()
    column_semantic = Semantic.from_proto_type(column_type)
    ds = dataset_lib.create_vertical_dataset(
        file_path,
        columns=[("f1", column_semantic), ("f2", column_semantic)],
        required_columns=["f1"],
        min_vocab_frequency=1,
    )
    # Note that the dataspec does not contain column f2 since it was not found
    # in the data.
    test_utils.assertProto2Equal(
        self,
        ds._dataset.data_spec(),
        self.get_inferred_dataspec_file_f1only(column_type),
    )
    self.assertEqual(ds._dataset.DebugString(), "f1\n1\n2\n3\n")


class DenseDictionaryTest(parameterized.TestCase):

  @parameterized.parameters(
      ([0], 1),
      ([0, 1], 2),
      ([0, 1, 1, 0], 2),
      ([1, 0], 2),
      ([4, 3, 4, 1, 2, 0, 1, 2, 4], 5),
  )
  def test_dense_integer_dictionary_size(self, values, expected):
    self.assertEqual(
        dataset_lib.dense_integer_dictionary_size(np.array(values)), expected
    )

  @parameterized.parameters(
      ([],),
      ([-1, 0, 1],),
      ([1, 2, 3, 4],),
      ([0, 1, 3, 4],),
  )
  def test_dense_integer_dictionary_size_is_none(self, values):
    self.assertIsNone(
        dataset_lib.dense_integer_dictionary_size(np.array(values))
    )


class DiscretizedNumericalTest(parameterized.TestCase):

  class DataFormat(enum.Enum):
    CSV = "csv"
    IN_MEMORY = "in_memory"

  def create_inmemory_dataset(self):
    return {
        "f1": np.array([100 + i for i in range(1, 11)]),
        "f2": np.array([100 + 2 * i for i in range(10, 0, -1)]),
    }

  def create_csv(self) -> str:
    tmp_dir = self.create_tempdir()
    ds = self.create_inmemory_dataset()
    csv_file = self.create_tempfile(
        content=pd.DataFrame(ds).to_csv(index=False),
        file_path=os.path.join(tmp_dir.full_path, "file.csv"),
    )
    return csv_file.full_path

  def create_data(self, data_format: DataFormat):
    if data_format == DiscretizedNumericalTest.DataFormat.CSV:
      return self.create_csv()
    elif data_format == DiscretizedNumericalTest.DataFormat.IN_MEMORY:
      return self.create_inmemory_dataset()
    else:
      raise ValueError(f"Unknown data format ${data_format}")

  def get_spec_options(
      self, data_format: DataFormat, is_manual_type: Optional[bool]
  ):
    if data_format == self.DataFormat.CSV:
      dtype = None  # Never set for CSV data.
      count_nas = None  # Never set for CSV data.
      if is_manual_type is None:
        is_manual_type = False  # Always set for CSV data.
    elif data_format == self.DataFormat.IN_MEMORY:
      dtype = ds_pb.DType.DTYPE_INT64  # Always set for in-memory data.
      count_nas = 0  # Always set for in-memory data.
      is_manual_type = None  # Never set for in-memory data.
    else:
      raise ValueError("Not reached")
    return dtype, count_nas, is_manual_type

  def col_spec_f1_discretized(
      self, data_format: DataFormat, is_manual_type: Optional[bool] = None
  ):
    dtype, count_nas, is_manual_type = self.get_spec_options(
        data_format, is_manual_type
    )
    col_spec = ds_pb.Column(
        name="f1",
        type=ds_pb.ColumnType.DISCRETIZED_NUMERICAL,
        is_manual_type=is_manual_type,
        numerical=ds_pb.NumericalSpec(
            mean=105.5,
            min_value=101.0,
            max_value=110.0,
            standard_deviation=2.8722813232690143,
        ),
        discretized_numerical=ds_pb.DiscretizedNumericalSpec(
            boundaries=[
                1.401298464324817e-45,
                103.5,
                105.49999237060547,
                105.50000762939453,
                106.5,
                109.5,
            ],
            original_num_unique_values=10,
            maximum_num_bins=255,
            min_obs_in_bins=3,
        ),
        dtype=dtype,
        count_nas=count_nas,
    )
    return col_spec

  def col_spec_f2_discretized(
      self, data_format: DataFormat, is_manual_type: Optional[bool] = None
  ):
    dtype, count_nas, is_manual_type = self.get_spec_options(
        data_format, is_manual_type
    )
    col_spec = ds_pb.Column(
        name="f2",
        type=ds_pb.ColumnType.DISCRETIZED_NUMERICAL,
        is_manual_type=is_manual_type,
        numerical=ds_pb.NumericalSpec(
            mean=111.0,
            min_value=102.0,
            max_value=120.0,
            standard_deviation=5.744562646538029,
        ),
        discretized_numerical=ds_pb.DiscretizedNumericalSpec(
            boundaries=[
                1.401298464324817e-45,
                107.0,
                110.99999237060547,
                111.00000762939453,
                113.0,
                119.0,
            ],
            original_num_unique_values=10,
            maximum_num_bins=255,
            min_obs_in_bins=3,
        ),
        dtype=dtype,
        count_nas=count_nas,
    )
    return col_spec

  def col_spec_f1_numerical(
      self, data_format: DataFormat, is_manual_type: Optional[bool] = None
  ):
    dtype, count_nas, is_manual_type = self.get_spec_options(
        data_format, is_manual_type
    )
    col_spec = ds_pb.Column(
        name="f1",
        type=ds_pb.ColumnType.NUMERICAL,
        is_manual_type=is_manual_type,
        numerical=ds_pb.NumericalSpec(
            mean=105.5,
            min_value=101.0,
            max_value=110.0,
            standard_deviation=2.8722813232690143,
        ),
        dtype=dtype,
        count_nas=count_nas,
    )
    return col_spec

  def col_spec_f2_numerical(
      self, data_format: DataFormat, is_manual_type: Optional[bool] = None
  ):
    dtype, count_nas, is_manual_type = self.get_spec_options(
        data_format, is_manual_type
    )
    col_spec = ds_pb.Column(
        name="f2",
        type=ds_pb.ColumnType.NUMERICAL,
        is_manual_type=is_manual_type,
        numerical=ds_pb.NumericalSpec(
            mean=111.0,
            min_value=102.0,
            max_value=120.0,
            standard_deviation=5.744562646538029,
        ),
        dtype=dtype,
        count_nas=count_nas,
    )
    return col_spec

  @parameterized.parameters(DataFormat.CSV, DataFormat.IN_MEMORY)
  def test_contents_inferred_dataspec(self, data_format):
    data = self.create_data(data_format)
    ds = dataset_lib.create_vertical_dataset(
        data,
        discretize_numerical_columns=True,
    )
    self.assertEqual(
        ds._dataset.DebugString(),
        """f1,f2
51.75,120
51.75,116
51.75,116
104.5,116
104.5,112
106,109
108,109
108,53.5
108,53.5
110.5,53.5
""",
    )

  @parameterized.parameters(DataFormat.CSV, DataFormat.IN_MEMORY)
  def test_contents_explicit_dataspec(self, data_format):
    data = self.create_data(data_format)
    data_spec = ds_pb.DataSpecification(
        columns=[
            self.col_spec_f1_discretized(data_format),
            self.col_spec_f2_discretized(data_format),
        ],
        created_num_rows=10,
    )
    ds = dataset_lib.create_vertical_dataset(data, data_spec=data_spec)
    self.assertEqual(
        ds._dataset.DebugString(),
        """f1,f2
51.75,120
51.75,116
51.75,116
104.5,116
104.5,112
106,109
108,109
108,53.5
108,53.5
110.5,53.5
""",
    )

  @parameterized.parameters(DataFormat.CSV, DataFormat.IN_MEMORY)
  def test_global_parameter_only(self, data_format):
    data = self.create_data(data_format)
    ds = dataset_lib.create_vertical_dataset(
        data,
        discretize_numerical_columns=True,
    )
    test_utils.assertProto2Equal(
        self,
        ds.data_spec(),
        ds_pb.DataSpecification(
            columns=[
                self.col_spec_f1_discretized(data_format),
                self.col_spec_f2_discretized(data_format),
            ],
            created_num_rows=10,
        ),
    )

  @parameterized.parameters(DataFormat.CSV, DataFormat.IN_MEMORY)
  def test_column_def(self, data_format):
    data = self.create_data(data_format)
    ds = dataset_lib.create_vertical_dataset(
        data,
        columns=[("f1", Semantic.DISCRETIZED_NUMERICAL)],
        include_all_columns=True,
    )
    test_utils.assertProto2Equal(
        self,
        ds.data_spec(),
        ds_pb.DataSpecification(
            columns=[
                self.col_spec_f1_discretized(data_format, is_manual_type=True),
                self.col_spec_f2_numerical(data_format),
            ],
            created_num_rows=10,
        ),
    )

  @parameterized.parameters(DataFormat.CSV, DataFormat.IN_MEMORY)
  def test_column_def_and_global(self, data_format):
    data = self.create_data(data_format)
    ds = dataset_lib.create_vertical_dataset(
        data,
        columns=[("f1", Semantic.NUMERICAL)],
        discretize_numerical_columns=True,
        include_all_columns=True,
    )
    test_utils.assertProto2Equal(
        self,
        ds.data_spec(),
        ds_pb.DataSpecification(
            columns=[
                self.col_spec_f1_numerical(data_format, is_manual_type=True),
                self.col_spec_f2_discretized(data_format),
            ],
            created_num_rows=10,
        ),
    )

  @parameterized.parameters(DataFormat.CSV, DataFormat.IN_MEMORY)
  def test_num_bins_global(self, data_format):
    data = self.create_data(data_format)
    ds = dataset_lib.create_vertical_dataset(
        data,
        discretize_numerical_columns=True,
        num_discretized_numerical_bins=4,
    )
    dtype, count_nas, is_manual_type = self.get_spec_options(data_format, None)
    test_utils.assertProto2Equal(
        self,
        ds.data_spec(),
        ds_pb.DataSpecification(
            columns=[
                ds_pb.Column(
                    name="f1",
                    type=ds_pb.ColumnType.DISCRETIZED_NUMERICAL,
                    is_manual_type=is_manual_type,
                    numerical=ds_pb.NumericalSpec(
                        mean=105.5,
                        min_value=101.0,
                        max_value=110.0,
                        standard_deviation=2.8722813232690143,
                    ),
                    discretized_numerical=ds_pb.DiscretizedNumericalSpec(
                        boundaries=[
                            -1.401298464324817e-45,
                            1.401298464324817e-45,
                            105.49999237060547,
                        ],
                        original_num_unique_values=10,
                        maximum_num_bins=4,
                        min_obs_in_bins=3,
                    ),
                    count_nas=count_nas,
                    dtype=dtype,
                ),
                ds_pb.Column(
                    name="f2",
                    type=ds_pb.ColumnType.DISCRETIZED_NUMERICAL,
                    is_manual_type=is_manual_type,
                    numerical=ds_pb.NumericalSpec(
                        mean=111.0,
                        min_value=102.0,
                        max_value=120.0,
                        standard_deviation=5.744562646538029,
                    ),
                    discretized_numerical=ds_pb.DiscretizedNumericalSpec(
                        boundaries=[
                            -1.401298464324817e-45,
                            1.401298464324817e-45,
                            110.99999237060547,
                        ],
                        original_num_unique_values=10,
                        maximum_num_bins=4,
                        min_obs_in_bins=3,
                    ),
                    count_nas=count_nas,
                    dtype=dtype,
                ),
            ],
            created_num_rows=10,
        ),
    )

  @parameterized.parameters(DataFormat.CSV, DataFormat.IN_MEMORY)
  def test_num_bins_feature_def(self, data_format):
    data = self.create_data(data_format)
    ds = dataset_lib.create_vertical_dataset(
        data,
        columns=[
            Column(
                name="f1",
                semantic=Semantic.DISCRETIZED_NUMERICAL,
                num_discretized_numerical_bins=4,
            )
        ],
    )
    dtype, count_nas, is_manual_type = self.get_spec_options(data_format, True)
    test_utils.assertProto2Equal(
        self,
        ds.data_spec(),
        ds_pb.DataSpecification(
            columns=[
                ds_pb.Column(
                    name="f1",
                    type=ds_pb.ColumnType.DISCRETIZED_NUMERICAL,
                    is_manual_type=is_manual_type,
                    numerical=ds_pb.NumericalSpec(
                        mean=105.5,
                        min_value=101.0,
                        max_value=110.0,
                        standard_deviation=2.8722813232690143,
                    ),
                    discretized_numerical=ds_pb.DiscretizedNumericalSpec(
                        boundaries=[
                            -1.401298464324817e-45,
                            1.401298464324817e-45,
                            105.49999237060547,
                        ],
                        original_num_unique_values=10,
                        maximum_num_bins=4,
                        min_obs_in_bins=3,
                    ),
                    count_nas=count_nas,
                    dtype=dtype,
                ),
            ],
            created_num_rows=10,
        ),
    )

  @parameterized.parameters(DataFormat.CSV, DataFormat.IN_MEMORY)
  def test_respects_data_spec(self, data_format: DataFormat):
    data_spec = ds_pb.DataSpecification(
        columns=[
            ds_pb.Column(
                name="f1",
                type=ds_pb.ColumnType.DISCRETIZED_NUMERICAL,
                numerical=ds_pb.NumericalSpec(
                    mean=105.5,
                    min_value=101.0,
                    max_value=110.0,
                    standard_deviation=2.8722813232690143,
                ),
                discretized_numerical=ds_pb.DiscretizedNumericalSpec(
                    boundaries=[
                        -1.401298464324817e-45,
                        1.401298464324817e-45,
                        105.49999237060547,
                    ],
                    original_num_unique_values=10,
                    maximum_num_bins=4,
                    min_obs_in_bins=3,
                ),
            ),
            self.col_spec_f2_discretized(data_format),
        ],
        created_num_rows=10,
    )
    data = self.create_data(data_format)
    ds = dataset_lib.create_vertical_dataset(data, data_spec=data_spec)
    self.assertEqual(
        ds._dataset.DebugString(),
        """f1,f2
52.75,120
52.75,116
52.75,116
52.75,116
52.75,112
106.5,109
106.5,109
106.5,53.5
106.5,53.5
106.5,53.5
""",
    )


class DataspecInferenceFromGeneratorTest(parameterized.TestCase):

  def test_infer_dataspec_integer_categorical(self):
    # Note: For column "f",  value "3" is too infrequent (min_vocab_frequency
    # constraint), and value "4" is pruned by value "1" and "2
    # (max_vocab_count).
    ds = {
        "l": np.array([0, 1, 1, 1, 2, 2, 2, 2, 2]),
        "f": np.array([1, 1, 1, 2, 2, 2, 3, 4, 4]),
    }
    ds_generator = dataset_io_lib.build_batched_example_generator(ds)
    dataspec = dataset_lib.infer_dataspec(
        ds_generator,
        dataspec_lib.DataSpecInferenceArgs(
            columns=[
                dataspec_lib.Column(
                    "l",
                    dataspec_lib.Semantic.CATEGORICAL,
                    min_vocab_frequency=1,
                ),
                dataspec_lib.Column("f", dataspec_lib.Semantic.CATEGORICAL),
            ],
            include_all_columns=True,
            max_vocab_count=2,
            min_vocab_frequency=2,
            discretize_numerical_columns=True,
            num_discretized_numerical_bins=10,
            max_num_scanned_rows_to_infer_semantic=-1,
            max_num_scanned_rows_to_compute_statistics=10_000,
        ),
    )

    expected_dataspec = ds_pb.DataSpecification(
        columns=(
            ds_pb.Column(
                name="l",
                type=ds_pb.ColumnType.CATEGORICAL,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=3,
                    items={
                        "<OOD>": VocabValue(index=0, count=1),
                        "2": VocabValue(index=1, count=5),
                        "1": VocabValue(index=2, count=3),
                    },
                ),
                count_nas=0,
                dtype=ds_pb.DType.DTYPE_INT64,
            ),
            ds_pb.Column(
                name="f",
                type=ds_pb.ColumnType.CATEGORICAL,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=3,
                    items={
                        "<OOD>": VocabValue(index=0, count=3),
                        "2": VocabValue(index=2, count=3),
                        "1": VocabValue(index=1, count=3),
                    },
                ),
                count_nas=0,
                dtype=ds_pb.DType.DTYPE_INT64,
            ),
        ),
        created_num_rows=9,
    )
    test_utils.assertProto2Equal(self, dataspec, expected_dataspec)

  def test_infer_dataspec_adult(self):
    adult = test_utils.load_datasets("adult")
    ds_generator = dataset_io_lib.build_batched_example_generator(
        adult.train_pd
    )
    dataspec = dataset_lib.infer_dataspec(
        ds_generator,
        dataspec_lib.DataSpecInferenceArgs(
            columns=None,
            include_all_columns=True,
            max_vocab_count=20,
            min_vocab_frequency=5,
            discretize_numerical_columns=True,
            num_discretized_numerical_bins=10,
            max_num_scanned_rows_to_infer_semantic=-1,
            max_num_scanned_rows_to_compute_statistics=100_000,
        ),
    )


class ReservoirSamplingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.cache_size = 1_000

  def test_initial_is_empty(self):
    r = dataset_lib.BatchReservoirSampling(
        np.random.default_rng(), self.cache_size
    )
    self.assertEqual(r._num_in_cache, 0)
    self.assertEqual(r._num_seen, 0)

  def test_add_small_data(self):
    r = dataset_lib.BatchReservoirSampling(
        np.random.default_rng(), self.cache_size
    )
    r.add(np.random.randn(self.cache_size // 4))
    self.assertEqual(r._num_in_cache, self.cache_size // 4)
    self.assertEqual(r._num_seen, self.cache_size // 4)
    self.assertEqual(r._samples.shape, (self.cache_size,))

  def test_add_multiple_small_data(self):
    r = dataset_lib.BatchReservoirSampling(
        np.random.default_rng(), self.cache_size
    )
    r.add(np.random.randn(self.cache_size // 4))
    r.add(np.random.randn(self.cache_size // 4))
    n = 2 * (self.cache_size // 4)
    self.assertEqual(r._num_in_cache, n)
    self.assertEqual(r._num_seen, n)
    self.assertEqual(r._samples.shape, (self.cache_size,))

  def test_add_large_data(self):
    r = dataset_lib.BatchReservoirSampling(
        np.random.default_rng(), self.cache_size
    )
    r.add(np.random.randn(self.cache_size * 2))
    self.assertEqual(r._num_in_cache, self.cache_size)
    self.assertEqual(r._num_seen, self.cache_size * 2)
    self.assertEqual(r._samples.shape, (self.cache_size,))

  def test_add_multiple_chunks(self):
    r = dataset_lib.BatchReservoirSampling(
        np.random.default_rng(), self.cache_size
    )
    n = 2 * (self.cache_size // 3)
    r.add(np.random.randn(n))
    r.add(np.random.randn(n))
    r.add(np.random.randn(n))
    self.assertEqual(r._num_in_cache, self.cache_size)
    self.assertEqual(r._num_seen, 3 * n)
    self.assertEqual(r._samples.shape, (self.cache_size,))

  @parameterized.parameters(
      (10, 5000, 10_000),
      (200, 5000, 10_000),
      (20_000, 10, 10_000),
  )
  def test_approx_quantiles_small(self, batch_size, num_batches, cache_size):
    r = dataset_lib.BatchReservoirSampling(np.random.default_rng(), cache_size)
    values = []
    for i in range(num_batches):
      batch = np.random.randn(batch_size) * i / num_batches
      values.append(batch)
      r.add(batch)
    quantiles, thresholds = r.get_quantiles(10)
    expected_quantiles = np.nanquantile(
        np.concatenate(values, axis=0), thresholds
    )
    # The first and last quantiles  (i.e., min/max) are very noisy.
    quantiles = quantiles[1:-1]
    expected_quantiles = expected_quantiles[1:-1]
    max_diff = np.max(np.absolute(quantiles - expected_quantiles))
    self.assertLessEqual(
        max_diff,
        0.05,
        msg=f"quantiles={quantiles} expected_quantiles={expected_quantiles}",
    )

  def test_cache1(self):
    rng = np.random.default_rng()
    samples = []
    for _ in range(100):
      sampler = dataset_lib.BatchReservoirSampling(rng=rng, cache_size=1)
      sampler.add(np.array([0]))
      sampler.add(np.array([1]))
      samples.append(sampler._samples[0])
    rate = np.mean(samples)
    self.assertAlmostEqual(rate, 0.5, delta=0.12)


class VectorSequenceTest(absltest.TestCase):

  def create_toy_avro_dataset(self) -> str:
    schema = fastavro.parse_schema({
        "fields": [
            {
                "name": "f1",
                "type": "float",
            },
            {
                "name": "f2",
                "type": {
                    "type": "array",
                    "items": {"type": "array", "items": "float"},
                },
            },
        ],
        "name": "",
        "type": "record",
    })
    records = [
        {"f1": 1, "f2": [[1, 2], [3, 4], [5, 6]]},
        {"f1": 2, "f2": [[7, 8], [1, 2]]},
    ]

    tmp_dir = self.create_tempdir()
    file = self.create_tempfile(
        file_path=os.path.join(tmp_dir.full_path, "file.avro")
    )
    with file.open_bytes("wb") as f:
      fastavro.writer(f, schema, records, codec="null")
    return file.full_path

  def test_read_from_avro(self):
    path = self.create_toy_avro_dataset()
    ds = dataset_lib.create_vertical_dataset("avro:" + path)
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="f1",
                type=ds_pb.ColumnType.NUMERICAL,
                is_manual_type=False,
                numerical=ds_pb.NumericalSpec(
                    mean=1.5,
                    min_value=1,
                    max_value=2,
                    standard_deviation=0.5,
                ),
            ),
            ds_pb.Column(
                name="f2",
                type=ds_pb.ColumnType.NUMERICAL_VECTOR_SEQUENCE,
                is_manual_type=False,
                numerical=ds_pb.NumericalSpec(
                    mean=3.9,
                    min_value=1,
                    max_value=8,
                    standard_deviation=2.3853720883753127,
                ),
                numerical_vector_sequence=ds_pb.NumericalVectorSequenceSpec(
                    vector_length=2,
                    count_values=10,
                    min_num_vectors=2,
                    max_num_vectors=3,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_read_from_numpy(self):
    ds = {
        "f1": np.array([1, 2]),
        "f2": [
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[7.0, 8.0], [1.0, 2.0]], dtype=np.float32),
        ],
    }
    vertical_ds = dataset_lib.create_vertical_dataset(ds)
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=2,
        columns=(
            ds_pb.Column(
                name="f1",
                type=ds_pb.ColumnType.NUMERICAL,
                numerical=ds_pb.NumericalSpec(
                    mean=1.5,
                    min_value=1,
                    max_value=2,
                    standard_deviation=0.5,
                ),
                count_nas=0,
                dtype=ds_pb.DType.DTYPE_INT64,
            ),
            ds_pb.Column(
                name="f2",
                type=ds_pb.ColumnType.NUMERICAL_VECTOR_SEQUENCE,
                numerical=ds_pb.NumericalSpec(
                    mean=3.9,
                    min_value=1,
                    max_value=8,
                    standard_deviation=2.3853720883753127,
                ),
                dtype=ds_pb.DType.DTYPE_INT64,
                numerical_vector_sequence=ds_pb.NumericalVectorSequenceSpec(
                    vector_length=2,
                    count_values=10,
                    min_num_vectors=2,
                    max_num_vectors=3,
                ),
            ),
        ),
    )
    test_utils.assertProto2Equal(
        self, vertical_ds.data_spec(), expected_data_spec
    )
    self.assertEqual(
        str(vertical_ds._dataset.DebugString()),
        """\
f1,f2
1,[[1, 2], [3, 4], [5, 6]]
2,[[7, 8], [1, 2]]
""",
    )


if __name__ == "__main__":
  absltest.main()
