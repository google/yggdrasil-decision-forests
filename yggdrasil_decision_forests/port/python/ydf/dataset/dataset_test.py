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

import os
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from ydf.dataset import dataset
from ydf.dataset import dataspec
from ydf.utils import test_utils

# Make "assertEqual" print more details.
unittest.util._MAX_LENGTH = 10000

Semantic = dataspec.Semantic
VocabValue = ds_pb.CategoricalSpec.VocabValue
Column = dataspec.Column


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
    self.assertEqual(dataset.infer_semantic("", value), expected_semantic)

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
    ds = dataset.create_vertical_dataset(df)
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="col_pos",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=dataspec.np_dtype_to_ydf_dtype(dtype),
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
                dtype=dataspec.np_dtype_to_ydf_dtype(dtype),
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
                dtype=dataspec.np_dtype_to_ydf_dtype(dtype),
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
            "col_single_nan": [1, 2, np.NaN],
            "col_nan_only": [np.NaN, np.NaN, np.NaN],
        },
        dtype=dtype,
    )
    ds = dataset.create_vertical_dataset(df)
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="col_single_nan",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=dataspec.np_dtype_to_ydf_dtype(dtype),
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
                dtype=dataspec.np_dtype_to_ydf_dtype(dtype),
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

    ds = dataset.create_vertical_dataset(
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

    ds = dataset.create_vertical_dataset(
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

    ds = dataset.create_vertical_dataset(df)
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

    ds = dataset.create_vertical_dataset(
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
    ds = dataset.create_vertical_dataset(df, columns=column_definition)
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
    ds = dataset.create_vertical_dataset(ds_dict)
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
        "col1": [1.1, 21.1, np.NaN],
        "col2": [1, 2, 3],
    })
    ds = dataset.create_vertical_dataset(df)
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
    ds = dataset.create_vertical_dataset(df, data_spec=data_spec)
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
    ds = dataset.create_vertical_dataset(ds_dict, data_spec=data_spec)
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
    ds = dataset.create_vertical_dataset(
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
    ds = dataset.create_vertical_dataset(df)
    expected_dataset_content = "col_str\n" + "A\nB\n" * 5
    self.assertEqual(expected_dataset_content, ds._dataset.DebugString())

  def test_max_vocab_count_minus_1(self):
    df = pd.DataFrame({
        "col1": ["A", "B", "C", "D", "D"],
    })
    ds = dataset.create_vertical_dataset(
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
      dataset.create_vertical_dataset(
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
    ds = dataset.create_vertical_dataset(
        {"col": np.array(values)},
        columns=[Column("col", dataspec.Semantic.CATEGORICAL)],
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
    ds = dataset.create_vertical_dataset(
        {"col": np.array([0, 1, 4, 3, 1, 2, 3, 4, 12, 11, 10, 9, 8, 7, 6, 5])},
        columns=[Column("col", dataspec.Semantic.CATEGORICAL)],
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
    ds = dataset.create_vertical_dataset(
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
    ds = dataset.create_vertical_dataset(
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
    ds = dataset.create_vertical_dataset(
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
    ds = dataset.create_vertical_dataset(
        "csv:" + csv_file.full_path, min_vocab_frequency=1
    )
    df_pd = pd.read_csv(csv_file)
    ds_pd = dataset.create_vertical_dataset(df_pd, data_spec=ds.data_spec())
    self.assertEqual(ds._dataset.DebugString(), ds_pd._dataset.DebugString())

  @unittest.skip("Requires building YDF with tensorflow io")
  def test_read_from_sharded_tfe(self):
    sharded_path = "tfrecord+tfe:" + os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "toy.tfe-tfrecord@2"
    )
    ds = dataset.create_vertical_dataset(
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
    ds = dataset.create_vertical_dataset(
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

  def test_multidimensional_input(self):
    ds = dataset.create_vertical_dataset(
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
    ds = dataset.create_vertical_dataset({"feature": np.zeros((3, num_dims))})
    self.assertLen(ds.data_spec().columns, num_dims)
    self.assertEqual(ds.data_spec().columns[0].name, expected_first_feature)

  def test_multi_dimensions_error_too_many_dims(self):
    with self.assertRaisesRegex(
        ValueError, "Input features can only be one or two dimensional"
    ):
      _ = dataset.create_vertical_dataset({"feature": np.zeros((3, 3, 3))})

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

    ds = dataset.create_vertical_dataset(
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

    ds = dataset.create_vertical_dataset({"feature": feature})
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

    ds = dataset.create_vertical_dataset({"feature": feature})
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
    ds = dataset.create_vertical_dataset(
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
        dataset.validate_dataspec(
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
        dataset.validate_dataspec(bad_dataspec, [0])[0],
        "Column 'f' is CATEGORICAL but",
    )

    # Bad column not selected
    self.assertEmpty(dataset.validate_dataspec(bad_dataspec, []))

  @parameterized.parameters("1", "1.", "1.0")
  def test_look_numerical(self, value: str):
    self.assertTrue(dataset.look_numerical(value))

  @parameterized.parameters("", "a", "hello")
  def test_does_not_look_numerical(self, value: str):
    self.assertFalse(dataset.look_numerical(value))

  def test_from_numpy(self):
    with self.assertRaisesRegex(
        ValueError, "YDF does not consume Numpy arrays directly"
    ):
      dataset.create_vertical_dataset(np.array([1, 2, 3]))

  def test_from_column_less_pandas(self):
    with self.assertRaisesRegex(
        ValueError, "The pandas DataFrame must have string column names"
    ):
      dataset.create_vertical_dataset(pd.DataFrame([[1, 2, 3], [4, 5, 6]]))


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
    ds = dataset.create_vertical_dataset(
        "csv:" + path_to_csv, min_vocab_frequency=1
    )
    expected_data_spec = self.toy_csv_dataspec_categorical()
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_csv_tokenization_when_semantic_specified(self):
    path_to_csv = self.create_toy_csv()
    ds = dataset.create_vertical_dataset(
        "csv:" + path_to_csv,
        min_vocab_frequency=1,
        columns=[("col_cat_set", Semantic.CATEGORICAL_SET)],
    )
    expected_data_spec = self.toy_csv_dataspec_catset()
    test_utils.assertProto2Equal(self, ds.data_spec(), expected_data_spec)

  def test_csv_file_reading_respects_data_spec_categorical(self):
    path_to_csv = self.create_toy_csv()
    data_spec = self.toy_csv_dataspec_categorical()
    ds = dataset.create_vertical_dataset(
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
    ds = dataset.create_vertical_dataset(
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

  def test_pd_list_of_list(self):
    df = pd.DataFrame({
        "feature": [
            ["single item"],
            ["two", "words"],
            ["three", "simple", "words", "words"],
            [""],
        ]
    })
    ds = dataset.create_vertical_dataset(
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
        ValueError, "Categorical Set columns must be a list of lists."
    ):
      _ = dataset.create_vertical_dataset(
          df,
          min_vocab_frequency=1,
          columns=[("feature", Semantic.CATEGORICAL_SET)],
      )

  def test_pd_np_bytes(self):
    df = pd.DataFrame({
        "feature": [
            np.array(["single item"], np.bytes_),
            np.array(["two", "words"], np.bytes_),
            np.array(["three", "simple", "words", "words"], np.bytes_),
            np.array([""], np.bytes_),
        ]
    })
    ds = dataset.create_vertical_dataset(
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
    ds = dataset.create_vertical_dataset(
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
    ds = dataset.create_vertical_dataset(
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

  def test_pd_type_inference_lists(self):
    df = pd.DataFrame(
        {
            "feature": [
                ["single item"],
                ["two", "words"],
            ]
        }
    )
    ds = dataset.create_vertical_dataset(
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
    ds = dataset.create_vertical_dataset(
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
      _ = dataset.create_vertical_dataset(df, data_spec=data_spec)

  def test_required_columns_pd_data_spec_empty(
      self, column_type: ds_pb.ColumnType
  ):
    data_spec = self.create_data_spec(column_type)
    df = pd.DataFrame({"f1": [1, 2, 3]})
    ds = dataset.create_vertical_dataset(
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
    ds = dataset.create_vertical_dataset(
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
      _ = dataset.create_vertical_dataset(
          df, data_spec=data_spec, required_columns=["f2"]
      )

  def test_required_columns_pd_inference_args_none(
      self, column_type: ds_pb.ColumnType
  ):
    df = pd.DataFrame({"f1": [1, 2, 3]})
    column_semantic = dataspec.Semantic.from_proto_type(column_type)
    with self.assertRaises(ValueError):
      _ = dataset.create_vertical_dataset(
          df, columns=[("f1", column_semantic), ("f2", column_semantic)]
      )

  def test_required_columns_pd_inference_args_empty(
      self, column_type: ds_pb.ColumnType
  ):
    df = pd.DataFrame({"f1": [1, 2, 3]})
    column_semantic = dataspec.Semantic.from_proto_type(column_type)
    ds = dataset.create_vertical_dataset(
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
    column_semantic = dataspec.Semantic.from_proto_type(column_type)
    with self.assertRaises(ValueError):
      _ = dataset.create_vertical_dataset(df, columns=[("f2", column_semantic)])

  def test_required_columns_pd_inference_args_explicit_success(
      self, column_type: ds_pb.ColumnType
  ):
    df = pd.DataFrame({"f1": [1, 2, 3]})
    column_semantic = dataspec.Semantic.from_proto_type(column_type)
    ds = dataset.create_vertical_dataset(
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
      _ = dataset.create_vertical_dataset(file_path, data_spec=data_spec)

  def test_required_columns_file_data_spec_empty(
      self, column_type: ds_pb.ColumnType
  ):
    data_spec = self.create_data_spec(column_type)
    file_path = self.create_csv()
    ds = dataset.create_vertical_dataset(
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
    ds = dataset.create_vertical_dataset(
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
      _ = dataset.create_vertical_dataset(
          file_path, data_spec=data_spec, required_columns=["f2"]
      )

  def test_required_columns_file_inference_args_none(
      self, column_type: ds_pb.ColumnType
  ):
    file_path = self.create_csv()
    column_semantic = dataspec.Semantic.from_proto_type(column_type)
    with self.assertRaises(ValueError):
      _ = dataset.create_vertical_dataset(
          file_path, columns=[("f1", column_semantic), ("f2", column_semantic)]
      )

  def test_required_columns_file_inference_args_empty(
      self, column_type: ds_pb.ColumnType
  ):
    file_path = self.create_csv()
    column_semantic = dataspec.Semantic.from_proto_type(column_type)
    ds = dataset.create_vertical_dataset(
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
    column_semantic = dataspec.Semantic.from_proto_type(column_type)
    with self.assertRaises(ValueError):
      _ = dataset.create_vertical_dataset(
          file_path, columns=[("f2", column_semantic)]
      )

  def test_required_columns_file_inference_args_explicit_success(
      self, column_type: ds_pb.ColumnType
  ):
    file_path = self.create_csv()
    column_semantic = dataspec.Semantic.from_proto_type(column_type)
    ds = dataset.create_vertical_dataset(
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
        dataset.dense_integer_dictionary_size(np.array(values)), expected
    )

  @parameterized.parameters(
      ([],),
      ([-1, 0, 1],),
      ([1, 2, 3, 4],),
      ([0, 1, 3, 4],),
  )
  def test_dense_integer_dictionary_size_is_none(self, values):
    self.assertIsNone(dataset.dense_integer_dictionary_size(np.array(values)))


if __name__ == "__main__":
  absltest.main()
