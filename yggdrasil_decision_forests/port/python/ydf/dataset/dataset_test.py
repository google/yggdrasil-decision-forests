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


class DatasetTest(parameterized.TestCase):

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
      (np.array(["a"], np.string_), Semantic.CATEGORICAL),
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
    self.assertEqual(ds.data_spec(), expected_data_spec)

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
                count_nas=3,
                numerical=ds_pb.NumericalSpec(),
            ),
        ),
    )
    self.assertEqual(ds.data_spec(), expected_data_spec)

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
    self.assertEqual(ds.data_spec(), expected_data_spec)

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
    self.assertEqual(ds.data_spec(), expected_data_spec)

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
                count_nas=0,
                boolean=ds_pb.BooleanSpec(
                    count_true=1,
                    count_false=2,
                ),
            ),
        ),
    )
    self.assertEqual(ds.data_spec(), expected_data_spec)

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
            ),
        ),
    )
    self.assertEqual(ds.data_spec(), expected_data_spec)

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
    self.assertEqual(ds.data_spec(), expected_data_spec)

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
    self.assertEqual(ds.data_spec(), expected_data_spec)

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
    self.assertEqual(ds.data_spec(), data_spec)

  def test_create_vds_dict_of_values_with_spec(self):
    data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="col_num",
                type=ds_pb.ColumnType.NUMERICAL,
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
    self.assertEqual(ds.data_spec(), data_spec)
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
    self.assertEqual(ds.data_spec(), expected_data_spec)

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
                count_nas=count_nas,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOV>": VocabValue(index=0, count=expected_counts[0]),
                        "false": VocabValue(index=1, count=expected_counts[1]),
                        "true": VocabValue(index=2, count=expected_counts[2]),
                    },
                    number_of_unique_values=3,
                ),
            ),
        ),
    )
    self.assertEqual(ds.data_spec(), expected_data_spec)

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
                count_nas=count_nas,
                categorical=ds_pb.CategoricalSpec(
                    items={
                        "<OOV>": VocabValue(index=0, count=expected_counts[0]),
                        "false": VocabValue(index=1, count=expected_counts[1]),
                        "true": VocabValue(index=2, count=expected_counts[2]),
                    },
                    number_of_unique_values=3,
                ),
            ),
        ),
    )
    self.assertEqual(ds.data_spec(), expected_data_spec)

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
    self.assertEqual(ds.data_spec(), expected_data_spec)

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
                is_manual_type=False,
                boolean=ds_pb.BooleanSpec(count_true=2, count_false=2),
            ),
            ds_pb.Column(
                name="Cat_2",
                type=ds_pb.CATEGORICAL,
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
    self.assertEqual(ds.data_spec(), expected_data_spec)

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
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=2,
                    standard_deviation=2,
                    min_value=0,
                    max_value=4,
                ),
            ),
            ds_pb.Column(
                name="feature.1_of_3",
                type=ds_pb.ColumnType.NUMERICAL,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=3,
                    standard_deviation=2,
                    min_value=1,
                    max_value=5,
                ),
            ),
            ds_pb.Column(
                name="feature.2_of_3",
                type=ds_pb.ColumnType.NUMERICAL,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=4,
                    standard_deviation=2,
                    min_value=2,
                    max_value=6,
                ),
            ),
        ),
    )
    self.assertEqual(ds.data_spec(), expected_data_spec)

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


if __name__ == "__main__":
  absltest.main()
