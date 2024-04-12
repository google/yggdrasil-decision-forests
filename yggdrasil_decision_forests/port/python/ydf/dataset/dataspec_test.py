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

"""Test dataspec utilities."""

from absl.testing import absltest
from absl.testing import parameterized

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from ydf.dataset import dataspec as dataspec_lib

Semantic = dataspec_lib.Semantic
VocabValue = ds_pb.CategoricalSpec.VocabValue
Column = dataspec_lib.Column
Monotonic = dataspec_lib.Monotonic
DataSpecInferenceArgs = dataspec_lib.DataSpecInferenceArgs


def toy_dataspec():
  return ds_pb.DataSpecification(
      columns=[
          ds_pb.Column(
              name="f0",
              type=ds_pb.ColumnType.NUMERICAL,
          ),
          ds_pb.Column(
              name="f1",
              type=ds_pb.ColumnType.CATEGORICAL,
              categorical=ds_pb.CategoricalSpec(
                  number_of_unique_values=3,
                  items={
                      "<OOD>": ds_pb.CategoricalSpec.VocabValue(index=0),
                      "x": ds_pb.CategoricalSpec.VocabValue(index=1),
                      "y": ds_pb.CategoricalSpec.VocabValue(index=2),
                  },
              ),
          ),
          ds_pb.Column(
              name="f2",
              type=ds_pb.ColumnType.CATEGORICAL,
              categorical=ds_pb.CategoricalSpec(
                  number_of_unique_values=3,
                  is_already_integerized=True,
              ),
          ),
          ds_pb.Column(
              name="f3",
              type=ds_pb.ColumnType.DISCRETIZED_NUMERICAL,
              discretized_numerical=ds_pb.DiscretizedNumericalSpec(
                  boundaries=[0, 1, 2],
              ),
          ),
          ds_pb.Column(
              name="f4_invalid",
              type=ds_pb.ColumnType.CATEGORICAL,
              categorical=ds_pb.CategoricalSpec(
                  number_of_unique_values=3,
                  items={
                      "<OOD>": ds_pb.CategoricalSpec.VocabValue(index=0),
                      "x": ds_pb.CategoricalSpec.VocabValue(index=1),
                      "y": ds_pb.CategoricalSpec.VocabValue(index=1),
                  },
              ),
          ),
          ds_pb.Column(
              name="f5_invalid",
              type=ds_pb.ColumnType.CATEGORICAL,
              categorical=ds_pb.CategoricalSpec(
                  number_of_unique_values=3,
                  items={
                      "<OOD>": ds_pb.CategoricalSpec.VocabValue(index=0),
                      "y": ds_pb.CategoricalSpec.VocabValue(index=2),
                  },
              ),
          ),
      ]
  )


class DataspecTest(absltest.TestCase):

  def test_categorical_column_dictionary_to_list(self):
    dataspec = toy_dataspec()

    self.assertEqual(
        dataspec_lib.categorical_column_dictionary_to_list(dataspec.columns[1]),
        ["<OOD>", "x", "y"],
    )

    self.assertEqual(
        dataspec_lib.categorical_column_dictionary_to_list(dataspec.columns[2]),
        ["0", "1", "2"],
    )

  def test_categorical_column_dictionary_to_list_issues(self):
    dataspec = toy_dataspec()
    with self.assertRaisesRegex(ValueError, "Duplicated index"):
      dataspec_lib.categorical_column_dictionary_to_list(dataspec.columns[4])
    with self.assertRaisesRegex(ValueError, "No value for index"):
      dataspec_lib.categorical_column_dictionary_to_list(dataspec.columns[5])

  def test_column_defs_contains_column(self):
    column_name = "target"
    self.assertFalse(
        dataspec_lib.column_defs_contains_column(column_name, None)
    )
    str_defs_positive = ["foo", "target", "bar", "", "*"]
    self.assertTrue(
        dataspec_lib.column_defs_contains_column(column_name, str_defs_positive)
    )
    str_defs_negative = ["foo", "tar", "bar", "", "*"]
    self.assertFalse(
        dataspec_lib.column_defs_contains_column(column_name, str_defs_negative)
    )
    tuple_defs_positive = [
        ("foo", Semantic.NUMERICAL),
        ("target", Semantic.CATEGORICAL),
    ]
    self.assertTrue(
        dataspec_lib.column_defs_contains_column(
            column_name, tuple_defs_positive
        )
    )
    tuple_defs_negative = [
        ("foo", Semantic.NUMERICAL),
        ("tar", Semantic.CATEGORICAL),
    ]
    self.assertFalse(
        dataspec_lib.column_defs_contains_column(
            column_name, tuple_defs_negative
        )
    )
    column_defs_positive = [Column("foo"), Column("target")]
    self.assertTrue(
        dataspec_lib.column_defs_contains_column(
            column_name, column_defs_positive
        )
    )
    column_defs_negative = [Column("foo"), Column("tar")]
    self.assertFalse(
        dataspec_lib.column_defs_contains_column(
            column_name, column_defs_negative
        )
    )

  def test_categorical_column_guide(self):
    self.assertEqual(
        dataspec_lib.categorical_column_guide(
            Column("a", Semantic.CATEGORICAL, max_vocab_count=3),
            DataSpecInferenceArgs(
                columns=[],
                include_all_columns=False,
                max_vocab_count=1,
                min_vocab_frequency=2,
                discretize_numerical_columns=False,
                num_discretized_numerical_bins=1,
                max_num_scanned_rows_to_infer_semantic=10000,
                max_num_scanned_rows_to_compute_statistics=10000,
            ),
        ),
        {"max_vocab_count": 3, "min_vocab_frequency": 2},
    )

  def test_priority(self):
    self.assertEqual(dataspec_lib.priority(1, 2), 1)
    self.assertEqual(dataspec_lib.priority(None, 2), 2)
    self.assertIsNone(dataspec_lib.priority(None, None), None)

  def test_get_all_columns(self):
    self.assertEqual(
        dataspec_lib.get_all_columns(
            ["a", "b", "c", "d"],
            DataSpecInferenceArgs(
                columns=[
                    Column("a"),
                    Column("b", Semantic.NUMERICAL),
                    Column("c", Semantic.CATEGORICAL),
                ],
                include_all_columns=False,
                max_vocab_count=1,
                min_vocab_frequency=1,
                discretize_numerical_columns=False,
                num_discretized_numerical_bins=1,
                max_num_scanned_rows_to_infer_semantic=10000,
                max_num_scanned_rows_to_compute_statistics=10000,
            ),
            required_columns=None,
        )[0],
        [
            Column("a"),
            Column("b", Semantic.NUMERICAL),
            Column("c", Semantic.CATEGORICAL),
        ],
    )

  def test_get_all_columns_include_all_columns(self):
    self.assertEqual(
        dataspec_lib.get_all_columns(
            ["a", "b"],
            DataSpecInferenceArgs(
                columns=[Column("a")],
                include_all_columns=True,
                max_vocab_count=1,
                min_vocab_frequency=1,
                discretize_numerical_columns=False,
                num_discretized_numerical_bins=1,
                max_num_scanned_rows_to_infer_semantic=10000,
                max_num_scanned_rows_to_compute_statistics=10000,
            ),
            required_columns=None,
        )[0],
        [
            Column("a"),
            Column("b"),
        ],
    )

  def test_get_all_columns_missing(self):
    with self.assertRaisesRegex(
        ValueError, "Column 'b' is required but was not found in the data."
    ):
      dataspec_lib.get_all_columns(
          ["a"],
          DataSpecInferenceArgs(
              columns=[Column("b")],
              include_all_columns=True,
              max_vocab_count=1,
              min_vocab_frequency=1,
              discretize_numerical_columns=False,
              num_discretized_numerical_bins=1,
              max_num_scanned_rows_to_infer_semantic=10000,
              max_num_scanned_rows_to_compute_statistics=10000,
          ),
          required_columns=None,
      )

  def test_get_all_columns_required_missing(self):
    with self.assertRaisesRegex(
        ValueError, "One of the required columns was not found in the data."
    ):
      dataspec_lib.get_all_columns(
          ["a"],
          DataSpecInferenceArgs(
              columns=[Column("a")],
              include_all_columns=True,
              max_vocab_count=1,
              min_vocab_frequency=1,
              discretize_numerical_columns=False,
              num_discretized_numerical_bins=1,
              max_num_scanned_rows_to_infer_semantic=10000,
              max_num_scanned_rows_to_compute_statistics=10000,
          ),
          required_columns=["b"],
      )[0]

  def test_get_all_columns_does_not_require_all_specified(self):
    self.assertEqual(
        dataspec_lib.get_all_columns(
            ["a"],
            DataSpecInferenceArgs(
                columns=[Column("b")],
                include_all_columns=True,
                max_vocab_count=1,
                min_vocab_frequency=1,
                discretize_numerical_columns=False,
                num_discretized_numerical_bins=1,
                max_num_scanned_rows_to_infer_semantic=10000,
                max_num_scanned_rows_to_compute_statistics=10000,
            ),
            required_columns=[],
        )[0],
        [
            Column("a"),
        ],
    )

  def test_get_all_columns_specified_and_available_always_included(self):
    self.assertEqual(
        dataspec_lib.get_all_columns(
            ["a"],
            DataSpecInferenceArgs(
                columns=[Column("a")],
                include_all_columns=False,
                max_vocab_count=1,
                min_vocab_frequency=1,
                discretize_numerical_columns=False,
                num_discretized_numerical_bins=1,
                max_num_scanned_rows_to_infer_semantic=10000,
                max_num_scanned_rows_to_compute_statistics=10000,
            ),
            required_columns=[],
        )[0],
        [
            Column("a"),
        ],
    )

  def test_get_all_columns_with_unrolled_features(self):
    columns, unroll_info = dataspec_lib.get_all_columns(
        ["a.0", "a.1", "a.2", "b.0", "b.1", "b.2"],
        DataSpecInferenceArgs(
            columns=[Column("a")],
            include_all_columns=False,
            max_vocab_count=1,
            min_vocab_frequency=1,
            discretize_numerical_columns=False,
            num_discretized_numerical_bins=1,
            max_num_scanned_rows_to_infer_semantic=10000,
            max_num_scanned_rows_to_compute_statistics=10000,
        ),
        required_columns=[],
        unroll_feature_info={
            "a": ["a.0", "a.1", "a.2"],
            "b": ["b.0", "b.1", "b.2"],
        },
    )
    self.assertEqual(
        columns,
        [Column("a.0"), Column("a.1"), Column("a.2")],
    )
    self.assertEqual(unroll_info, {"a": ["a.0", "a.1", "a.2"]})

  def test_normalize_column_defs(self):
    self.assertEqual(
        dataspec_lib.normalize_column_defs([
            "a",
            ("b", Semantic.NUMERICAL),
            Column("c"),
            Column("d", Semantic.CATEGORICAL),
        ]),
        [
            Column("a"),
            Column("b", Semantic.NUMERICAL),
            Column("c"),
            Column("d", Semantic.CATEGORICAL),
        ],
    )

  def test_normalize_column_defs_none(self):
    self.assertIsNone(dataspec_lib.normalize_column_defs(None))


class MonotonicTest(parameterized.TestCase):

  @parameterized.parameters(
      Monotonic.INCREASING,
      Monotonic.DECREASING,
  )
  def test_already_normalized_value(self, value):
    self.assertEqual(Column("f", monotonic=value).normalized_monotonic, value)

  def test_already_normalized_value_none(self):
    self.assertIsNone(Column("f", monotonic=None).normalized_monotonic)

  @parameterized.parameters(
      (+1, Monotonic.INCREASING),
      (-1, Monotonic.DECREASING),
  )
  def test_normalize_value(self, non_normalized_value, normalized_value):
    self.assertEqual(
        Column("f", monotonic=non_normalized_value).normalized_monotonic,
        normalized_value,
    )

  def test_normalize_value_none(self):
    self.assertIsNone(Column("f", monotonic=0).normalized_monotonic)

  def test_good_semantic(self):
    _ = Column("f", monotonic=+1)
    _ = Column("f", semantic=Semantic.NUMERICAL, monotonic=+1)

  def test_bad_semantic(self):
    with self.assertRaisesRegex(
        ValueError, "with monotonic constraint is expected to have"
    ):
      _ = Column("feature", semantic=Semantic.CATEGORICAL, monotonic=+1)


if __name__ == "__main__":
  absltest.main()
