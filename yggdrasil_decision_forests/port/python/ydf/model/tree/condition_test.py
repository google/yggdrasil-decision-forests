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

from typing import Sequence
from absl.testing import absltest
from absl.testing import parameterized
from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.model.decision_tree import decision_tree_pb2
from ydf.model.tree import condition as condition_lib
from ydf.utils import test_utils


class ConditionTest(parameterized.TestCase):

  def setUp(self):
    self.data_spec_columns = {
        "f_nothing_1": data_spec_pb2.Column(name="f_nothing_1"),
        "f_nothing_2": data_spec_pb2.Column(name="f_nothing_2"),
        "f_numerical_1": data_spec_pb2.Column(
            name="f_numerical_1",
            type=data_spec_pb2.ColumnType.NUMERICAL,
            numerical=data_spec_pb2.NumericalSpec(),
        ),
        "f_numerical_2": data_spec_pb2.Column(
            name="f_numerical_2",
            type=data_spec_pb2.ColumnType.NUMERICAL,
            numerical=data_spec_pb2.NumericalSpec(),
        ),
        "f_discretized": data_spec_pb2.Column(
            name="f_discretized",
            discretized_numerical=data_spec_pb2.DiscretizedNumericalSpec(
                boundaries=[10, 20, 30, 40]
            ),
        ),
        "f_categorical_large_vocab": data_spec_pb2.Column(
            name="f_categorical_large_vocab",
            type=data_spec_pb2.ColumnType.CATEGORICAL,
            categorical=data_spec_pb2.CategoricalSpec(
                number_of_unique_values=1000,
                is_already_integerized=True,
            ),
        ),
        "f_categorical_small_vocab": data_spec_pb2.Column(
            name="f_categorical_small_vocab",
            type=data_spec_pb2.ColumnType.CATEGORICAL,
            categorical=data_spec_pb2.CategoricalSpec(
                number_of_unique_values=4,
                items={
                    "OOD": data_spec_pb2.CategoricalSpec.VocabValue(index=0),
                    "A": data_spec_pb2.CategoricalSpec.VocabValue(index=1),
                    "B": data_spec_pb2.CategoricalSpec.VocabValue(index=2),
                    "D": data_spec_pb2.CategoricalSpec.VocabValue(index=3),
                },
            ),
        ),
        "f_categorical_set_large_vocab": data_spec_pb2.Column(
            name="f_categorical_set_large_vocab",
            type=data_spec_pb2.ColumnType.CATEGORICAL_SET,
            categorical=data_spec_pb2.CategoricalSpec(
                number_of_unique_values=1000,
                is_already_integerized=True,
            ),
        ),
        "f_categorical_set_small_vocab": data_spec_pb2.Column(
            name="f_categorical_set_small_vocab",
            type=data_spec_pb2.ColumnType.CATEGORICAL_SET,
            categorical=data_spec_pb2.CategoricalSpec(
                number_of_unique_values=4,
                is_already_integerized=True,
            ),
        ),
    }
    self.full_data_spec = data_spec_pb2.DataSpecification(
        columns=self.data_spec_columns.values()
    )
    super().setUp()

  @parameterized.named_parameters(
      (
          "two random bytes",
          b"\x2D\x03",
          10,
          [0, 2, 3, 5, 8, 9],
      ),  # b1100101101 => 0x032D
      ("empty", b"", 0, []),
      ("full of 0s", b"\x00\x00", 16, []),
      ("full of 1s", b"\xFF\xFF", 16, list(range(16))),
  )
  def test_bitmap_to_items(
      self,
      bitmap: bytes,
      number_of_unique_values: int,
      expected_items: Sequence[int],
  ):
    column_spec = data_spec_pb2.Column(
        categorical=data_spec_pb2.CategoricalSpec(
            number_of_unique_values=number_of_unique_values
        )
    )

    self.assertEqual(
        condition_lib.bitmap_to_items(column_spec, bitmap),
        expected_items,
    )

  @parameterized.named_parameters(
      (
          "two random bytes",
          [0, 2, 3, 5, 8, 9],
          10,
          b"\x2D\x03",
      ),  # b1100101101 => 0x032D
      ("empty", [], 0, b""),
      ("repeated bits", [3, 4, 3], 10, b"\x18\x00"),  # b00011000 => 0x0018
      ("set high bit in 1 byte", [7], 8, b"\x80"),
      ("set low bit in 1 byte", [0], 8, b"\x01"),
      ("set high bit in 4 bytes", [31], 32, b"\x00\x00\x00\x80"),
      ("set low bit in 4 bytes", [0], 32, b"\x01\x00\x00\x00"),
      ("full of 0s", [], 32, b"\x00\x00\x00\x00"),
      ("full of 1s", range(32), 32, b"\xFF\xFF\xFF\xFF"),
  )
  def test_items_to_bitmap_with_valid_input(
      self,
      items: Sequence[int],
      number_of_unique_values: int,
      expected_bitmap: bytes,
  ):
    column_spec = data_spec_pb2.Column(
        categorical=data_spec_pb2.CategoricalSpec(
            number_of_unique_values=number_of_unique_values
        )
    )

    self.assertEqual(
        condition_lib.items_to_bitmap(column_spec, items),
        expected_bitmap,
    )

  def test_items_to_bitmap_invalid_item_is_negative(self):
    column_spec = data_spec_pb2.Column(
        categorical=data_spec_pb2.CategoricalSpec(number_of_unique_values=10)
    )

    with self.assertRaisesRegex(ValueError, "-1"):
      condition_lib.items_to_bitmap(column_spec, [-1])

  def test_items_to_bitmap_invalid_item_is_too_large(self):
    column_spec = data_spec_pb2.Column(
        categorical=data_spec_pb2.CategoricalSpec(number_of_unique_values=10)
    )

    with self.assertRaisesRegex(ValueError, "10"):
      condition_lib.items_to_bitmap(column_spec, [10])

  def _assert_conditions_equivalent(
      self,
      condition: condition_lib.AbstractCondition,
      proto_condition: decision_tree_pb2.NodeCondition,
      dataspec: data_spec_pb2.DataSpecification,
  ) -> None:
    """Assets that a condition and a proto condition are equivalent."""

    # Condition to proto condition.
    test_utils.assertProto2Equal(
        self,
        condition_lib.to_proto_condition(condition, dataspec),
        proto_condition,
    )

    # Proto condition to condition.
    self.assertEqual(
        condition_lib.to_condition(proto_condition, dataspec), condition
    )

  @parameterized.parameters(0, 1)
  def test_condition_is_missing_with_valid_input(self, attribute_idx):
    condition = condition_lib.IsMissingInCondition(
        attribute=attribute_idx, missing=False, score=2
    )
    dataspec = data_spec_pb2.DataSpecification(
        columns=[data_spec_pb2.Column(name="f_nothing_2")]
    )
    if attribute_idx > 0:
      dataspec = self.full_data_spec
    proto_condition = decision_tree_pb2.NodeCondition(
        na_value=False,
        split_score=2,
        attribute=attribute_idx,
        condition=decision_tree_pb2.Condition(
            na_condition=decision_tree_pb2.Condition.NA(),
        ),
    )

    self._assert_conditions_equivalent(condition, proto_condition, dataspec)

    self.assertEqual(
        condition.pretty(dataspec),
        "'f_nothing_2' is missing [score=2 missing=False]",
    )

  @parameterized.parameters(0, 1)
  def test_condition_is_true_valid_input(self, attribute_idx):
    condition = condition_lib.IsTrueCondition(
        attribute=attribute_idx, missing=False, score=2
    )
    dataspec = data_spec_pb2.DataSpecification(
        columns=[data_spec_pb2.Column(name="f_nothing_2")]
    )
    if attribute_idx > 0:
      dataspec = self.full_data_spec
    proto_condition = decision_tree_pb2.NodeCondition(
        na_value=False,
        split_score=2,
        attribute=attribute_idx,
        condition=decision_tree_pb2.Condition(
            true_value_condition=decision_tree_pb2.Condition.TrueValue(),
        ),
    )

    self._assert_conditions_equivalent(condition, proto_condition, dataspec)

    self.assertEqual(
        condition.pretty(dataspec),
        "'f_nothing_2' is True [score=2 missing=False]",
    )

  @parameterized.parameters(0, 1)
  def test_condition_is_higher_valid_input(self, attribute_idx):
    condition = condition_lib.NumericalHigherThanCondition(
        attribute=attribute_idx, missing=False, score=2, threshold=3
    )
    dataspec = data_spec_pb2.DataSpecification(
        columns=[data_spec_pb2.Column(name="f_nothing_2")]
    )
    if attribute_idx > 0:
      dataspec = self.full_data_spec
    proto_condition = decision_tree_pb2.NodeCondition(
        na_value=False,
        split_score=2,
        attribute=attribute_idx,
        condition=decision_tree_pb2.Condition(
            higher_condition=decision_tree_pb2.Condition.Higher(threshold=3),
        ),
    )

    self._assert_conditions_equivalent(condition, proto_condition, dataspec)

    self.assertEqual(
        condition.pretty(dataspec),
        "'f_nothing_2' >= 3 [score=2 missing=False]",
    )

  @parameterized.parameters(0, 4)
  def test_condition_discretized_is_higher_valid_input(self, attribute_idx):
    condition = condition_lib.DiscretizedNumericalHigherThanCondition(
        attribute=attribute_idx, missing=False, score=2, threshold_idx=2
    )
    dataspec = data_spec_pb2.DataSpecification(
        columns=[
            data_spec_pb2.Column(
                name="f_discretized",
                discretized_numerical=data_spec_pb2.DiscretizedNumericalSpec(
                    boundaries=[10, 20, 30, 40]
                ),
            )
        ]
    )
    if attribute_idx > 0:
      dataspec = self.full_data_spec
    proto_condition = decision_tree_pb2.NodeCondition(
        na_value=False,
        split_score=2,
        attribute=attribute_idx,
        condition=decision_tree_pb2.Condition(
            discretized_higher_condition=decision_tree_pb2.Condition.DiscretizedHigher(
                threshold=2
            ),
        ),
    )

    self._assert_conditions_equivalent(condition, proto_condition, dataspec)

    self.assertEqual(
        condition.pretty(dataspec),
        "'f_discretized' >= 20 [threshold_idx=2 score=2 missing=False]",
    )

  @parameterized.parameters(0, 5)
  def test_condition_is_in_categorical_valid_input(self, attribute_idx):
    condition = condition_lib.CategoricalIsInCondition(
        attribute=attribute_idx,
        missing=False,
        score=2,
        mask=[1, 3],
    )
    dataspec = data_spec_pb2.DataSpecification(
        columns=[
            data_spec_pb2.Column(
                name="f_categorical_large_vocab",
                type=data_spec_pb2.ColumnType.CATEGORICAL,
                categorical=data_spec_pb2.CategoricalSpec(
                    number_of_unique_values=1000,
                    is_already_integerized=True,
                ),
            )
        ]
    )
    if attribute_idx > 0:
      dataspec = self.full_data_spec
    proto_condition = decision_tree_pb2.NodeCondition(
        na_value=False,
        split_score=2,
        attribute=attribute_idx,
        condition=decision_tree_pb2.Condition(
            contains_condition=decision_tree_pb2.Condition.ContainsVector(
                elements=[1, 3]
            ),
        ),
    )

    self._assert_conditions_equivalent(condition, proto_condition, dataspec)

    self.assertEqual(
        condition.pretty(dataspec),
        "'f_categorical_large_vocab' in [1, 3] [score=2 missing=False]",
    )

  @parameterized.parameters(0, 6)
  def test_condition_is_in_categorical_bitmap_valid_input(self, attribute_idx):
    condition = condition_lib.CategoricalIsInCondition(
        attribute=attribute_idx,
        missing=False,
        score=2,
        mask=[1, 3],
    )
    dataspec = data_spec_pb2.DataSpecification(
        columns=[
            data_spec_pb2.Column(
                name="f_categorical_small_vocab",
                type=data_spec_pb2.ColumnType.CATEGORICAL,
                categorical=data_spec_pb2.CategoricalSpec(
                    number_of_unique_values=4,
                    items={
                        "OOD": data_spec_pb2.CategoricalSpec.VocabValue(
                            index=0
                        ),
                        "A": data_spec_pb2.CategoricalSpec.VocabValue(index=1),
                        "B": data_spec_pb2.CategoricalSpec.VocabValue(index=2),
                        "D": data_spec_pb2.CategoricalSpec.VocabValue(index=3),
                    },
                ),
            )
        ]
    )
    if attribute_idx > 0:
      dataspec = self.full_data_spec
    proto_condition = decision_tree_pb2.NodeCondition(
        na_value=False,
        split_score=2,
        attribute=attribute_idx,
        condition=decision_tree_pb2.Condition(
            contains_bitmap_condition=decision_tree_pb2.Condition.ContainsBitmap(
                elements_bitmap=b"\x0A"
            ),
        ),
    )

    self._assert_conditions_equivalent(condition, proto_condition, dataspec)

    self.assertEqual(
        condition.pretty(dataspec),
        "'f_categorical_small_vocab' in ['A', 'D'] [score=2 missing=False]",
    )

  @parameterized.parameters(0, 7)
  def test_condition_is_in_categorical_set_valid_input(self, attribute_idx):
    condition = condition_lib.CategoricalSetContainsCondition(
        attribute=attribute_idx,
        missing=False,
        score=2,
        mask=[1, 3],
    )
    dataspec = data_spec_pb2.DataSpecification(
        columns=[
            data_spec_pb2.Column(
                name="f_categorical_set_large_vocab",
                type=data_spec_pb2.ColumnType.CATEGORICAL_SET,
                categorical=data_spec_pb2.CategoricalSpec(
                    number_of_unique_values=1000,
                    is_already_integerized=True,
                ),
            )
        ]
    )
    if attribute_idx > 0:
      dataspec = self.full_data_spec
    proto_condition = decision_tree_pb2.NodeCondition(
        na_value=False,
        split_score=2,
        attribute=attribute_idx,
        condition=decision_tree_pb2.Condition(
            contains_condition=decision_tree_pb2.Condition.ContainsVector(
                elements=[1, 3]
            ),
        ),
    )

    self._assert_conditions_equivalent(condition, proto_condition, dataspec)

    self.assertEqual(
        condition.pretty(dataspec),
        "'f_categorical_set_large_vocab' intersect [1, 3] [score=2"
        " missing=False]",
    )

  @parameterized.parameters(0, 8)
  def test_condition_is_in_categorical_set_bitmap_valid_input(
      self, attribute_idx
  ):
    condition = condition_lib.CategoricalSetContainsCondition(
        attribute=attribute_idx,
        missing=False,
        score=2,
        mask=[1, 3],
    )
    dataspec = data_spec_pb2.DataSpecification(
        columns=[
            data_spec_pb2.Column(
                name="f_categorical_set_small_vocab",
                type=data_spec_pb2.ColumnType.CATEGORICAL_SET,
                categorical=data_spec_pb2.CategoricalSpec(
                    number_of_unique_values=4
                ),
            )
        ]
    )
    if attribute_idx > 0:
      dataspec = self.full_data_spec
    proto_condition = decision_tree_pb2.NodeCondition(
        na_value=False,
        split_score=2,
        attribute=attribute_idx,
        condition=decision_tree_pb2.Condition(
            contains_bitmap_condition=decision_tree_pb2.Condition.ContainsBitmap(
                elements_bitmap=b"\x0A"
            ),
        ),
    )

    self._assert_conditions_equivalent(condition, proto_condition, dataspec)

  def test_condition_sparse_oblique_valid_input(self):
    condition = condition_lib.NumericalSparseObliqueCondition(
        attributes=[0, 1],
        missing=False,
        score=2,
        weights=[1, 2],
        threshold=3,
    )
    dataspec = data_spec_pb2.DataSpecification(
        columns=[data_spec_pb2.Column(name="f"), data_spec_pb2.Column(name="g")]
    )
    proto_condition = decision_tree_pb2.NodeCondition(
        na_value=False,
        split_score=2,
        attribute=0,
        condition=decision_tree_pb2.Condition(
            oblique_condition=decision_tree_pb2.Condition.Oblique(
                attributes=[0, 1],
                weights=[1, 2],
                threshold=3,
            ),
        ),
    )

    self._assert_conditions_equivalent(condition, proto_condition, dataspec)

    self.assertEqual(
        condition.pretty(dataspec),
        "'f' x 1 + 'g' x 2 >= 3 [score=2 missing=False]",
    )

  def test_condition_sparse_oblique_empty_valid_input(self):
    condition = condition_lib.NumericalSparseObliqueCondition(
        attributes=[],
        missing=False,
        score=2,
        weights=[],
        threshold=3,
    )
    dataspec = data_spec_pb2.DataSpecification(columns=[data_spec_pb2.Column()])
    proto_condition = decision_tree_pb2.NodeCondition(
        na_value=False,
        split_score=2,
        attribute=-1,
        condition=decision_tree_pb2.Condition(
            oblique_condition=decision_tree_pb2.Condition.Oblique(
                attributes=[],
                weights=[],
                threshold=3,
            ),
        ),
    )

    self._assert_conditions_equivalent(condition, proto_condition, dataspec)

    self.assertEqual(
        condition.pretty(dataspec),
        "*nothing* >= 3 [score=2 missing=False]",
    )


if __name__ == "__main__":
  absltest.main()
