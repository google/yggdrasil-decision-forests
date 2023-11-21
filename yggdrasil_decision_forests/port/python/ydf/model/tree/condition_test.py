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
from ydf.model.tree import condition as condition_lib


class ConditionTest(parameterized.TestCase):

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


if __name__ == "__main__":
  absltest.main()
