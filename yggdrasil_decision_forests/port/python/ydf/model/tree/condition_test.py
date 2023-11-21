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
from absl.testing import parameterized
from yggdrasil_decision_forests.dataset import data_spec_pb2
from ydf.model.tree import condition as condition_lib


class ConditionTest(parameterized.TestCase):

  @parameterized.parameters(
      (b"\x2D\x03", 10, [0, 2, 3, 5, 8, 9]),  # b1100101101 => 0x032D
      (b"", 0, []),
  )
  def test_bitmap_to_items(
      self,
      bitmap: bytes,
      number_of_unique_values: int,
      expected_list: Sequence[int],
  ):
    column_spec = data_spec_pb2.Column(
        categorical=data_spec_pb2.CategoricalSpec(
            number_of_unique_values=number_of_unique_values
        )
    )
    self.assertEqual(
        condition_lib.bitmap_to_items(column_spec, bitmap),
        expected_list,
    )
