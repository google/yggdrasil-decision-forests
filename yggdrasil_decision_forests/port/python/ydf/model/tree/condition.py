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

"""Conditions / splits for non-leaf nodes."""

from typing import Sequence
from yggdrasil_decision_forests.dataset import data_spec_pb2


def _bitmap_has_item(bitmap: bytes, value: int) -> bool:
  """Checks if the "value"-th bit is set."""

  byte_idx = value // 8
  sub_bit_idx = value & 7
  return (bitmap[byte_idx] & (1 << sub_bit_idx)) != 0


def bitmap_to_items(
    column_spec: data_spec_pb2.Column, bitmap: bytes
) -> Sequence[int]:
  """Converts a bitmap into a list of elements."""

  return [
      value_idx
      for value_idx in range(column_spec.categorical.number_of_unique_values)
      if _bitmap_has_item(bitmap, value_idx)
  ]
