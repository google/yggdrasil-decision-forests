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

from typing import List

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb


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
