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

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from ydf.dataset import dataspec as dataspec_lib


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


if __name__ == "__main__":
  absltest.main()
