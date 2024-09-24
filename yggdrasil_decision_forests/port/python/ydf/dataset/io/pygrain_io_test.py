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

"""Test dataspec utilities for PyGrain data loaders and datasets."""

from absl.testing import absltest
from absl.testing import parameterized
import grain.python as grain
import numpy as np

from ydf.dataset.io import dataset_io_types
from ydf.dataset.io import pygrain_io


class PygrainIoTest(parameterized.TestCase):

  @parameterized.parameters(
      (grain.load([0, 1, 2], num_epochs=1), True),
      (grain.MapDataset.source([0, 1, 2]), True),
      (grain.MapDataset.source([0, 1, 2]).to_iter_dataset(), True),
      ({}, False),
  )
  def test_is_pygrain(self, data: dataset_io_types.IODataset, expected: bool):
    self.assertEqual(pygrain_io.is_pygrain(data), expected)
    self.assertEqual(pygrain_io.is_pygrain(data.__iter__()), expected)

  @parameterized.parameters(
      (
          grain.load(
              [{"feature": 0}, {"feature": 1}, {"feature": 2}], num_epochs=1
          ),
          {"feature": np.asarray([0, 1, 2])},
      ),
      (
          grain.load(
              [{"feature": "x"}, {"feature": "y"}, {"feature": "z"}],
              num_epochs=1,
          ),
          {"feature": np.asarray(["x", "y", "z"])},
      ),
      (
          grain.load(
              [
                  {"feature": np.asarray([0])},
                  {"feature": np.asarray([1])},
                  {"feature": np.asarray([2])},
              ],
              num_epochs=1,
          ),
          {"feature": np.asarray([[0], [1], [2]])},
      ),
      (
          grain.load(
              [
                  {"feature": np.asarray(["x"])},
                  {"feature": np.asarray(["y"])},
                  {"feature": np.asarray(["z"])},
              ],
              num_epochs=1,
          ),
          {"feature": np.asarray([["x"], ["y"], ["z"]])},
      ),
      (
          grain.load(
              [
                  {"feature": np.asarray([0, 1])},
                  {"feature": np.asarray([2])},
              ],
              num_epochs=1,
          ),
          {
              "feature": np.asarray(
                  [np.asarray([0, 1]), np.asarray([2])], dtype=np.object_
              )
          },
      ),
      (
          grain.load(
              [
                  {"feature": np.asarray(["x", "y"])},
                  {"feature": np.asarray(["z"])},
              ],
              num_epochs=1,
          ),
          {
              "feature": np.asarray(
                  [np.asarray(["x", "y"]), np.asarray(["z"])], dtype=np.object_
              )
          },
      ),
      (
          grain.MapDataset.source(
              [{"feature": 0}, {"feature": 1}, {"feature": 2}]
          ),
          {"feature": np.asarray([0, 1, 2])},
      ),
      (
          grain.MapDataset.source(
              [{"feature": "x"}, {"feature": "y"}, {"feature": "z"}]
          ),
          {"feature": np.asarray(["x", "y", "z"])},
      ),
      (
          grain.MapDataset.source([
              {"feature": np.asarray([0])},
              {"feature": np.asarray([1])},
              {"feature": np.asarray([2])},
          ]),
          {"feature": np.asarray([[0], [1], [2]])},
      ),
      (
          grain.MapDataset.source([
              {"feature": np.asarray(["x"])},
              {"feature": np.asarray(["y"])},
              {"feature": np.asarray(["z"])},
          ]),
          {"feature": np.asarray([["x"], ["y"], ["z"]])},
      ),
      (
          grain.MapDataset.source([
              {"feature": np.asarray([0, 1])},
              {"feature": np.asarray([2])},
          ]),
          {
              "feature": np.asarray(
                  [np.asarray([0, 1]), np.asarray([2])], dtype=np.object_
              )
          },
      ),
      (
          grain.MapDataset.source([
              {"feature": np.asarray(["x", "y"])},
              {"feature": np.asarray(["z"])},
          ]),
          {
              "feature": np.asarray(
                  [np.asarray(["x", "y"]), np.asarray(["z"])], dtype=np.object_
              )
          },
      ),
      (
          grain.MapDataset.source(
              [{"feature": 0}, {"feature": 1}, {"feature": 2}]
          ).to_iter_dataset(),
          {"feature": np.asarray([0, 1, 2])},
      ),
      (
          grain.MapDataset.source(
              [{"feature": "x"}, {"feature": "y"}, {"feature": "z"}]
          ).to_iter_dataset(),
          {"feature": np.asarray(["x", "y", "z"])},
      ),
      (
          grain.MapDataset.source([
              {"feature": np.asarray([0])},
              {"feature": np.asarray([1])},
              {"feature": np.asarray([2])},
          ]).to_iter_dataset(),
          {"feature": np.asarray([[0], [1], [2]])},
      ),
      (
          grain.MapDataset.source([
              {"feature": np.asarray(["x"])},
              {"feature": np.asarray(["y"])},
              {"feature": np.asarray(["z"])},
          ]).to_iter_dataset(),
          {"feature": np.asarray([["x"], ["y"], ["z"]])},
      ),
      (
          grain.MapDataset.source([
              {"feature": np.asarray([0, 1])},
              {"feature": np.asarray([2])},
          ]).to_iter_dataset(),
          {
              "feature": np.asarray(
                  [np.asarray([0, 1]), np.asarray([2])], dtype=np.object_
              )
          },
      ),
      (
          grain.MapDataset.source([
              {"feature": np.asarray(["x", "y"])},
              {"feature": np.asarray(["z"])},
          ]).to_iter_dataset(),
          {
              "feature": np.asarray(
                  [np.asarray(["x", "y"]), np.asarray(["z"])], dtype=np.object_
              )
          },
      ),
  )
  def test_to_dict(
      self,
      data: dataset_io_types.IODataset,
      expected: dataset_io_types.DictInputValues,
  ):
    for data in [data, data.__iter__()]:
      got = pygrain_io.to_dict(data)
      self.assertSetEqual(set(got.keys()), set(expected.keys()))
      for key in got.keys():
        self.assertEqual(len(got[key]), len(expected[key]))
        self.assertEqual(
            np.asarray(got[key]).dtype, np.asarray(expected[key]).dtype
        )
        for i, got_value in enumerate(got[key]):
          np.testing.assert_array_equal(got_value, expected[key][i])


if __name__ == "__main__":
  absltest.main()
