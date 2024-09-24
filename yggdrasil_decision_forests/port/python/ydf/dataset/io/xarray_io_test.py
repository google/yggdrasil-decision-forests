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

"""Test dataspec utilities for XArray."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import xarray as xr

from ydf.dataset.io import dataset_io_types
from ydf.dataset.io import xarray_io


class XArrayIOTest(parameterized.TestCase):

  def test_is_xarray(self):
    self.assertTrue(xarray_io.is_xarray_dataset(xr.Dataset()))
    self.assertFalse(xarray_io.is_xarray_dataset({}))

  @parameterized.parameters(
      (
          xr.Dataset({"feature": xr.DataArray([0, 1, 2])}),
          {"feature": np.asarray([0, 1, 2])},
      ),
      (
          xr.Dataset({"feature": xr.DataArray([0, 1, np.nan])}),
          {"feature": np.asarray([0, 1, np.nan])},
      ),
      (
          xr.Dataset({"feature": xr.DataArray(["x", "y", "z"])}),
          {"feature": np.asarray(["x", "y", "z"])},
      ),
      (
          xr.Dataset.from_dataframe(
              pd.DataFrame({"feature": ["x", "y", None]})
          ),
          {"feature": np.asarray(["x", "y", ""], dtype=np.object_)},
      ),
      (
          xr.Dataset.from_dataframe(
              pd.DataFrame({"feature": ["x", "y", np.nan]})
          ),
          {"feature": np.asarray(["x", "y", ""], dtype=np.object_)},
      ),
      (
          xr.Dataset.from_dataframe(
              pd.DataFrame({"feature": ["x", "y", pd.NA]})
          ),
          {"feature": np.asarray(["x", "y", ""], dtype=np.object_)},
      ),
  )
  def test_to_dict(
      self,
      data: dataset_io_types.IODataset,
      expected: dataset_io_types.DictInputValues,
  ):
    got = xarray_io.to_dict(data)
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
