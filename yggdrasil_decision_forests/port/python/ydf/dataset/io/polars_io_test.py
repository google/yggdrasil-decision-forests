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

"""Test dataspec utilities for Polars."""

from absl.testing import absltest
import pandas as pd

## import polars as pl   # TODO: Re-enable.

from ydf.dataset.io import polars_io


class PolarsIOTest(absltest.TestCase):

  # TODO: Re-enable.
  # def test_is_polars(self):
  #   self.assertTrue(polars_io.is_polars_dataframe(pl.DataFrame()))
  #   self.assertFalse(polars_io.is_polars_dataframe({}))

  def test_pandas_is_not_polars(self):
    self.assertFalse(polars_io.is_polars_dataframe(pd.DataFrame()))


if __name__ == "__main__":
  absltest.main()
