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

"""Test dataspec utilities for pandas."""

from absl.testing import absltest
import pandas as pd

# import polars as pl   # TODO: Re-enable.

from ydf.dataset.io import pandas_io
from ydf.utils import test_utils


class PandasIOTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.adult = test_utils.load_datasets("adult")

  def test_is_pandas(self):
    self.assertTrue(pandas_io.is_pandas_dataframe(pd.DataFrame()))
    self.assertFalse(pandas_io.is_pandas_dataframe({}))

  # TODO: Re-enable.
  # def test_polars_is_not_pandas(self):
  #   self.assertFalse(pandas_io.is_pandas_dataframe(pl.DataFrame()))

  def test_pandas_generator(self):
    ds = pandas_io.PandasBatchedExampleGenerator(self.adult.train_pd)
    self.assertEqual(ds.num_batches(100), 228)
    num_batches = 0
    for batch in ds.generate(batch_size=100, shuffle=False):
      self.assertSameElements(
          batch.keys(),
          [
              "age",
              "capital_gain",
              "capital_loss",
              "education",
              "education_num",
              "fnlwgt",
              "hours_per_week",
              "income",
              "marital_status",
              "native_country",
              "occupation",
              "race",
              "relationship",
              "sex",
              "workclass",
          ],
      )
      num_batches += 1
      if num_batches < 228:
        self.assertEqual(batch["age"].shape, (100,))

    self.assertEqual(num_batches, 228)


if __name__ == "__main__":
  absltest.main()
