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

"""Connectors for loading data from Pandas dataframes."""

import sys
from typing import Any, Dict, Iterator, Optional

from absl import logging
import numpy as np

from ydf.dataset.io import dataset_io_types
from ydf.dataset.io import generator as generator_lib


def import_pd():
  try:
    import pandas as pd  # pylint: disable=g-import-not-at-top

    return pd
  except ImportError:
    logging.warning(
        "Importing data from pandas dataframes requires pandas to be installed."
        " Install Pandas with pip using `pip install ydf[pandas]` or"
        " `pip install pandas`."
    )
    raise


def is_pandas_dataframe(data: dataset_io_types.IODataset) -> bool:
  if "pandas" in sys.modules:
    return isinstance(data, sys.modules["pandas"].DataFrame)
  return False


def to_dict(
    data: dataset_io_types.IODataset,
) -> Dict[str, dataset_io_types.InputValues]:
  """Converts a Pandas dataframe to a dict of numpy arrays."""
  pd = import_pd()

  assert isinstance(data, pd.DataFrame)
  if data.ndim != 2:
    raise ValueError("The pandas DataFrame must be two-dimensional.")
  data_dict = data.to_dict("series")

  for k in data_dict:
    if not isinstance(k, str):
      raise ValueError("The pandas DataFrame must have string column names.")

  def clean(values):
    if values.dtype == "object":
      return values.to_numpy(copy=False, na_value="")
    else:
      return values.to_numpy(copy=False)

  data_dict = {k: clean(v) for k, v in data_dict.items()}

  return data_dict


class PandasBatchedExampleGenerator(generator_lib.BatchedExampleGenerator):
  """Class to consume Pandas Dataframes."""

  def __init__(self, dataframe: Any):
    pd = import_pd()
    assert isinstance(dataframe, pd.DataFrame)
    self._dataframe = dataframe
    super().__init__(num_examples=len(dataframe))

  def generate(
      self,
      batch_size: int,
      shuffle: bool,
      seed: Optional[int] = None,
  ) -> Iterator[generator_lib.NumpyExampleBatch]:
    assert self._num_examples is not None
    if not shuffle:
      i = 0
      while i < self._num_examples:
        begin_idx = i
        end_idx = min(i + batch_size, self._num_examples)
        yield {
            str(k): v.iloc[begin_idx:end_idx].to_numpy()
            for k, v in self._dataframe.items()
        }
        i += batch_size
    else:
      if seed is None:
        raise ValueError("seed is required if shuffle=True")
      rng = np.random.default_rng(seed)
      idxs = rng.permutation(self._num_examples)
      i = 0
      while i < self._num_examples:
        begin_idx = i
        end_idx = min(i + batch_size, self._num_examples)
        selected_idxs = idxs[begin_idx:end_idx]
        yield {
            str(k): v.iloc[selected_idxs].to_numpy()
            for k, v in self._dataframe.items()
        }
        i += batch_size
