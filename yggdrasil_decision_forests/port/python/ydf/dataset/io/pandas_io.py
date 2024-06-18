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
from typing import Dict

from absl import logging

from ydf.dataset.io import dataset_io_types


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
