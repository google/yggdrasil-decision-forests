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

"""Connectors for loading data from Polars dataframes."""

import sys
from typing import Dict

from absl import logging

from ydf.dataset.io import dataset_io_types


def import_pl():
  try:
    # pytype:disable=import-error
    import polars as pl  # pylint: disable=g-import-not-at-top
    # pytype:enable=import-error

    return pl
  except ImportError:
    logging.warning(
        "Importing data from polars dataframes requires polars to be installed."
        " Install Polars with pip using `pip install polars`."
    )
    raise


def is_polars_dataframe(data: dataset_io_types.IODataset) -> bool:
  if "polars" in sys.modules:
    return isinstance(data, sys.modules["polars"].DataFrame)
  return False


def to_dict(
    data: dataset_io_types.IODataset,
) -> Dict[str, dataset_io_types.InputValues]:
  """Converts a Polars dataframe to a dict of numpy arrays."""
  pl = import_pl()

  assert isinstance(data, pl.DataFrame)
  data_dict = data.to_dict(as_series=True)

  for k in data_dict:
    if not isinstance(k, str):
      raise ValueError("The Polars DataFrame must have string column names.")

  def clean(values):
    if values.dtype == pl.List:
      # Numpy cannot represent variable length objects. We feed the data as a
      # python list instead.
      # Note: We could try to np.stack the values, but this would make the
      # output type dependent on the source values (instead of just the source
      # value types).
      return values.to_list()
    else:
      return values.to_numpy()

  return {k: clean(v) for k, v in data_dict.items()}
