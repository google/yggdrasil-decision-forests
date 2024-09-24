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

"""Connectors for loading data from Xarray datasets."""

import sys
from typing import Dict

from absl import logging

from ydf.dataset.io import dataset_io_types


def import_xarray():
  try:
    import xarray as xr  # pylint: disable=g-import-not-at-top

    return xr
  except ImportError:
    logging.warning(
        "Importing data from xarray datasets requires xarray to be installed."
        " Install xarray with pip using `pip install xarray`. See"
        " https://docs.xarray.dev/en/stable/getting-started-guide/installing.html"
        " to install xarrays with extentions."
    )
    raise


def is_xarray_dataset(data: dataset_io_types.IODataset) -> bool:
  if "xarray" in sys.modules:
    return isinstance(data, sys.modules["xarray"].Dataset)
  return False


def to_dict(
    data: dataset_io_types.IODataset,
) -> Dict[str, dataset_io_types.InputValues]:
  """Converts a Xarray dataframe to a dict of numpy arrays."""
  xr = import_xarray()

  assert isinstance(data, xr.Dataset)

  for k in data:
    if not isinstance(k, str):
      raise ValueError("The xarray Dataset must have string column names.")

  def clean(values):
    if values.dtype == "object":
      return values.fillna("").to_numpy()
    else:
      return values.to_numpy()

  data_dict = {k: clean(v) for k, v in data.items()}

  return data_dict
