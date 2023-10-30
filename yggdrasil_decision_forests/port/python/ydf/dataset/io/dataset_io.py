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

"""Common functionality for all dataset I/O connectors."""

from typing import Dict

from ydf.dataset.io import dataset_io_types
from ydf.dataset.io import pandas_io
from ydf.dataset.io import tensorflow_io


def cast_input_dataset_to_dict(
    data: dataset_io_types.IODataset,
) -> Dict[str, dataset_io_types.InputValues]:
  """Transforms the input dataset into a dictionary of values."""
  if pandas_io.is_pandas_dataframe(data):
    return pandas_io.to_dict(data)
  elif tensorflow_io.is_tensorflow_dataset(data):
    return tensorflow_io.to_dict(data)

  elif isinstance(data, dict):
    # Dictionary of values
    return data

  # TODO: Maybe this error should be raised at a layer above this one?
  raise ValueError(
      "Cannot import dataset from"
      f" {type(data)}.\n{dataset_io_types.SUPPORTED_INPUT_DATA_DESCRIPTION}"
  )
