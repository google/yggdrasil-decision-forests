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

import logging
import sys
from typing import Dict
from ydf.dataset.io import dataset_io_types


def is_tensorflow_dataset(data: dataset_io_types.IODataset) -> bool:
  # Note: We only test if the dataset is a TensorFlow dataset if the object name
  # look like a TensorFlow object. This way, we avoid importing TF is not
  # necessary.
  str_class = str(type(data))
  if "tensorflow" in str_class and hasattr(data, "rebatch"):

    if data.__class__.__name__ in (
        "_BatchDataset",
        "_MapDataset",
        "DatasetV1Adapter",
        "CacheDataset",
    ):
      return True

    if "data.ops" in str_class:
      logging.warning(
          "The dataset %s object is not listed as a YDF compatible TensorFlow"
          " Dataset, but it looks like one",
          str_class,
      )
      return True

  return False


def to_dict(
    data: dataset_io_types.IODataset,
) -> Dict[str, dataset_io_types.InputValues]:
  """Converts a Tensorflow dataset to a dict of numpy arrays."""
  assert hasattr(data, "rebatch")
  full_batch = next(iter(data.rebatch(sys.maxsize)))
  return {k: v.numpy() for k, v in full_batch.items()}
