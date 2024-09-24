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

"""Connectors for loading data from PyGrain data loaders and datasets."""

import sys
from typing import Any

from absl import logging
import numpy as np

from ydf.dataset.io import dataset_io_types


def import_pygrain():
  # pytype: disable=import-error
  # pylint: disable=g-import-not-at-top
  try:
    import grain.python as grain

    return grain
  except ImportError:
    logging.warning(
        "Importing data from PyGrain data loaders and datasets requires PyGrain"
        " to be installed. Install PyGrain with pip using `pip install grain`."
    )
    raise
  # pylint: enable=g-import-not-at-top
  # pytype: enable=import-error


def import_map_structure():
  # pytype: disable=import-error
  # pylint: disable=g-import-not-at-top
  try:
    from jax import tree_util

    return tree_util.tree_map
  except ImportError:
    try:
      import tree

      return tree.map_structure
    except ImportError:
      logging.warning(
          "Importing data from PyGrain data loaders and datasets requires JAX"
          " or tree to be installed. Install JAX with pip using `pip install"
          " jax` or tree with pip using `pip install dm-tree`."
      )
      raise
  # pylint: enable=g-import-not-at-top
  # pytype: enable=import-error


def is_pygrain(data: dataset_io_types.IODataset) -> bool:
  if "grain.python" in sys.modules:
    grain = sys.modules["grain.python"]
    return isinstance(
        data,
        (
            grain.DataLoader,
            grain.PyGrainDatasetIterator,
            grain.MapDataset,
            grain.IterDataset,
            grain.DatasetIterator,
        ),
    )
  return False


def to_dict(
    data: dataset_io_types.IODataset,
) -> dataset_io_types.DictInputValues:
  """Converts a PyGrain data loader or dataset to a dict of numpy arrays."""
  grain = import_pygrain()
  map_structure = import_map_structure()

  assert isinstance(
      data,
      (
          grain.DataLoader,
          grain.PyGrainDatasetIterator,
          grain.MapDataset,
          grain.IterDataset,
          grain.DatasetIterator,
      ),
  )

  def stack(*xs: Any) -> np.ndarray:
    """Stacks possibly heterogeneous elements into a numpy array."""
    try:
      return np.stack(xs)
    except ValueError:
      # If the elements do not have the same structure we have to convert them
      # to NumPy arrays of objects.
      return np.array(xs, dtype=np.object_)

  return map_structure(stack, *data)
