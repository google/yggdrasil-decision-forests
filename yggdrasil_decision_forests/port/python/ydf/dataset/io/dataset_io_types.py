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

"""Typing annotations for dataset I/O connectors."""

import typing
from typing import Any, Dict, List, Sequence, Union

import numpy as np

if typing.TYPE_CHECKING:
  import pandas as pd  # pylint: disable=unused-import,g-bad-import-order
  import xarray as xr  # pylint: disable=unused-import,g-bad-import-order
  import polars as pl  # pylint: disable=unused-import,g-bad-import-order

# Supported type of column input values.
InputValues = Union[np.ndarray, List[Any]]

DictInputValues = Dict[str, InputValues]


# Information about unrolled features
# Maps an original feature name to a list of unrolled features.
# e.g. {"f" : ["f.0", "f.1", "f.2"]}
UnrolledFeaturesInfo = Dict[str, List[str]]


# Supported types of datasets.
IODataset = Union[
    DictInputValues,  # Dictionary of values (e.g., lists, NumPy arrays)
    "pd.DataFrame",  # A Pandas DataFrame
    str,  # A typed path e.g. "csv:/tmp/train.csv". Supports globs.
    Sequence[str],  # A list of typed paths
    "xr.Dataset",  # A XArray dataset
    "pl.DataFrame",  # A Polars DataFrame
    # Not listed: TensorFlow Datasets (e.g., CacheDataset, _BatchDataset),
    # PyGrain DataLoaders, PyGrain Datasets (e.g., MapDataset, IterDataset),
    # PyGrain Iterators (e.g., PyGrainDatasetIterator, DatasetIterator).
    Any,
]

HOW_TO_FEED_NUMPY = """
YDF does not consume Numpy arrays directly. Instead, use a dictionary of NumPy arrays. For example:

Instead of:
  ```python
  dataset = np.array([[1, 2, 3], [4, 5, 6]])
  model = ydf.RandomForestLearner(label="label").train(dataset)
  ```

Do:
  ```python
  dataset = {
      "label": np.array([1, 4]),
      "features": np.array([[2, 3], [5, 6]]),  # A two dimensional feature
  }
  model = ydf.RandomForestLearner(label="label").train(dataset)
  ```

And instead of:
  ```python
  model.predict(np.array([[2, 3], [5, 6]]))
  ```

Do:
  ```python
  model.predict({"features": np.array([[2, 3], [5, 6]])})
  ```
"""


SUPPORTED_INPUT_DATA_DESCRIPTION = """\
A dataset can be one of the following:
  1. A dictionary of string (column names) to column values. The values of a column can be a list of int, float, bool, str, bytes, or a NumPy array. A 2D NumPy array is treated as a multi-dimensional column.
  2. A Pandas DataFrame. Does not support multi-dimensional columns.
  3. A Polars DataFrame.
  4. A Xarray dataset.
  5. A YDF VerticalDataset created with `ydf.create_vertical_dataset`. This option is the most efficient when the same dataset is used multiple times.
  6. A batched TensorFlow Dataset.
  7. A typed path to a csv file e.g. "csv:/tmp/dataset.csv". See supported types below. The path can be sharded (e.g. "csv:/tmp/dataset@10") or globbed ("csv:/tmp/dataset*").
  8. A list of typed paths e.g. ["csv:/tmp/data1.csv", "csv:/tmp/data2.csv"]. See supported types below.
  9. A PyGrain DataLoader or Dataset (experimental, Linux only).

The supported file formats and corresponding prefixes are:
  - 'csv:' CSV files. Great for small datasets and integration with other
    tools. CSV file do not support multi-dimensional columns.
  - 'avro:' Avro files. Great for large datasets. Avro files support
    multi-dimensional columns.
  - 'tfrecord:' Compressed TFRecord of Tensorflow Examples. Official dataset format for TensorFlow. Great to work with TensorFlow and TFX pipelines.
  - 'tfrecord-nocompression:' Non-compressed TFRecord of Tensorflow Examples. Only here is you forgot to enable compression when creating the TFRecord.
  
For Googlers, see go/ydf for more formats.
"""
