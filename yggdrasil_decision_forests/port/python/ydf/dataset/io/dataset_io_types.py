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

# Supported type of column input values.
InputValues = Union[np.ndarray, List[Any]]

DictInputValues = Dict[str, InputValues]


# Information about unrolled features
# Maps an original feature name to a list of unrolled features.
# e.g. {"f" : ["f.0", "f.1", "f.2"]}
UnrolledFeaturesInfo = Dict[str, List[str]]


# Supported types of datasets.
IODataset = Union[Dict[str, InputValues], "pd.DataFrame", str, Sequence[str]]

HOW_TO_FEED_NUMPY = """
YDF does not consume Numpy arrays directly. Instead, use a dictionary of Numpy arrays. For example:

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
  1. A dictionary of string (column names) to column values. The values of a column can be a list of int, float, bool, str, bytes, or a numpy array. A 2D numpy array is treated as a multi-dimensional column.
  2. A Pandas DataFrame.
  3. A YDF VerticalDataset created with `ydf.create_vertical_dataset`. This option is the most efficient when the same dataset is used multiple times.
  4. A batched TensorFlow Dataset.
  5. A typed path to a csv file e.g. "csv:/tmp/dataset.csv". See supported types below. The path can be sharded (e.g. "csv:/tmp/dataset@10") or globbed ("csv:/tmp/dataset*").
  6. A list of typed paths e.g. ["csv:/tmp/data1.csv", "csv:/tmp/data2.csv"]. See supported types below.

The supported file formats and corresponding prefixes are:
  - CSV file. prefix 'csv:'
  - Non-compressed TFRecord of Tensorflow Examples. prefix 'tfrecordv2+tfe:'
  - Compressed TFRecord of Tensorflow Examples. prefix 'tfrecord+tfe:'; not available in default public build.
"""
