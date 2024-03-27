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


SUPPORTED_INPUT_DATA_DESCRIPTION = """\
A dataset can be one of the following:
- A Pandas DataFrame.
- A dictionary of column names (str) to values. Values can be lists of int, float, bool, str or bytes. Values can also be Numpy arrays.
- A YDF VerticalDataset
- A TensorFlow Batched Dataset.
- A typed (possibly sharded) path to a CSV file (e.g. csv:mydata).
- A list of typed paths (e.g. ["csv:mydata1", "csv:mydata2"]).
"""
