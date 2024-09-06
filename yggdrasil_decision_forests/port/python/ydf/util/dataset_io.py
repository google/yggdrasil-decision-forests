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

"""Utilities for dataset IO operations."""

import itertools
from typing import Dict, Sequence, Union

import numpy as np


Data = Dict[str, np.ndarray]
Path = Union[str, Sequence[str]]


def expand_paths(path: Path, input: bool) -> Sequence[str]:
  """Expands a path to a list of paths."""

  if isinstance(path, str):
    path = [path]

  return path
