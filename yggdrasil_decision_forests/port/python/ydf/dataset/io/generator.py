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

"""Utility to handle datasets."""

import abc
from typing import Dict, Iterator, Optional
import numpy as np

# A single batch of data in various formats. The attribute values are indexed by
# attribute names.
NumpyExampleBatch = Dict[str, np.ndarray]


class BatchedExampleGenerator(abc.ABC):
  """A class able to generate batches of examples."""

  def __init__(self, num_examples: Optional[int]):
    self._num_examples = num_examples

  @property
  def num_examples(self) -> Optional[int]:
    """Number of examples in the dataset."""
    return self._num_examples

  def num_batches(self, batch_size: int) -> Optional[int]:
    if self._num_examples is None:
      return None
    return (self._num_examples + batch_size - 1) // batch_size

  @abc.abstractmethod
  def generate(
      self,
      batch_size: int,
      shuffle: bool,
      seed: Optional[int] = None,
  ) -> Iterator[NumpyExampleBatch]:
    """Generate an iterator."""
    raise NotImplementedError


def get_num_examples(batch: NumpyExampleBatch) -> int:
  """Gets the number of examples in a batch."""
  return len(next(iter(batch.values())))
