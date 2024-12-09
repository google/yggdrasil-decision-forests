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

"""Dataset generator for dict of numpy arrays."""

from typing import Dict, Iterator, Optional
import numpy as np
from ydf.dataset.io import generator as generator_lib


class NumpyDictBatchedExampleGenerator(generator_lib.BatchedExampleGenerator):
  """Class to consume dictionaries of Numpy arrays."""

  def __init__(self, data: Dict[str, np.ndarray]):
    self._data = data
    super().__init__(num_examples=len(next(iter(data.values()))))

  def generate(
      self,
      batch_size: int,
      shuffle: bool,
      seed: Optional[int] = None,
  ) -> Iterator[generator_lib.NumpyExampleBatch]:
    assert self._num_examples is not None
    if not shuffle:
      i = 0
      while i < self._num_examples:
        begin_idx = i
        end_idx = min(i + batch_size, self._num_examples)
        yield {str(k): v[begin_idx:end_idx] for k, v in self._data.items()}
        i += batch_size
    else:
      if seed is None:
        raise ValueError("seed is required if shuffle=True")
      rng = np.random.default_rng(seed)
      idxs = rng.permutation(self._num_examples)
      i = 0
      while i < self._num_examples:
        begin_idx = i
        end_idx = min(i + batch_size, self._num_examples)
        selected_idxs = idxs[begin_idx:end_idx]
        yield {str(k): v[selected_idxs] for k, v in self._data.items()}
        i += batch_size
