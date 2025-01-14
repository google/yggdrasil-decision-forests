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

from typing import Dict, Union
import jax
from ydf.dataset.io import generator as generator_lib

# A single batch of data in various formats. The attribute values are indexed by
# attribute names.
NumpyExampleBatch = generator_lib.NumpyExampleBatch
JaxExampleBatch = Dict[str, jax.Array]


def get_num_examples(batch: Union[NumpyExampleBatch, JaxExampleBatch]) -> int:
  """Gets the number of examples in a batch."""
  return len(next(iter(batch.values())))


def batch_numpy_to_jax(src: NumpyExampleBatch) -> JaxExampleBatch:
  """Converts a batch of examples from numpy to jax format."""
  return {k: jax.numpy.asarray(v) for k, v in src.items()}
