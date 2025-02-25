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

from typing import Any, Dict, Union

import jaxtyping
import numpy as np
import numpy.typing as np_typing

_FLATTENING_TOKEN = "::"


def flatten_weights(
    weights: Dict[str, Any], prefix=""
) -> Dict[str, Union[jaxtyping.Array, np_typing.NDArray]]:
  """Flatten weights for use with safetensors."""
  result = {}
  for k, v in weights.items():
    new_prefix = f"{prefix}{_FLATTENING_TOKEN}{k}" if prefix else k
    if isinstance(v, dict):
      result.update(flatten_weights(v, new_prefix))
    elif isinstance(v, (jaxtyping.Array, np.ndarray)):
      result[new_prefix] = v
    else:
      raise ValueError(f"Invalid weight type {type(v)} at {new_prefix}")
  return result


def deflatten_weights(flat_weights):
  """Unflatten weights for use with safetensors."""
  if not isinstance(flat_weights, dict):
    return flat_weights
  result = {}
  for k, v in flat_weights.items():
    split_k = k.split(_FLATTENING_TOKEN, 1)
    if len(split_k) == 1:
      result[k] = v
    else:
      if split_k[0] not in result:
        result[split_k[0]] = {}
      result[split_k[0]][split_k[1]] = v
  for res_k, res_v in result.items():
    result[res_k] = deflatten_weights(res_v)
  return result
