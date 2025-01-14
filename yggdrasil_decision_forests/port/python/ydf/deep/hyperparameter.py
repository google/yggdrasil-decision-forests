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

"""Utility to parse models and learners hyperparameters."""

from typing import Any, Dict


class HyperparameterConsumer:
  """Utility class to facilitate the consumption of hyperparameters.

  Usage example:

  ```python
  hp = HyperparameterConsumer({"a":1, "b": 0.5})
  hp.get_int("a")
  >> 1
  hp.get_int("c")
  >> error: c does not exist
  hp.get_optional_int("c")
  >> None
  hp.get_int("b")
  >> error: b is not an integer
  hp.finalize()
  >> error: b was not consumed
  hp.get_float("b")
  >> 0.5
  hp.finalize()
  hp.get_float("b")
  >> error: cannot get a value after it is finalized
  hp.finalize()
  >> error: already finalize
  ```
  """

  def __init__(self, values: Dict[str, Any]):
    self._values = values
    self._consumed = set()
    self._finalized = False

  def get_float(self, name: str) -> float:
    """Gets a float value."""
    value = self._get_value(name)
    if isinstance(value, int):
      # Allow for the value to be an integer.
      value = float(value)
    if not isinstance(value, float):
      raise ValueError(
          f"Hyperparameter {name!r} is expected to be a floating point value."
          f" Instead, got {value!r} of type {type(value)}"
      )
    self._consumed.add(name)
    return value

  def get_int(self, name: str) -> int:
    """Gets an integer value."""
    value = self._get_value(name)
    if isinstance(value, bool) or not isinstance(value, int):
      raise ValueError(
          f"Hyperparameter {name!r} is expected to be a integer value. Instead,"
          f" got {value!r} of type {type(value)}"
      )
    self._consumed.add(name)
    return value

  def get_optional_int(self, name: str) -> int:
    """Gets an integer value. Returns None is the value does not exist."""
    value = self._get_value(name)
    if isinstance(value, bool) or (
        not isinstance(value, int) and value is not None
    ):
      raise ValueError(
          f"Hyperparameter {name!r} is expected to be a integer or None value."
          f" Instead, got {value!r} of type {type(value)}"
      )
    self._consumed.add(name)
    return value

  def get_str(self, name: str) -> str:
    """Gets a string value."""
    value = self._get_value(name)
    if not isinstance(value, str):
      raise ValueError(
          f"Hyperparameter {name!r} is expected to be a string value. Instead,"
          f" got {value!r} of type {type(value)}"
      )
    self._consumed.add(name)
    return value

  def get_bool(self, name: str) -> bool:
    """Gets a bool value."""
    value = self._get_value(name)
    if not isinstance(value, bool):
      raise ValueError(
          f"Hyperparameter {name!r} is expected to be a bool value. Instead,"
          f" got {value!r} of type {type(value)}"
      )
    self._consumed.add(name)
    return value

  def finalize(self) -> None:
    """Ensures all the hyperparameters have been consumed."""
    if self._finalized:
      raise ValueError("Hyperparameter consumer already finalized")
    available = set(self._values.keys())
    missing = sorted(list(available - self._consumed))  # Determinisic
    if missing:
      raise ValueError(
          f"Some hyperparameters have not been consumed: {missing!r}"
      )
    self._finalized = True

  def _get_value(self, name: str) -> Any:
    if self._finalized:
      raise ValueError("Hyperparameter consumer already finalized")
    if name not in self._values:
      raise ValueError(
          f"The hyperparameter {name!r} does not exist. The available"
          f" hyperparameters are: {self._values!r}"
      )
    value = self._values[name]
    return value
