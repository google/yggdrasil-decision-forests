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

"""The Log Book is a utility keep track and analyse experiment results."""

import enum
import json
import os
import sqlite3
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd

# The values stored in an experiment.
Value = Optional[Union[str, float, int, bool, List, Dict, Set, Tuple]]
# The unique identifier of an experiment.
ExperimentKey = Dict[str, Value]
# The results of an experiment.
ExperimentResult = Dict[str, Value]

# Reserved experiment sub-keys. Those fields are generated automatically and
# cannot be set by the user.
_RESERVED_KEYS = ["id", "timestamp"]


class Mode(enum.Enum):
  """Backend used to store the experiment values."""

  SQLITE = "SQLITE"


class LogBook:
  """A utility to key track of ML experiments.

  Usage example:

    ```python
    # Build a log book. The pointed location might already contain results.
    # Multiple log books can possibly use the same location at the same time
    # (e.g. in different processes).
    lb = LogBook(path="~/experiment")

    # Create an experiment identifier.
    key = {"param1": 1, "param2": "abc"}

    # Check that the experiment does not already exist. In practice, you might
    # simply want to skip already existing experiments.
    assert not lb.exist(key)

    # Create some fake results.
    result = {"obs1": 2, "obs2": [1,2,3], "obs3": {"a":1} }

    # Record the experiment results.
    lb.new(key, result)

    # Show at all the recorded experiments in a table.
    lb.to_dataframe()
    id  date  param1  param2  obs1  obs2     obs3
    0   ***   1       abc     2     [1,2,3]  {"a":1}

    # The keys of the identifier or the results do not have to be consistant.
    # For example, we can record a new experiment with a new "param3" key.
    lb.new({"param1": 1, "param2": "abc", "param3": 5}, result)
    lb.to_dataframe()
    id  date  param1  param2  params3   obs1  obs2     obs3
    0   ***   1       abc     NA        2     [1,2,3]  {"a":1}
    1   ***   1       abc     5         2     [1,2,3]  {"a":1}

    # In the case the new key already implicitely existing (e.g. you forget to
    # record it), you can specify key default values in the log book
    # constructor. This way, displays and experiment queries remain consistant
    # between your old and new experiments.
    #
    # Create a new log book with a default experiment key value.
    lb = LogBook(path="~/experiment", default_keys={"param3":-1})
    lb.to_dataframe()
    id  date  param1  param2  params3   obs1  obs2     obs3
    0   ***   1       abc     -1        2     [1,2,3]  {"a":1}
    1   ***   1       abc     5         2     [1,2,3]  {"a":1}

    # You can query subset of experiments. For example, the next line only show
    # the experiment with param1=1.
    lb.to_dataframe({"param1":1})

    # Pandas's plot method makes it easy to plot results. For example, the
    # following line plot the "obs3" for the experiments with param1=1.
    lb.to_dataframe({"param1":1})["obs3"].plot()
    ```

    Attributes:
      directory: Path to the directory containing the experiment data.
      print_num_experiments: If true, print the number of experiments when
        initializing the LogBook.
      mode: Backend used to store the experiment values. Only SQLITE is
        supported.
      default_keys: Dictionary of default keys for new experiments.
  """

  def __init__(
      self,
      directory: str,
      print_num_experiments: bool = True,
      mode: Mode = Mode.SQLITE,
      default_keys: Optional[ExperimentKey] = None,
  ):
    """Initialize the log book in a directory."""

    # Only implemented mode
    assert mode == Mode.SQLITE

    self._default_keys = default_keys or {}

    self._directory = directory
    if not self._is_initialized():
      self._initialize()

    self._conn = sqlite3.connect(self._db_path(), timeout=60)
    self._cursor = self._conn.cursor()

    if print_num_experiments:
      print(f"Found {self.num_experiments()} experiments")

  def num_experiments(self) -> int:
    """Number of recoded experiments."""
    self._cursor.execute("SELECT COUNT(*) FROM experiments")
    return self._cursor.fetchone()[0]

  def exist(self, key: ExperimentKey) -> bool:
    """Tests if an existing exist."""
    key = self._augment_key(key)
    query = "SELECT key FROM experiments"
    self._cursor.execute(query)
    for (row_serialized_key,) in self._cursor.fetchall():
      row_key = self._augment_key(json.loads(row_serialized_key))
      if row_key == key:
        return True
    return False

  def add(self, key: ExperimentKey, result: ExperimentResult) -> None:
    """Creates a new experiment.

    Fails if the experiment exist already.

    Args:
      key: Key of the experiment. A dictionary containing scalar values (float,
        str, int, bool) or composed objects (list, dict, set, tuple). Cannot
        contain the keys "id" or "timestamp".
      result: Result of the experiment. A dictionary containing scalar values
        (float, str, int, bool) or composed objects (list, dict, set, tuple).
    """

    if not isinstance(key, dict):
      raise ValueError("`key` is not a dictionary")
    if not isinstance(result, dict):
      raise ValueError("`result` is not a dictionary")
    for reserved_key in _RESERVED_KEYS:
      if reserved_key in key:
        raise ValueError(f"`key` contains a reserved key `{reserved_key}`")
      if reserved_key in result:
        raise ValueError(f"`result` contains a reserved key `{reserved_key}`")

    serialized_key = json.dumps(key)
    serialized_result = json.dumps(result)
    self._cursor.execute(
        "INSERT INTO experiments (key, result) VALUES (?, ?)",
        (
            serialized_key,
            serialized_result,
        ),
    )
    self._conn.commit()

  def to_dataframe(
      self, key_filter: Optional[ExperimentKey] = None
  ) -> pd.DataFrame:
    """Creates a Pandas Dataframe with all the experiments.

    Args:
      key_filter: If set, only return experiments with a key containing (i.e.,
        super-set) "key_filter".

    Returns:
      A dataframe with the experiments.
    """
    query = "SELECT id, timestamp, key, result FROM experiments"
    self._cursor.execute(query)
    records = []
    for row_id, row_timestamp, row_key, row_result in self._cursor.fetchall():
      key = self._augment_key(json.loads(row_key))
      if key_filter is not None:
        skip_experiment = False
        for k, v in key_filter.items():
          if k not in key:
            continue
          if v != key[k]:
            skip_experiment = True
            break
        if skip_experiment:
          continue
      result = json.loads(row_result)
      records.append({
          "id": row_id,
          "timestamp": row_timestamp,
          **key,
          **result,
      })
    return pd.DataFrame.from_records(records)

  def _augment_key(self, key: ExperimentKey) -> ExperimentKey:

    if not isinstance(key, dict):
      raise ValueError(
          f"The key should be a dictionary. Instead, got {type(key)}."
      )

    new_key = key.copy()
    for k, v in self._default_keys.items():
      if k not in new_key:
        new_key[k] = v
    return new_key

  def _db_path(self) -> str:
    return os.path.join(self._directory, "records.db")

  def _is_initialized(self) -> bool:
    """Checks if the database is initialized."""
    return os.path.exists(self._db_path())

  def _initialize(self) -> None:
    """Initializes the database."""

    os.makedirs(self._directory, exist_ok=True)
    with sqlite3.connect(self._db_path()) as conn:
      cursor = conn.cursor()
      cursor.execute("""
      CREATE TABLE IF NOT EXISTS experiments (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
          key TEXT NOT NULL,
          result TEXT NOT NULL
      )
      """)
      conn.commit()
