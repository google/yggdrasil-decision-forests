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

"""Utils for dealing with paths."""

from typing import List


def normalize_list_of_paths(paths: List[str]) -> str:
  """Formats a list of typed paths for consumption by YDFs reader.

  YDF expects a comma-separated list of untyped paths prefixed with the type.

  Args:
    paths: List of paths

  Returns:
    A typed string of the paths as comma-separated items.
  Raises:
    ValueError: The list is empty, or the paths in the listed are not all typed
    with the same type.
  """
  if not paths:
    raise ValueError("The list of paths to the dataset may not be empty")
  split_first_path = paths[0].split(":", maxsplit=1)
  if len(split_first_path) == 1:
    raise ValueError(
        "All typed paths to the dataset be typed with the same type, e.g.,"
        " ['csv:/path/file1', 'csv:/path/file2']"
    )
  else:
    # The first path is typed, remove types from all paths.
    prefix = split_first_path[0] + ":"
    if not all(path.startswith(prefix) for path in paths):
      raise ValueError(
          "All typed paths to the dataset should have the same type, e.g.,"
          " ['csv:/path/file1', 'csv:/path/file2']"
      )
    return prefix + ",".join([path[len(prefix) :] for path in paths])
