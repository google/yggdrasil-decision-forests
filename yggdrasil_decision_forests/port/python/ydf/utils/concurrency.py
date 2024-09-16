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

"""Utilities for concurrency."""

import os
from ydf.utils import log


def determine_optimal_num_threads(training: bool):
  """Returns the optimal number of computation threads."""
  num_threads = os.cpu_count()
  if num_threads is None:
    log.warning("Cannot determine the number of CPUs. Set `num_threads=6`")
    num_threads = 6
  else:
    # TODO: Remove 32 limit.
    if training and num_threads > 32:
      log.warning(
          "The `num_threads` argument is not set and the number of CPU is"
          " os.cpu_count()=%d > 32. Set `num_threads` manually to use more than"
          " 32 threads.",
          num_threads,
      )
      num_threads = 32
  return num_threads
