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

"""Benchmark the I/O speed."""

import dataclasses
import datetime
import os
import pprint
from typing import Dict, List, Optional

import numpy as np

from ydf.dataset import dataset as dataset_lib
from ydf.util import tf_example as tfrecord
from ydf.utils import log


@dataclasses.dataclass
class Options:
  """Configuration of the benchmark.

  Attributes:
    work_dir: Temporary directory to write and read the data.
    num_examples: Number of examples in the dataset.
    num_features: Number of features in the dataset.
    num_shards: Number of shards in the dataset.
    tests: List of tests to run. If None, run all the tests.
    create_workdir: Whether to create the workdir. If False, the workdir must
      already exist.
  """

  work_dir: str
  num_examples: int
  num_features: int
  num_shards: int
  tests: Optional[List[str]]
  create_workdir: bool


@dataclasses.dataclass
class Result:
  """Result of a benchmark."""

  test: str
  read_time: Optional[datetime.timedelta]
  write_time: Optional[datetime.timedelta]
  options: dataclasses.InitVar[Options]
  reading_example_per_seconds: Optional[int] = dataclasses.field(
      init=False, default=None
  )
  writing_example_per_seconds: Optional[int] = dataclasses.field(
      init=False, default=None
  )

  def __post_init__(self, options: Options):
    if self.read_time is not None:
      self.reading_example_per_seconds = int(
          options.num_examples / self.read_time.total_seconds()
      )
    if self.write_time is not None:
      self.writing_example_per_seconds = int(
          options.num_examples / self.write_time.total_seconds()
      )

  def __str__(self):
    return (
        f"test: {self.test!r}\n"
        f"read_time: {self.read_time}\n"
        f"write_time: {self.write_time}\n"
        f"example/s [reading]: {self.reading_example_per_seconds}\n"
        f"example/s [writing]: {self.writing_example_per_seconds}"
    )


def gen_numpy_data(options: Options) -> Dict[str, np.ndarray]:
  """Generates a random dataset."""
  return {
      f"f_{i}": np.random.rand(options.num_examples)
      for i in range(options.num_features)
  }


def test_tfrecord_python(work_dir: str, options: Options):
  data = gen_numpy_data(options)
  path = os.path.join(work_dir, f"tfrecord@{options.num_shards}")
  t1 = datetime.datetime.now()
  tfrecord.write_tf_record(data, path=path)
  t2 = datetime.datetime.now()
  tfrecord.read_tf_record(path)
  t3 = datetime.datetime.now()
  return Result(
      test="tfrecord",
      read_time=t3 - t2,
      write_time=t2 - t1,
      options=options,
  )


def run(
    work_dir: str,
    *,
    num_examples: int = 100_000,
    num_features: int = 50,
    num_shards: int = 20,
    tests: Optional[List[str]] = None,
    create_workdir: bool = True,
) -> List[Result]:
  """Runs the benchmarks.

  Args:
    work_dir: Directory to store the data.
    options: Benchmark options.

  Returns:
    List of results.
  """

  options = Options(
      work_dir=work_dir,
      num_examples=num_examples,
      num_features=num_features,
      num_shards=num_shards,
      tests=tests,
      create_workdir=create_workdir,
  )

  log.info("work_dir: %s\n%s", work_dir, pprint.pformat(options))

  if options.create_workdir:
    os.makedirs(work_dir, exist_ok=True)
  available_tests = [
      ("tfrecord_python", test_tfrecord_python),
  ]
  results = []
  for test_name, test_fn in available_tests:
    if options.tests is None or test_name in options.tests:
      log.info("Running test: %s", test_name)
      result = test_fn(work_dir, options)
      log.info("%s\n", result)
      results.append(result)
  return results
