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

"""Tests for distributed model learning."""

import dataclasses
import os
import pathlib
from typing import Callable, List, Sequence

from absl import logging
from absl.testing import absltest
import pandas as pd
import portpicker

from ydf.learner import specialized_learners
from ydf.learner import tuner as tuner_lib
from ydf.learner import worker
from ydf.utils import test_utils


@dataclasses.dataclass
class Workers:
  """Locally running workers.

  Attributes:
    ips: IP addresses of the workers.
    stop: Stops all the workers.
  """

  ips: Sequence[str]
  stop: Callable[[], None]


def create_in_process_workers(num_workers: int) -> Workers:
  """Creates a set of workers running locally."""

  # Select ports
  ports = [portpicker.pick_unused_port() for _ in range(num_workers)]

  # Start workers
  stop_worker_list = [
      worker.start_worker(port=port, blocking=False) for port in ports
  ]

  # To stop all workers
  def stop():
    for stop_worker in stop_worker_list:
      stop_worker()

  ips = [f"localhost:{port}" for port in ports]
  logging.info("ips: %s", ips)
  return Workers(
      ips=ips,
      stop=stop,
  )


def split_dataset(
    path: pathlib.Path, tmp_dir: pathlib.Path, num_shards: int
) -> List[pathlib.Path]:
  """Splits a csv file into multiple csv files."""

  dataset = pd.read_csv(path)
  num_row_per_shard = (dataset.shape[0] + num_shards - 1) // num_shards
  paths = []
  for shard_idx in range(num_shards):
    begin_idx = shard_idx * num_row_per_shard
    end_idx = (shard_idx + 1) * num_row_per_shard
    shard_dataset = dataset.iloc[begin_idx:end_idx]
    shard_path = tmp_dir / f"shard_{shard_idx}.csv"
    paths.append(shard_path)
    shard_dataset.to_csv(shard_path, index=False)
  return paths


class Utilities(absltest.TestCase):

  def test_local_worker(self):
    workers = create_in_process_workers(4)
    self.assertLen(workers.ips, 4)
    workers.stop()

  def test_split_dataset(self):
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    dataset_path = (
        test_utils.ydf_test_data_pathlib() / "dataset" / "adult_train.csv"
    )
    sharded_paths = split_dataset(dataset_path, tmp_dir, 10)
    self.assertLen(sharded_paths, 10)
    ds = pd.concat([pd.read_csv(path) for path in sharded_paths]).reset_index(
        drop=True
    )
    expected_ds = pd.read_csv(dataset_path)
    # Loading and concatenating the sharded dataset is equal to loading the
    # original dataset.
    pd.testing.assert_frame_equal(ds, expected_ds)


class DistributedGradientBoostedTreesLearnerTest(absltest.TestCase):

  def test_adult(self):
    # Prepare datasets
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    dataset_directory = test_utils.ydf_test_data_pathlib() / "dataset"
    splitted_train_ds = split_dataset(
        dataset_directory / "adult_train.csv", tmp_dir, 10
    )
    logging.info("Dataset paths: %s", splitted_train_ds)
    test_ds = dataset_directory / "adult_test.csv"

    # Start workers
    workers = create_in_process_workers(4)

    # Train model
    model = specialized_learners.DistributedGradientBoostedTreesLearner(
        label="income",
        working_dir=os.path.join(tmp_dir, "work_dir"),
        resume_training=True,
        num_trees=10,
        workers=workers.ips,
    ).train(",".join(map(str, splitted_train_ds)))

    workers.stop()
    self.assertAlmostEqual(model.evaluate(str(test_ds)).accuracy, 0.850, 1)


class HyperParameterTunerTest(absltest.TestCase):

  def test_adult_from_file(self):
    # Prepare datasets
    tmp_dir = pathlib.Path(self.create_tempdir().full_path)
    dataset_directory = test_utils.ydf_test_data_pathlib() / "dataset"
    train_ds = dataset_directory / "adult_train.csv"
    test_ds = dataset_directory / "adult_test.csv"

    # Start workers
    workers = create_in_process_workers(4)

    # Tune model
    tuner = tuner_lib.RandomSearchTuner(num_trials=10)
    tuner.choice("shrinkage", [0.2, 0.1, 0.05])
    tuner.choice("subsample", [1.0, 0.9, 0.8])

    model = specialized_learners.GradientBoostedTreesLearner(
        label="income",
        working_dir=os.path.join(tmp_dir, "work_dir"),
        num_trees=20,
        workers=workers.ips,
        tuner=tuner,
    ).train(str(train_ds))

    workers.stop()
    self.assertGreater(model.evaluate(str(test_ds)).accuracy, 0.850, 1)


if __name__ == "__main__":
  absltest.main()
