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

"""Simple training benchmark for YDF and TF-DF."""

import functools
import os
import time
from typing import List, NamedTuple, Optional, Tuple, Union

import pandas as pd
import tensorflow_decision_forests as tfdf
import ydf


TEST_DATA_PATH = (
"google3/external/ydf_cc/yggdrasil_decision_forests/test_data/dataset"
)

def get_test_file(path):
  return  os.path.join(TEST_DATA_PATH, path)


class BenchmarkResult(NamedTuple):
  name: str
  wall_time_seconds: float
  cpu_time_seconds: float

  def __str__(self):
    return (
        f"{self.name:40}    "
        f"{self.wall_time_seconds:10.5f}    "
        f"{self.cpu_time_seconds:10.5f}"
    )


class Runner:
  """Wrapper for the benchmark runs."""

  def __init__(self):
    # Name and training time (in seconds) of each benchmarks.
    self._results: List[Union[BenchmarkResult, None]] = []

  def benchmark(
      self,
      name,
      callback,
      repetitions,
      warmup_repetitions=1,
  ):
    """Runs a sequence of benchmarks."""
    print(f"Running {name}")
    # Warmup
    for _ in range(warmup_repetitions):
      callback()

    begin_wall_time = time.perf_counter()
    begin_cpu_time = time.process_time()

    for _ in range(repetitions):
      callback()

    end_wall_time = time.perf_counter()
    end_cpu_time = time.process_time()

    result = BenchmarkResult(
        name=name,
        wall_time_seconds=(end_wall_time - begin_wall_time) / repetitions,
        cpu_time_seconds=(end_cpu_time - begin_cpu_time) / repetitions,
    )
    print(f"Completed {result}")
    self._results.append(result)

  def add_separator(self):
    self._results.append(None)

  def print_results(self):
    sep_length = 80
    print("=" * sep_length)
    print(f"{'Name':40}    {'Wall time (s)':10}    {'CPU time (s)':10}")
    print("=" * sep_length)
    for idx, result in enumerate(self._results):
      if result is None:
        if idx != 0:
          print("-" * sep_length, flush=True)
      else:
        print(result)
    print("=" * sep_length, flush=True)


def get_pd_dataset(
    name: str,
) -> Tuple[pd.DataFrame, ydf.Task, str, Optional[str]]:
  """Load the given dataset, returns dataset, task, label and ranking group."""
  csv_path = get_test_file(f"{name}.csv")
  df: pd.DataFrame = pd.read_csv(csv_path)
  labels = {
      "adult": "income",
      "iris": "class",
      "abalone": "Rings",
      "dna": "LABEL",
      "synthetic_ranking_train": "LABEL",
  }
  tasks = {
      "adult": ydf.Task.CLASSIFICATION,
      "iris": ydf.Task.CLASSIFICATION,
      "abalone": ydf.Task.REGRESSION,
      "synthetic_ranking_train": ydf.Task.RANKING,
      "dna": ydf.Task.CLASSIFICATION,
  }
  ranking_group = "GROUP" if name == "synthetic_ranking_train" else None
  return df, tasks[name], labels[name], ranking_group


def build_learning_params(configuration: str, num_threads: int):
  """Returns a dictionary with the parameters for the learning algorithm."""
  hyperparameters = {}
  if configuration == "oblique":
    hyperparameters.update({
        "split_axis": "SPARSE_OBLIQUE",
        "sparse_oblique_num_projections_exponent": 1.0,
        "sparse_oblique_normalization": "MIN_MAX",
    })
  hyperparameters["num_threads"] = num_threads
  return hyperparameters


def bench_train_ydf(
    df: pd.DataFrame,
    label: str,
    task: ydf.Task,
    hyperparameters: dict,
    algo: str,
):
  """Trains a model with YDF."""
  algo_to_function = {
      "rf": ydf.RandomForestLearner,
      "gbt": functools.partial(
          ydf.GradientBoostedTreesLearner,
          early_stopping="NONE",
      ),
      "cart": ydf.CartLearner,
  }
  train_func = algo_to_function[algo]
  train_func(label=label, task=task, **hyperparameters).train(df)


def bench_train_tfdf(
    df: pd.DataFrame,
    label: str,
    task: ydf.Task,
    hyperparameters: dict,
    algo: str,
):
  """Trains a model with TF-DF."""
  algo_to_function = {
      "rf": tfdf.keras.RandomForestModel,
      "gbt": functools.partial(
          tfdf.keras.GradientBoostedTreesModel, early_stopping="NONE"
      ),
      "cart": tfdf.keras.CartModel,
  }
  tfdf_task = tfdf.keras.Task.Value(task.value)
  ds = tfdf.keras.pd_dataframe_to_tf_dataset(df, label=label, task=tfdf_task)
  train_func = algo_to_function[algo]
  model = train_func(**hyperparameters, task=tfdf_task, verbose=0)
  model.fit(ds)


def bench_train(
    ds: str,
    configuration: str,
    algo: str,
    application: str,
    runner: Runner,
    num_threads: int,
    num_repetitions: int,
) -> None:
  """Train a model and report timing to the runner."""
  ydf.verbose(0)  # Don't print logs
  runner.add_separator()
  df, task, label, ranking_group = get_pd_dataset(ds)
  hyperparameters = build_learning_params(configuration, num_threads)
  hyperparameters["ranking_group"] = ranking_group
  bench = None
  if application == "tfdf":
    bench = functools.partial(bench_train_tfdf, df, label, task, hyperparameters, algo)
  if application == "ydf":
    bench = functools.partial(bench_train_ydf, df, label, task, hyperparameters, algo)
  runner.benchmark(
      f"{application};{ds};{algo};{configuration}",
      bench,
      repetitions=num_repetitions,
  )
