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

"""Simple Benchmark for PYDF and its sister projects.

Usage example:
sudo apt install linux-cpupower
sudo cpupower frequency-set --governor performance
bazel run -c opt <fastest possible flags> \
//external/ydf_cc/yggdrasil_decision_forests/port/python/utils/benchmark:benchmark
"""

import functools
import itertools
import os
import time
from typing import List, NamedTuple, Union

from absl import app
from absl import flags
import pandas as pd
import tensorflow_decision_forests as tfdf

from ydf.dataset import dataset
from ydf.learner import generic_learner
from ydf.learner import specialized_learners

AVAILABLE_APPLICATIONS = ["tfdf", "pydf"]
AVAILABLE_DATASETS = ["adult", "abalone", "iris", "dna"]
AVAILABLE_ALGORITHMS = ["rf", "gbt", "cart"]
AVAILABLE_CONFIGURATIONS = ["default", "oblique"]

_DATASETS = flags.DEFINE_list(
    "datasets",
    AVAILABLE_DATASETS,
    "Datasets to use for benchmarking. If not specified, all datasets are"
    f" used. Possible values: {AVAILABLE_DATASETS}",
)
_APPLICATIONS = flags.DEFINE_list(
    "applications",
    AVAILABLE_APPLICATIONS,
    "Applications to use, defaults to all available. Possible values:"
    f" {AVAILABLE_APPLICATIONS}",
)
_ALGORITHMS = flags.DEFINE_list(
    "algorithms",
    ["rf", "gbt"],
    "Algorithms to use, defaults to [rf, gbt]. Possible values:"
    f" {AVAILABLE_ALGORITHMS}",
)
_CONFIGURATIONS = flags.DEFINE_list(
    "configurations",
    AVAILABLE_CONFIGURATIONS,
    "Hyperparameter configurations to benchmark. Defaults to all available"
    " configurations. Possible values:"
    f" {AVAILABLE_CONFIGURATIONS}",
)
_NUM_REPETITIONS = flags.DEFINE_integer(
    "num_repetitions",
    5,
    "Number of repetitions (outside warmup) to use.",
    lower_bound=1,
)
_NUM_THREADS = flags.DEFINE_integer(
    "num_threads",
    16,
    "Number of threads to use.",
    lower_bound=1,
)


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


def get_pd_dataset(name: str):
  """Load the given dataset to a dataframe, return it with task and label."""
  csv_path = get_test_file(f"{name}.csv")
  df = pd.read_csv(csv_path)
  labels = {
      "adult": "income",
      "iris": "class",
      "abalone": "Rings",
      "dna": "LABEL",
  }
  tasks = {
      "adult": generic_learner.Task.CLASSIFICATION,
      "iris": generic_learner.Task.CLASSIFICATION,
      "abalone": generic_learner.Task.REGRESSION,
      "dna": generic_learner.Task.CLASSIFICATION,
  }
  return df, labels[name], tasks[name]


def build_learning_params(configuration: str):
  """Returns a dictionary with the parameters for the learning algorithm."""
  hyperparameters = {}
  if configuration == "oblique":
    hyperparameters.update({
        "split_axis": "SPARSE_OBLIQUE",
        "sparse_oblique_num_projections_exponent": 1.0,
        "sparse_oblique_normalization": "MIN_MAX",
    })
  hyperparameters["num_threads"] = _NUM_THREADS.value
  return hyperparameters


def bench_train_pydf(
    df: pd.DataFrame,
    label: str,
    task: generic_learner.Task,
    hp_dict: dict,
    algo: str,
):
  algo_to_function = {
      "rf": specialized_learners.RandomForestLearner,
      "gbt": functools.partial(
          specialized_learners.GradientBoostedTreesLearner,
          early_stopping="NONE",
      ),
      "cart": specialized_learners.CartLearner,
  }
  ds = dataset.create_vertical_dataset(df)
  train_func = algo_to_function[algo]
  train_func(label=label, task=task, **hp_dict).train(ds)


def bench_train_tfdf(
    df: pd.DataFrame,
    label: str,
    task: generic_learner.Task,
    hp_dict: dict,
    algo: str,
):
  algo_to_function = {
      "rf": tfdf.keras.RandomForestModel,
      "gbt": functools.partial(
          tfdf.keras.GradientBoostedTreesModel, early_stopping="NONE"
      ),
      "cart": tfdf.keras.CartModel,
  }
  ds = tfdf.keras.pd_dataframe_to_tf_dataset(df, label=label, task=task)
  train_func = algo_to_function[algo]
  model = train_func(**hp_dict, task=task, verbose=0)
  model.fit(ds)


def bench_train(
    ds: str,
    configuration: str,
    algo: str,
    application: str,
    runner: Runner,
):
  """Train a model and report timing to the runner."""
  runner.add_separator()
  df, label, task = get_pd_dataset(ds)
  hp_dict = build_learning_params(configuration)
  bench = None
  if application == "tfdf":
    bench = functools.partial(bench_train_tfdf, df, label, task, hp_dict, algo)
  if application == "pydf":
    bench = functools.partial(bench_train_pydf, df, label, task, hp_dict, algo)
  runner.benchmark(
      f"{application};{ds};{algo};{configuration}",
      bench,
      repetitions=_NUM_REPETITIONS.value,
  )


def main(argv):
  print("Running PYDF benchmark")
  runner = Runner()

  for application, config, ds, algo in itertools.product(
      _APPLICATIONS.value,
      _CONFIGURATIONS.value,
      _DATASETS.value,
      _ALGORITHMS.value,
  ):
    try:
      bench_train(ds, config, algo, application, runner)
    except NameError:
      print(f"Application '{application}' not found.")

  print("All results (again)")
  runner.print_results()


if __name__ == "__main__":
  app.run(main)
