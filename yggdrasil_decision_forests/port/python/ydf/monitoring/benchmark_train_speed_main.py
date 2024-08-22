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

r"""Simple Benchmark for PYDF and TF-DF.

Usage example:

  sudo apt install linux-cpupower
  sudo cpupower frequency-set --governor performance

  bazel run -c opt --copt=-mfma --copt=-mavx2 --copt=-mavx \
    //external/ydf_cc/yggdrasil_decision_forests/port/python/ydf/monitoring:benchmark_train_speed_main
"""

import itertools

from absl import app
from absl import flags

from ydf.monitoring import benchmark_train_speed as benchmark

AVAILABLE_APPLICATIONS = ["tfdf", "ydf"]
AVAILABLE_DATASETS = [
    "adult",
    "abalone",
    "iris",
    "dna",
    "synthetic_ranking_train",  # Incompatible with Random Forests
]
AVAILABLE_ALGORITHMS = ["rf", "gbt", "cart"]
AVAILABLE_CONFIGURATIONS = ["default", "oblique"]

_DATASETS = flags.DEFINE_list(
    "datasets",
    ["adult", "abalone", "iris", "dna"],
    f"Datasets to use for benchmarking. Possible values: {AVAILABLE_DATASETS}",
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
    f"Algorithms to use. Possible values: {AVAILABLE_ALGORITHMS}",
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


def main(argv):
  print("Running YDF benchmark")
  runner = benchmark.Runner()

  for application, config, ds, algo in itertools.product(
      _APPLICATIONS.value,
      _CONFIGURATIONS.value,
      _DATASETS.value,
      _ALGORITHMS.value,
  ):
    try:
      benchmark.bench_train(
          ds,
          config,
          algo,
          application,
          runner,
          _NUM_THREADS.value,
          _NUM_REPETITIONS.value,
      )
    except NameError:
      print(f"Application '{application}' not found.")

  print("All results (again)")
  runner.print_results()


if __name__ == "__main__":
  app.run(main)
