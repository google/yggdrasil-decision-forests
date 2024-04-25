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

"""Utility to measure the inference speed of a model with different backends.

For productionization of a model, use `model.benchmark(...)` instead.
"""

import dataclasses
import time
from typing import Any, Dict, Iterator, List, Optional

import jax
import numpy as np

from ydf.dataset import dataset as dataset_lib
from ydf.learner import specialized_learners
from ydf.model import export_jax
from ydf.model import generic_model
from ydf.utils import string_lib

_MICRO_SECONDS = 1e6  # Number of µs in a s.


@dataclasses.dataclass(frozen=True)
class Time:
  """An quantity of time expressed in seconds."""

  wall_time_seconds: float
  cpu_time_seconds: float

  @classmethod
  def now(cls) -> "Time":
    return Time(
        wall_time_seconds=time.perf_counter(),
        cpu_time_seconds=time.process_time(),
    )


@dataclasses.dataclass(frozen=True)
class RunConfiguration:
  """Generic configuration of a benchmark execution.

  batch_size: Batch size. If None, use the number of examples as batch size.
  num_warmup_rounds: Number of rounds before the benchmark.
  num_rounds: Number of rounds in the benchmark.
  profiler: If true, enable the profiler. For JAX, use xprof. Not used for other
    engines.
  show_logs: If true, prints details about the benchmark during its execution.
  """

  batch_size: Optional[int] = 1000
  num_warmup_rounds: int = 3
  num_rounds: int = 5
  profiler: bool = False
  show_logs: bool = False


@dataclasses.dataclass
class Benchmark:
  """A unit of benchmark result.

  Attributes:
    begin: Begin time in unix epoch format.
    end: End  time in unix epoch format.
    name: Name of the benchmark.
    engine: Implementation / engine runing the benchmark.
    per_example: Inference duration for a single example.
    num_examples: Number of examples.
    batch_size: Batch size.
    str_wall_time_seconds_us: String wall time per exaple in µs.
    str_cpu_time_seconds_us: String cpu time per exaple in µs.
  """

  begin: dataclasses.InitVar[Time]
  end: dataclasses.InitVar[Time]

  name: str
  engine: str
  per_example: Time = dataclasses.field(init=False)
  num_examples: int
  batch_size: int

  def __post_init__(self, begin: Time, end: Time):
    self.per_example = Time(
        wall_time_seconds=(end.wall_time_seconds - begin.wall_time_seconds)
        / self.num_examples,
        cpu_time_seconds=(end.cpu_time_seconds - begin.cpu_time_seconds)
        / self.num_examples,
    )

  @property
  def str_wall_time_seconds_us(self) -> str:
    return f"{self.per_example.wall_time_seconds * _MICRO_SECONDS:10.5f}"

  @property
  def str_cpu_time_seconds_us(self) -> str:
    return f"{self.per_example.cpu_time_seconds * _MICRO_SECONDS:10.5f}"

  def __str__(self) -> str:
    return f"""\
{self.name}
  wall-time (µs/ex): {self.str_wall_time_seconds_us}
  cpu-time (µs/ex):  {self.str_wall_time_seconds_us}
"""


def np_dict_to_jax_dict(df: Dict[str, np.ndarray]) -> Dict[str, jax.Array]:
  """Converts a dictionary of numpy array into a dictionary of jax arrays."""
  return {k: jax.numpy.asarray(v) for k, v in df.items()}


def build_synthetic_dataset(
    num_examples: int,
    seed: int,
    numerical_feature: bool = False,
    categorical_feature: bool = False,
    label_type: str = "regression",
) -> Dict[str, np.ndarray]:
  """Creates a synthetic dataset.

  Args:
    num_examples: Number of examples.
    seed: Seed for the pseudo random generator.
    numerical_feature: Has the dataset numerical features?
    categorical_feature: Has the dataset categorical features?
    label_type: Type of label.

  Returns:
    The dataset.
  """

  np.random.seed(seed)
  n1 = np.random.random(size=num_examples)
  n2 = np.random.random(size=num_examples)

  c1 = np.random.choice(["C1_X", "C1_Y", "C1_Z"], size=num_examples)
  c2 = np.random.choice(["C2_X", "C2_Y"], size=num_examples)

  c1_map = {"C1_X": 0.1, "C1_Y": 0.2, "C1_Z": 0.3}
  c2_map = {"C2_X": 0.1, "C2_Y": 0.3}

  raw_label = (
      1.0 * n1
      + 0.7 * n2
      + np.array([c1_map[x] for x in c1])
      + np.array([c2_map[x] for x in c2])
      + np.random.random(size=num_examples) * 0.2
  )

  label_map = {0: "a", 1: "b", 2: "c"}

  if label_type == "regression":
    label = raw_label
  elif label_type == "binary":
    l1 = np.median(raw_label)
    label = np.array([label_map[l >= l1] for l in raw_label])
  elif label_type == "multiclass":
    l1 = np.quantile(raw_label, 0.33)
    l2 = np.quantile(raw_label, 0.66)
    label = np.array([label_map[(l >= l1) + (l >= l2)] for l in raw_label])
  else:
    raise ValueError(f"Unknown label type: {label_type}")

  ds = {"label": label}
  if numerical_feature:
    ds["n1"] = n1
    ds["n2"] = n2
  if categorical_feature:
    ds["c1"] = c1
    ds["c2"] = c2

  return ds


def get_num_examples(dataset: Dict[str, np.ndarray]) -> int:
  """Number of examples in a dataset."""
  return len(next(iter(dataset.values())))


def gen_batch(
    dataset: Dict[str, Any], batch_size: int
) -> Iterator[Dict[str, Any]]:
  """Generates batches of examples."""

  num_examples = get_num_examples(dataset)
  i = 0
  while i < num_examples:
    begin_idx = i
    end_idx = min(i + batch_size, num_examples)
    yield {k: v[begin_idx:end_idx] for k, v in dataset.items()}
    i += batch_size


def run_cpp_engine(
    name: str,
    model: generic_model.GenericModel,
    dataset: Dict[str, np.ndarray],
    run_config: RunConfiguration,
) -> List[Benchmark]:
  """Runs benchmark on a model and dataset.

  Args:
    name: Name of the benchmark.
    model: Model to benchmark.
    dataset: Dataset to benchmark on.
    run_config: Benchmark run configuration.

  Returns:
    List of generated benchmarks.
  """
  num_examples = get_num_examples(dataset)

  vertical_ds = dataset_lib.create_vertical_dataset(
      dataset, data_spec=model.data_spec()
  )

  benchmarks = []
  for engine_name in model.list_compatible_engines():
    model.force_engine(engine_name)

    for _ in range(run_config.num_warmup_rounds):
      _ = model.predict(vertical_ds)

    begin_native_time = Time.now()
    for _ in range(run_config.num_rounds):
      _ = model.predict(vertical_ds)
    end_native_time = Time.now()

    benchmark = Benchmark(
        name=name,
        engine=f"c++ {engine_name}",
        begin=begin_native_time,
        end=end_native_time,
        num_examples=run_config.num_rounds * num_examples,
        batch_size=-1,  # Note: No need to batch.
    )
    if run_config.show_logs:
      print(benchmark)
    benchmarks.append(benchmark)

  return benchmarks


def run_jax_engine(
    name: str,
    model: generic_model.GenericModel,
    dataset: Dict[str, np.ndarray],
    run_config: RunConfiguration,
) -> List[Benchmark]:
  """Runs benchmark on a model and dataset.

  Args:
    name: Prefix to the benchmark names.
    model: Model to benchmark.
    dataset: Dataset to benchmark on.
    run_config: Benchmark run configuration.

  Returns:
    List of generated benchmarks.
  """
  num_examples = get_num_examples(dataset)
  batch_size = run_config.batch_size or num_examples

  benchmarks = []
  devices = jax.devices()

  dataset_without_labels = dataset.copy()
  del dataset_without_labels["label"]

  for device in devices:
    with jax.default_device(device):

      jax_model, feature_encoding = export_jax.to_jax_function(model)
      if feature_encoding is not None:
        input_values = feature_encoding.encode(dataset_without_labels)
      else:
        input_values = np_dict_to_jax_dict(dataset_without_labels)

      for _ in range(run_config.num_warmup_rounds):
        for batch in gen_batch(input_values, batch_size):
          _ = jax_model(batch).block_until_ready()

      begin_time = Time.now()
      for round_idx in range(run_config.num_rounds):
        if run_config.profiler:
          with jax.profiler.StepTraceAnnotation(
              "inference", step_num=round_idx
          ):
            for batch in gen_batch(input_values, batch_size):
              _ = jax_model(batch).block_until_ready()
        else:
          for batch in gen_batch(input_values, batch_size):
            _ = jax_model(batch).block_until_ready()

      end_time = Time.now()
      benchmark = Benchmark(
          begin_time,
          end_time,
          name,
          engine=f"Jax on {device}",
          num_examples=run_config.num_rounds * num_examples,
          batch_size=batch_size,
      )
      if run_config.show_logs:
        print(benchmark)
      benchmarks.append(benchmark)

  return benchmarks


def pretty_benchmarks(benchmarks: List[Benchmark]) -> str:
  """Pretty prints a list of benchmarks."""

  content = []
  row_labels = []
  for benchmark in benchmarks:
    row_labels.append(benchmark.name)
    content.append([
        benchmark.name,
        benchmark.engine,
        benchmark.str_wall_time_seconds_us,
        benchmark.str_cpu_time_seconds_us,
        benchmark.batch_size,
    ])

  return string_lib.table(
      content=content,
      column_labels=[
          "name",
          "engine",
          "wall-time (µs/ex)",
          "cpu-time (µs/ex)",
          "batch",
      ],
      data_row_separator=False,
      squeeze_column=True,
  )


def run_on_synthetic_dataset(
    name: str,
    synthetic_dataset_kwargs: Dict[str, Any],
    learner_kwargs: Dict[str, Any],
    num_train_examples: int,
    num_test_examples: int,
    run_config: RunConfiguration,
    cpp_engine: bool = False,
    jax_engine: bool = False,
) -> List[Benchmark]:
  """Runs benchmark on a synthetic dataset."""

  # Generate data
  train_ds = build_synthetic_dataset(
      num_train_examples, seed=1, **synthetic_dataset_kwargs
  )
  test_ds = build_synthetic_dataset(
      num_test_examples, seed=2, **synthetic_dataset_kwargs
  )

  # Build model
  model = specialized_learners.GradientBoostedTreesLearner(
      label="label",
      validation_ratio=0.0,
      **learner_kwargs,
  ).train(train_ds)

  benchmarks = []
  if cpp_engine:
    benchmarks.extend(run_cpp_engine(name, model, test_ds, run_config))
  if jax_engine:
    benchmarks.extend(run_jax_engine(name, model, test_ds, run_config))
  return benchmarks


def run_preconfigured(profiler: bool = False) -> List[Benchmark]:
  """Runs a set of preconfigured benchmarks."""

  common_kwargs = {
      "cpp_engine": True,
      "jax_engine": True,
      "num_train_examples": 10000,
      "num_test_examples": 10000,
      "run_config": RunConfiguration(profiler=profiler),
  }

  benchmarks = []

  benchmarks.extend(
      run_on_synthetic_dataset(
          name="reg gbt; num feat",
          synthetic_dataset_kwargs={"numerical_feature": True},
          learner_kwargs={"task": generic_model.Task.REGRESSION},
          **common_kwargs,
      )
  )

  benchmarks.extend(
      run_on_synthetic_dataset(
          name="reg gbt+cat; num feat",
          synthetic_dataset_kwargs={
              "numerical_feature": True,
              "categorical_feature": True,
          },
          learner_kwargs={"task": generic_model.Task.REGRESSION},
          **common_kwargs,
      )
  )

  benchmarks.extend(
      run_on_synthetic_dataset(
          name="multi-class gbt; num feat",
          synthetic_dataset_kwargs={
              "numerical_feature": True,
              "label_type": "multiclass",
          },
          learner_kwargs={"task": generic_model.Task.CLASSIFICATION},
          **common_kwargs,
      )
  )

  print(pretty_benchmarks(benchmarks), flush=True)

  return benchmarks
