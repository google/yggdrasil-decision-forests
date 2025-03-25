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

"""Reader and writer implementations of TF Examples formats."""

from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import contextlib
import dataclasses
import datetime
import functools
import math
from multiprocessing import pool as multiprocessing_pool
import numpy as np

from ydf.util import dataset_io
from ydf.utils import log

# pytype: disable=import-error
# pylint: disable=g-import-not-at-top
try:
  import tensorflow as tf
except ImportError as exc:
  raise ImportError("Cannot import tensorflow") from exc
# pylint: enable=g-import-not-at-top
# pytype: enable=import-error


@dataclasses.dataclass
class ColumnSpec:
  default_value: Any
  dim: int


def read_tf_record(
    path: dataset_io.Path,
    compressed: bool,
    process: Optional[Callable[[tf.train.Example], tf.train.Example]],
    verbose: bool,
    threads: int,
) -> dataset_io.Data:
  """Implementation of read_tf_record. See tf_example.py."""

  def reader_generator(
      shard: str,
  ) -> contextlib.AbstractContextManager[Iterator[bytes]]:

    def iterator_fn() -> Iterator[bytes]:
      for tensor in tf.data.TFRecordDataset(
          shard,
          compression_type="GZIP" if compressed else "",
          buffer_size=10_000_000,
      ):
        yield tensor.numpy()

    @contextlib.contextmanager
    def cm():
      yield iterator_fn()

    return cm()

  return read_tensorflow_examples(
      reader_generator=reader_generator,
      path=path,
      process=process,
      verbose=verbose,
      threads=threads,
  )


def write_tf_record(
    data: dataset_io.Data,
    path: dataset_io.Path,
    compressed: bool,
    process: Optional[Callable[[tf.train.Example], tf.train.Example]],
    verbose: bool,
    threads: int,
) -> None:
  """Implementation of write_tf_record. See tf_example.py."""

  @contextlib.contextmanager
  def writer_generator(path: str):
    with tf.io.TFRecordWriter(
        path,
        options=tf.io.TFRecordOptions(
            compression_type="GZIP" if compressed else ""
        ),
    ) as writer:

      yield writer.write  # pytype: disable=attribute-error

  write_tensorflow_examples(
      writer_generator=writer_generator,
      data=data,
      path=path,
      process=process,
      verbose=verbose,
      threads=threads,
  )


def _inconsistent_error_message(key_difference: Set[str]) -> str:
  return (
      f"Inconsistent features {key_difference!r} between TensorFlow"
      " examples. Make sure all the examples have the same"
      " features. If you expect for the TensorFlow Examples to"
      " have different features, use `process` to either"
      " filter-out some features or to add them manually in all"
      " the examples. Note that YDF treats respectively NaN and ''"
      " as missing values for numerical and categorical features."
  )


def _read_shard(
    shard: str,
    reader_generator: Callable[
        [str], contextlib.AbstractContextManager[Iterator[bytes]]
    ],
    process: Optional[Callable[[tf.train.Example], tf.train.Example]],
    verbose: bool,
) -> Tuple[int, Dict[str, Tuple[np.ndarray, ColumnSpec]]]:
  """Reads a single shard of data.

  Args:
    shard: The file shard to read.
    reader_generator: The implementation able to read the shard examples.
    process: Optional preprocessing of the examples.
    verbose: If true, print more loggings.

  Returns:
    The number of read rows, and a map of column names to columns values +
    column spec.
  """

  # Map to each column name, the list of observed values and the missing value.
  local_data: Dict[str, Tuple[List[Any], ColumnSpec]] = {}
  local_num_examples = 0

  if verbose:
    log.info("Reading shard %s", shard)
  with reader_generator(shard) as reader:
    for record in reader:
      example = tf.train.Example.FromString(record)
      if process is not None:
        example = process(example)

      # Columns without values for this example
      example_keys = set(example.features.feature.keys())
      for key, (values, spec) in local_data.items():
        if key in example_keys:
          continue
        values.append(spec.default_value)

      # Column with values for this example
      for key, value in example.features.feature.items():
        if value.HasField("float_list"):
          dst_value = value.float_list.value
          single_default_value = math.nan
        elif value.HasField("bytes_list"):
          dst_value = value.bytes_list.value
          single_default_value = b""
        elif value.HasField("int64_list"):
          dst_value = value.int64_list.value
          single_default_value = 0
        else:
          raise ValueError(f"Unsupported value {value!r}")

        dim = len(dst_value)
        if dim == 0:
          # Giving an empty array of values is equivalent to not giving the
          # value.
          continue

        if key not in local_data:
          # This is a new column
          default_value = [single_default_value] * dim
          local_data[key] = (
              [default_value] * local_num_examples,
              ColumnSpec(
                  default_value=default_value,
                  dim=dim,
              ),
          )

        local_data[key][0].append(dst_value)

      local_num_examples += 1

  if not local_data:
    assert local_num_examples == 0
    return 0, {}

  return local_num_examples, {
      key: (
          np.array(values),
          ColumnSpec(default_value=np.array(spec.default_value), dim=spec.dim),
      )
      for key, (values, spec) in local_data.items()
  }


def read_tensorflow_examples(
    reader_generator: Callable[
        [str], contextlib.AbstractContextManager[Iterator[bytes]]
    ],
    path: dataset_io.Path,
    process: Optional[Callable[[tf.train.Example], tf.train.Example]],
    verbose: bool,
    threads: int,
) -> dataset_io.Data:
  """Reads a stream of TensorFlow Examples into a dictionary of numpy arrays."""

  time_begin = datetime.datetime.now()
  paths = dataset_io.expand_paths(path, input=True)

  if verbose:
    log.info("Reading %d shards", len(paths))

  num_examples = 0
  num_examples_per_shard = []
  data: Dict[str, Tuple[List[np.ndarray], ColumnSpec]] = {}
  with multiprocessing_pool.ThreadPool(threads) as pool:
    # Read the shards in parallel.
    for local_num_examples, shard_data in log.maybe_tqdm(
        pool.imap(
            functools.partial(
                _read_shard,
                reader_generator=reader_generator,
                process=process,
                verbose=verbose,
            ),
            paths,
        ),
        total=len(paths),
        desc="Read examples",
    ):
      if local_num_examples == 0:
        continue

      # Existing columns without values for this shard.
      shard_keys = set(shard_data.keys())
      for key, (values, spec) in data.items():
        if key in shard_keys:
          continue
        values.append(np.tile(spec.default_value, (local_num_examples, 1)))

      # Columns with values for this shard
      for key, (values, spec) in shard_data.items():
        if key not in data:
          # This is a new column.
          default_value = [
              np.tile(spec.default_value, (n, 1))
              for n in num_examples_per_shard
          ]
          data[key] = (default_value, spec)
        data[key][0].append(values)

      num_examples += local_num_examples
      num_examples_per_shard.append(local_num_examples)

  if data is None:
    raise ValueError("No data was read.")

  if verbose:
    log.info(
        "%d examples read in %s",
        num_examples,
        datetime.datetime.now() - time_begin,
    )

  # Finalize the shard aggregation
  def finalize_data(value: List[np.ndarray]) -> np.ndarray:
    value = np.concatenate(value, axis=0)
    if value.shape[1] == 1:
      value = np.squeeze(value, axis=1)
    return value

  return {key: finalize_data(values) for key, (values, _) in data.items()}


def write_tensorflow_examples(
    writer_generator: Callable[
        [str], contextlib.AbstractContextManager[Callable[[bytes], None]]
    ],
    data: dataset_io.Data,
    path: dataset_io.Path,
    process: Optional[Callable[[tf.train.Example], tf.train.Example]],
    verbose: bool,
    threads: int,
) -> None:
  """Writes a dict of numpy arrays into a stream of TensorFlow Examples."""

  time_begin = datetime.datetime.now()
  paths = dataset_io.expand_paths(path, input=False)
  num_examples = _num_example_in_dict(data)
  num_example_per_shard = _num_example_per_shard(num_examples, len(paths))

  if verbose:
    log.info("Writing %d examples in %d shards", num_examples, len(paths))

  def write_shard(arg: Tuple[int, str]) -> None:
    shard_idx, shard = arg
    begin_example_idx = num_example_per_shard * shard_idx
    end_example_idx = min(num_examples, num_example_per_shard * (shard_idx + 1))
    if verbose:
      log.info("Writing shard %s", shard)
    with writer_generator(shard) as writer:
      for example_idx in range(begin_example_idx, end_example_idx):
        tf_example = _dict_row_to_tf_example(data, example_idx)
        if process is not None:
          tf_example = process(tf_example)
        writer(tf_example.SerializeToString())

  with multiprocessing_pool.ThreadPool(threads) as pool:
    for _ in log.maybe_tqdm(
        pool.imap(write_shard, enumerate(paths)),
        total=len(paths),
        desc="Write examples",
    ):
      pass

  if verbose:
    log.info("Examples written in %s", datetime.datetime.now() - time_begin)


def _num_example_per_shard(num_examples: int, num_shards: int) -> int:
  """Returns the number of examples per shard."""
  return (num_examples + num_shards - 1) // num_shards


def _num_example_in_dict(ds: dataset_io.Data) -> int:
  """Gets the number of examples in a dictionary of values."""
  return len(next(iter(ds.values())))


def _dict_row_to_tf_example(
    data: dataset_io.Data, example_idx: int
) -> tf.train.Example:
  """Converts a row of a dict of numpy arrays into a TensorFlow Example."""

  example = tf.train.Example()

  for key, feature_values in data.items():

    values = feature_values[example_idx]
    dst_feature = example.features.feature[key]

    if isinstance(values, list):
      # List of values
      if values:
        first_value = values[0]
        # pytype: disable=attribute-error
        if isinstance(first_value, float):
          dst_feature.float_list.value.extend(values)
        elif isinstance(first_value, int):
          dst_feature.int64_list.value.extend(values)
        elif isinstance(first_value, str):
          dst_feature.bytes_list.value.extend(x.encode("utf-8") for x in values)
        elif isinstance(first_value, bytes):
          dst_feature.bytes_list.value.extend(values)
        else:
          raise ValueError(
              f"Unsupported value {values!r} of type {type(values)} for key"
              f" {key!r}"
          )
        # pytype: enable=attribute-error
    elif isinstance(feature_values, np.ndarray):
      if len(feature_values.shape) == 1:
        # Scalar values in numpy array
        values = values.item()
        if isinstance(values, float):
          dst_feature.float_list.value.append(values)
        elif isinstance(values, int):
          dst_feature.int64_list.value.append(values)
        elif isinstance(values, str):
          dst_feature.bytes_list.value.append(values.encode("utf-8"))
        elif isinstance(values, bytes):
          dst_feature.bytes_list.value.append(values)
      else:
        # Multi-dim values in numpy array
        if np.issubdtype(values.dtype, np.integer):
          dst_feature.int64_list.value.extend(values)
        elif np.issubdtype(values.dtype, np.floating):
          dst_feature.float_list.value.extend(values)
        elif np.issubdtype(values.dtype, np.bytes_):
          dst_feature.bytes_list.value.extend(values)
        elif np.issubdtype(values.dtype, np.str_):
          dst_feature.bytes_list.value.extend(x.encode("utf-8") for x in values)
        else:
          raise ValueError(
              f"Unsupported value {values!r} of type {type(values)} for key"
              f" {key!r}"
          )
        # pytype: enable=attribute-error
    elif isinstance(values, float):
      # Scalar python values
      dst_feature.float_list.value.append(values)
    elif isinstance(values, int):
      dst_feature.int64_list.value.append(values)
    elif isinstance(values, str):
      dst_feature.bytes_list.value.append(values.encode("utf-8"))
    elif isinstance(values, bytes):
      dst_feature.bytes_list.value.append(values)
    else:
      raise ValueError(
          f"Unsupported value {values!r} of type {type(values)} for key {key!r}"
      )

  return example
