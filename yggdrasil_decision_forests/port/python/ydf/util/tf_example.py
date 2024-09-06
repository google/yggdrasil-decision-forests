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

"""Reader and writer of TF Examples formats."""

from typing import Callable, Optional

from ydf.util import dataset_io
from ydf.utils import log


def read_tf_record(
    path: dataset_io.Path,
    *,
    compressed: bool = True,
    process: Optional[
        Callable[["tf.train.Example"], "tf.train.Example"]  # pytype: disable=name-error
    ] = None,  # pylint: disable=bad-whitespace
    verbose: bool = False,
    threads: int = 20,
) -> dataset_io.Data:
  """Reads a TensorFlow Records data and return a dict of numpy arrays.

  This method can be used to read TFRecord before applying YDF on it.

  Warning: Reading examples in python is very slow. Consider providing paths to
  YDF directly (e.g. `model.predict("record:" + path)`) instead (~20x
  faster).

  Usage example:

  ```python
  import ydf

  # Load a dataset
  ds = ydf.util.read_tf_record(path="/path/to/tfrecord")

  # Apply some pre-processing
  ds["my_label"] = np.log(ds["my_label"])

  # Train a model
  ydf.RandomForestLearner(label="my_label",
    task=ydf.Task.REGRESSION).train(ds)
  ```

  This method requires for all the TF Examples to have the same features and for
  all the features to have the same type and number of values. If your TF Record
  encode missing values by skipping features, you can use the `process` argument
  to add missing values manually:

  ```python
  import math

  def process(example: tf.train.Example):
    # Add missing values for categorical features.
    for key in ["feature_1", "feature_2]:
      if key not in example.features.feature:
        example.features.feature[key].bytes_list.value.append(b"")

    # Add missing values for numerical features.
    for key in ["feature_3", "feature_4]:
      if key not in example.features.feature:
        example.features.feature[key].float_list.value.append(math.nan)

    return example

  read_ds = tf_example.read_tf_recordio(path, process=process)
  ```

  Args:
    path: Path or list of paths to TFRecord files. Supports sharded paths.
    compressed: Whether the TFRecord is compressed.
    process: Optional function to process each TF Example. Can be used to filter
      out some features or to fix some example values (e.g., to ensure all the
      example have consistent feature values).
    verbose: If True, print status of the dataset reading.
    threads: Number of reading threads.

  Returns:
    A dict of numpy arrays.
  """

  log.warning(
      "ydf.util.read_tf_record is slow. For large datasets, consider providing"
      " paths to YDF directly (e.g. `model.predict('record:' + path)`).",
      message_id=log.WarningMessage.TFE_READING_IN_PYTHON_IS_SLOW,
  )

  from ydf.util import tf_example_impl  # pylint: disable=g-importing-member,g-import-not-at-top

  return tf_example_impl.read_tf_record(
      path=path,
      compressed=compressed,
      process=process,
      verbose=verbose,
      threads=threads,
  )


def write_tf_record(
    data: dataset_io.Data,
    *,
    path: dataset_io.Path,
    compressed: bool = True,
    process: Optional[
        Callable[["tf.train.Example"], "tf.train.Example"]  # pytype: disable=name-error
    ] = None,  # pylint: disable=bad-whitespace
    verbose: bool = False,
    threads: int = 20,
) -> None:
  """Writes a dict of numpy arrays into a TensorFlow Record.

  This method can be used to prepare TFRecord for distributed training.

  Usage example:

  ```python
  import ydf
  import numpy as np

  # Generate a dataset
  dataset = {
      "f1": np.array([1, 2, 3]),
      "f2": np.array([1.1, 2.2, 3.3])}

  # Write the dataset
  ydf.util.write_tf_recordio(dataset, path="/path/to/tfrecord")
  ```

  Args:
    data: A dict of numpy arrays. Support sharded paths.
    path: Path or list of paths to TFRecord files.
    compressed: Whether the TFRecord is compressed.
    process: Optional function to process each TF Example.
    verbose: If True, print status of the dataset writing.
    threads: Number of writing threads.
  """

  from ydf.util import tf_example_impl  # pylint: disable=g-importing-member,g-import-not-at-top

  tf_example_impl.write_tf_record(
      data=data,
      path=path,
      compressed=compressed,
      process=process,
      verbose=verbose,
      threads=threads,
  )
