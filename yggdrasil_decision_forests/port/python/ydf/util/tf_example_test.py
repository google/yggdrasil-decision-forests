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

import math
import os
from absl import logging
from absl.testing import absltest
import numpy as np
import tensorflow as tf
from ydf.util import tf_example
from ydf.utils import test_utils


class TFExampleTest(absltest.TestCase):

  def test_write_and_read(self):
    original_ds = {
        "f1": np.array([1, 2, 3]),
        "f2": np.array([1.1, 2.2, math.nan]),
        "f3": np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]),
        "f4": np.array([b"X", b"Y", b"Z"]),
        "f5": np.array(
            [[b"X", b"Y", b"Z"], [b"Z", b"Y", b"X"], [b"Z", b"Y", b"X"]]
        ),
    }
    tmp_dir = self.create_tempdir().full_path
    path = os.path.join(tmp_dir, "file")
    tf_example.write_tf_record(original_ds, path=path)
    read_ds = tf_example.read_tf_record(path)
    logging.info("read_ds:\n%s", read_ds)
    test_utils.assert_almost_equal(original_ds, read_ds)

  def test_read_compressed(self):
    ds_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "dataset",
        "toy.tfe-tfrecord-00001-of-00002",
    )

    def process(example):
      for key in list(example.features.feature.keys()):
        if key != "Num_1":
          del example.features.feature[key]
      return example

    ds = tf_example.read_tf_record(ds_path, process=process)
    logging.info("%s", ds)
    self.assertEqual(ds["Num_1"].shape, (1,))

  def test_read_non_compressed(self):
    ds_path = os.path.join(
        test_utils.ydf_test_data_path(),
        "dataset",
        "toy.nocompress-tfe-tfrecord-00000-of-00002",
    )

    def process(example):
      for key in list(example.features.feature.keys()):
        if key != "Num_1":
          del example.features.feature[key]
      return example

    ds = tf_example.read_tf_record(ds_path, compressed=False, process=process)
    logging.info("%s", ds)
    self.assertEqual(ds["Num_1"].shape, (3,))

  def test_read_inconsistent(self):
    path1 = self.create_tempfile().full_path
    path2 = self.create_tempfile().full_path
    with tf.io.TFRecordWriter(
        path1,
        options=tf.io.TFRecordOptions(compression_type="GZIP"),
    ) as writer:
      # 0
      example = tf.train.Example()
      example.features.feature["a"].float_list.value.append(1.0)
      writer.write(example.SerializeToString())
      # 1
      example = tf.train.Example()
      example.features.feature["b"].int64_list.value.append(1)
      example.features.feature["f"].bytes_list.value.extend([b"Y", b"Z"])
      writer.write(example.SerializeToString())
      # 2
      example = tf.train.Example()
      example.features.feature["c"].bytes_list.value.append(b"X")
      example.features.feature["g"].bytes_list.value.append(b"X")
      writer.write(example.SerializeToString())
    with tf.io.TFRecordWriter(
        path2,
        options=tf.io.TFRecordOptions(compression_type="GZIP"),
    ) as writer:
      # 3
      example = tf.train.Example()
      example.features.feature["d"].bytes_list.value.extend([b"Y", b"Z"])
      example.features.feature["e"].int64_list.value.append(3)
      writer.write(example.SerializeToString())
      # 4
      example = tf.train.Example()
      example.features.feature["a"].float_list.value.append(2.0)
      example.features.feature["g"].bytes_list.value.append(b"XYZ")
      writer.write(example.SerializeToString())

    read_ds = tf_example.read_tf_record([path1, path2])
    expected_ds = {
        "a": np.array([1.0, math.nan, math.nan, math.nan, 2.0]),
        "b": np.array([0, 1, 0, 0, 0]),
        "c": np.array([b"", b"", b"X", b"", b""]),
        "d": np.array(
            [[b"", b""], [b"", b""], [b"", b""], [b"Y", b"Z"], [b"", b""]],
        ),
        "e": np.array([0, 0, 0, 3, 0]),
        "f": np.array(
            [[b"", b""], [b"Y", b"Z"], [b"", b""], [b"", b""], [b"", b""]],
        ),
        "g": np.array([b"", b"", b"X", b"", b"XYZ"]),
    }
    test_utils.assert_almost_equal(expected_ds, read_ds)


if __name__ == "__main__":
  absltest.main()
