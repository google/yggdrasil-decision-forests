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
from ydf.util import tf_example
from ydf.utils import test_utils


class TFExampleTest(absltest.TestCase):

  def test_write_and_read(self):
    original_ds = {
        "f1": np.array([1, 2, 3]),
        "f2": np.array([1.1, 2.2, math.nan]),
        "f3": np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]),
        "f4": np.array(["X", "Y", "Z"]),
        "f5": np.array([["X", "Y", "Z"], ["Z", "Y", "X"], ["Z", "Y", "X"]]),
    }
    tmp_dir = self.create_tempdir().full_path
    path = os.path.join(tmp_dir, "file")
    tf_example.write_tf_record(original_ds, path=path)
    read_ds = tf_example.read_tf_record(path)
    logging.info("read_ds:\n%s", read_ds)
    test_utils.test_almost_equal(original_ds, read_ds)

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


if __name__ == "__main__":
  absltest.main()
