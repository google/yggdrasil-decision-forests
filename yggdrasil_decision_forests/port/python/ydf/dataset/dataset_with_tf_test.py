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

"""Tests for the dataset with TensorFlow dependency."""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from ydf.dataset import dataset


class DatasetWithTfTest(absltest.TestCase):

  def test_create_tensorflow_batched_dataset(self):
    ds_tf = tf.data.Dataset.from_tensor_slices({
        "a": np.array([1, 2, 3]),
    }).batch(1)
    ds = dataset.create_vertical_dataset(ds_tf)
    expected_data_spec = ds_pb.DataSpecification(
        created_num_rows=3,
        columns=(
            ds_pb.Column(
                name="a",
                type=ds_pb.ColumnType.NUMERICAL,
                dtype=ds_pb.DType.DTYPE_INT64,
                count_nas=0,
                numerical=ds_pb.NumericalSpec(
                    mean=2,
                    standard_deviation=0.8164965809277263,  # ~math.sqrt(2 / 3)
                    min_value=1,
                    max_value=3,
                ),
            ),
        ),
    )
    self.assertEqual(ds.data_spec(), expected_data_spec)


if __name__ == "__main__":
  absltest.main()
