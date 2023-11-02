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

"""Model tests depending on TensorFlow.

TensorFlow cannot be compiled in debug mode, so these tests are separated out to
improve debuggability of the remaining model tests.
"""

import os
import tempfile

from absl.testing import absltest
import numpy.testing as npt
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

from ydf.model import model_lib
from ydf.utils import test_utils


class TfModelTest(absltest.TestCase):

  def test_to_tensorflow_saved_model(self):
    model_path = os.path.join(
        test_utils.ydf_test_data_path(), "model", "abalone_regression_rf"
    )
    test_dataset_path = os.path.join(
        test_utils.ydf_test_data_path(), "dataset", "abalone.csv"
    )
    test_df = pd.read_csv(test_dataset_path)
    tf_test = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, "Rings")

    model = model_lib.load_model(model_path)
    ydf_predictions = model.predict(test_df)

    with tempfile.TemporaryDirectory() as tempdir:
      model.to_tensorflow_saved_model(tempdir)
      tf_model = tf.keras.models.load_model(tempdir)

    tfdf_predictions = tf_model.predict(tf_test)
    npt.assert_array_equal(ydf_predictions, tfdf_predictions.flatten())


if __name__ == "__main__":
  absltest.main()
