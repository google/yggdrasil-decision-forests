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

from absl.testing import absltest
from absl.testing import parameterized
import xarray as xr
from ydf.learner import specialized_learners as learners
from ydf.utils import test_utils


class XArrayTest(parameterized.TestCase):

  def test_simple(self):
    ds = xr.Dataset({
        "f1": xr.DataArray([1, 2, 3, 4], dims="example"),
        "f2": ("example", [0.1, 0.2, 0.3, 0.4]),
        "f3": xr.Variable("example", ["X", "Y", "", "X"]),
        "f4": (("example", "pixel"), [[1, 2], [3, 4], [5, 6], [7, 8]]),
        "l": ("example", ["A", "B", "A", "B"]),
    })
    model = learners.GradientBoostedTreesLearner(label="l").train(ds)
    self.assertEqual(
        model.input_feature_names(),
        ["f1", "f2", "f3", "f4.0_of_2", "f4.1_of_2"],
    )
    self.assertEqual(model.label(), "l")
    _ = model.predict(ds)
    _ = model.evaluate(ds)

  def test_adult(self):
    adult = test_utils.load_datasets("adult")
    train_ds = xr.Dataset.from_dataframe(adult.train_pd)
    test_ds = xr.Dataset.from_dataframe(adult.test_pd)
    model = learners.GradientBoostedTreesLearner(label="income").train(train_ds)
    self.assertEqual(
        model.input_feature_names(),
        [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education_num",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country",
        ],
    )
    self.assertEqual(model.label(), "income")
    _ = model.predict(test_ds)
    evaluation = model.evaluate(test_ds)
    self.assertAlmostEqual(evaluation.accuracy, 0.8721, delta=0.01)


if __name__ == "__main__":
  absltest.main()
