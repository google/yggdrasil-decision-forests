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

"""Utilities for unit tests."""

import dataclasses
import logging
import os
import pathlib
from typing import Optional, Sequence

from absl import flags
from absl.testing import absltest
import pandas as pd

from ydf.dataset import dataset
from ydf.dataset import dataspec


def data_root_path() -> str:
  """Root directory of the repo."""
  return ""

def assertProto2Equal(self: absltest.TestCase, a, b):
  """Checks that protos "a" and "b" are equal."""
  self.assertEqual(a, b)

def pydf_test_data_path() -> str:
  return os.path.join(data_root_path(), "test_data")


@dataclasses.dataclass(frozen=True)
class TrainAndTestDataset:
  """Training / test dataset as path, VerticalDataset and DataFrame."""

  train_path: str
  test_path: str
  train_pd: pd.DataFrame
  test_pd: pd.DataFrame
  train: dataset.VerticalDataset
  test: dataset.VerticalDataset


def ydf_test_data_path() -> str:
  return os.path.join(
      data_root_path(),
      "external/ydf_cc/yggdrasil_decision_forests/test_data",
  )


def ydf_test_data_pathlib() -> pathlib.Path:
  return (
      pathlib.Path(data_root_path())
      / "external/ydf_cc/yggdrasil_decision_forests/test_data"
  )


def load_datasets(
    name: str, column_args: Optional[Sequence[dataspec.Column]] = None
) -> TrainAndTestDataset:
  """Returns the given dataset loaded as different formats."""
  train_path = os.path.join(
      ydf_test_data_path(), "dataset", f"{name}_train.csv"
  )
  test_path = os.path.join(ydf_test_data_path(), "dataset", f"{name}_test.csv")
  train_pd = pd.read_csv(train_path)
  test_pd = pd.read_csv(test_path)
  train_vds = dataset.create_vertical_dataset(
      train_pd, columns=column_args, include_all_columns=True
  )
  test_vds = dataset.create_vertical_dataset(
      test_pd, data_spec=train_vds.data_spec()
  )
  return TrainAndTestDataset(
      train_path, test_path, train_pd, test_pd, train_vds, test_vds
  )


def toy_dataset():
  df = pd.DataFrame({
      "col_three_string": ["A", "A", "B", "B", "C"],
      "col_float": [1, 2.1, 1.3, 5.5, 2.4],
      "col_two_string": ["bar", "foo", "foo", "foo", "foo"],
      "weights": [3, 2, 3.1, 28, 3],
      "binary_int_label": [0, 0, 0, 1, 1],
  })
  return df


def toy_dataset_uplift():
  df = pd.DataFrame({
      "f1": [1, 2, 3, 4] * 10,
      "treatement": ["A", "A", "B", "B"] * 10,
      "effect_binary": [0, 1, 1, 1] * 10,
      "effect_numerical": [0.1, 0.5, 0.6, 0.7] * 10,
  })
  return df


# Exception raised in python when the c++ raises an invalid argument error
AbslInvalidArgumentError = (ValueError, RuntimeError)


def golden_check_string(
    test, value: str, golden_path: str, postfix: str = ""
) -> None:
  """Ensures that "value" is equal to the content of the file "golden_path".

  Args:
    test: A test.
    value: Value to test.
    golden_path: Path to golden file expressed from the root of the repo.
    postfix: Optional postfix to the path of the file containing the actual
      value.
  """

  golden_data = open(os.path.join(data_root_path(), golden_path)).read()

  if value != golden_data:
    value_path = os.path.join(
        absltest.TEST_TMPDIR.value, os.path.basename(golden_path) + postfix
    )
    logging.info("os.path.dirname(value_path): %s", os.path.dirname(value_path))
    os.makedirs(os.path.dirname(value_path), exist_ok=True)
    logging.info(
        "Golden test failed. Save the effetive value to %s", value_path
    )
    with open(value_path, "w") as f:
      f.write(value)

  test.assertEqual(value, golden_data)
