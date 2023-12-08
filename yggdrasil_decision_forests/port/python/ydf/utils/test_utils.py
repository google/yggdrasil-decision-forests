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

import logging
import os
import pathlib
from absl import flags
from absl.testing import absltest


def data_root_path() -> str:
  """Root directory of the repo."""
  return ""

def assertProto2Equal(self: absltest.TestCase, a, b):
  """Checks that protos "a" and "b" are equal."""
  self.assertEqual(a, b)

def pydf_test_data_path() -> str:
  return os.path.join(data_root_path(), "test_data")


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
