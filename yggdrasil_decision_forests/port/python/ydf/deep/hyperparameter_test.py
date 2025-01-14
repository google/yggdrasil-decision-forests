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

import re
from absl.testing import absltest
from absl.testing import parameterized
from ydf.deep import hyperparameter as hyperparameter_lib


class HyperparameterTest(parameterized.TestCase):

  def test_consumer(self):
    c = hyperparameter_lib.HyperparameterConsumer(
        {"a": 1, "b": 0.5, "c": None, "d": "hello", "e": True}
    )
    self.assertEqual(c.get_int("a"), 1)
    self.assertEqual(c.get_float("a"), 1.0)
    self.assertEqual(c.get_str("d"), "hello")
    self.assertEqual(c.get_bool("e"), True)
    with self.assertRaisesRegex(
        ValueError,
        re.escape("The hyperparameter 'non_existing' does not exist"),
    ):
      _ = c.get_int("non_existing")
    with self.assertRaisesRegex(
        ValueError,
        re.escape("Hyperparameter 'b' is expected to be a integer value"),
    ):
      _ = c.get_int("b")
    with self.assertRaisesRegex(
        ValueError,
        re.escape("Some hyperparameters have not been consumed: ['b', 'c']"),
    ):
      c.finalize()
    self.assertEqual(c.get_float("b"), 0.5)
    self.assertIsNone(c.get_optional_int("c"))
    c.finalize()
    with self.assertRaisesRegex(
        ValueError, re.escape("Hyperparameter consumer already finalized")
    ):
      c.finalize()


if __name__ == "__main__":
  absltest.main()
