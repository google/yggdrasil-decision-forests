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

import dataclasses
from absl.testing import absltest
import numpy as np
from ydf.utils import test_utils


@dataclasses.dataclass(frozen=True)
class A:
  a: str
  b: int
  c: np.ndarray


class TestUtilsTest(absltest.TestCase):

  def test_test_almost_equal_true(self):
    self.assertIsNone(test_utils.test_almost_equal(1, 1))
    self.assertIsNone(test_utils.test_almost_equal("a", "a"))
    self.assertIsNone(test_utils.test_almost_equal(True, True))
    self.assertIsNone(test_utils.test_almost_equal([], []))
    self.assertIsNone(test_utils.test_almost_equal([1, 2], [1, 2]))
    self.assertIsNone(
        test_utils.test_almost_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
    )
    self.assertIsNone(
        test_utils.test_almost_equal(np.array(["a", "b"]), np.array(["a", "b"]))
    )
    self.assertIsNone(test_utils.test_almost_equal({1: {1: 2}}, {1: {1: 2}}))
    self.assertIsNone(
        test_utils.test_almost_equal(
            A("a", 1, np.array([1, 2])), A("a", 1, np.array([1, 2]))
        )
    )

  def test_test_almost_equal_false(self):
    self.assertIsNotNone(test_utils.test_almost_equal(1, 2))
    self.assertIsNotNone(test_utils.test_almost_equal("a", "b"))
    self.assertIsNotNone(test_utils.test_almost_equal("a", 1))
    self.assertIsNotNone(test_utils.test_almost_equal(True, False))
    self.assertIsNotNone(test_utils.test_almost_equal([], "a"))
    self.assertIsNotNone(test_utils.test_almost_equal([1, 2], [2, 2]))
    self.assertIsNotNone(
        test_utils.test_almost_equal(np.array([1, 2, 3]), np.array([1, 2, 4]))
    )
    self.assertIsNotNone(
        test_utils.test_almost_equal(np.array([1, 2, 3]), np.array(["a", "b"]))
    )
    self.assertIsNotNone(
        test_utils.test_almost_equal(np.array(["a", "b"]), np.array(["a", "c"]))
    )
    self.assertIsNotNone(test_utils.test_almost_equal({1: {1: 2}}, {1: {5: 2}}))
    self.assertIsNotNone(
        test_utils.test_almost_equal(
            A("a", 1, np.array([1, 2])), A("a", 1, np.array([1, 3]))
        )
    )
    self.assertIsNotNone(
        test_utils.test_almost_equal(
            A("a", 1, np.array([1, 2])), A("b", 1, np.array([1, 2]))
        )
    )
    self.assertIsNotNone(
        test_utils.test_almost_equal(
            A("a", 1, np.array([1, 2])), A("b", 2, np.array([1, 2]))
        )
    )


if __name__ == "__main__":
  absltest.main()
