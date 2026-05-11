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
import numpy as np
from numpy import testing as npt

from ydf.dataset.io import dataset_io


class DatasetIoTest(parameterized.TestCase):

  def test_unrolled_feature_names(self):
    self.assertEqual(
        dataset_io.unrolled_feature_names("a", 5),
        [
            "a.0_of_5",
            "a.1_of_5",
            "a.2_of_5",
            "a.3_of_5",
            "a.4_of_5",
        ],
    )

  def test_unrolled_feature_names_with_zero_dim(self):
    with self.assertRaisesRegex(ValueError, "should be strictly positive"):
      dataset_io.unrolled_feature_names("a", 0)

  @parameterized.parameters(
      ("a.3_of_5", ("a", 3, 5)),
      ("a.00_of_100", ("a", 0, 100)),
      (".3_of_5", ("", 3, 5)),
      (" .3_of_5", (" ", 3, 5)),
      ("123.123.3_of_5", ("123.123", 3, 5)),
  )
  def test_parse_unrolled_feature_name(self, name, expected):
    self.assertEqual(dataset_io.parse_unrolled_feature_name(name), expected)

  @parameterized.parameters(
      ("",),
      (" ",),
      ("a",),
      ("1234",),
      ("1_or_5",),
  )
  def test_parse_unrolled_feature_name_is_none(self, name):
    self.assertIsNone(dataset_io.parse_unrolled_feature_name(name))

  def test_unroll_column_lists_with_empty_strings(self):
    # All elements are lists of size 2, except one empty string
    src = np.array([[1.0, 2.0], "", [3.0, 4.0]], dtype=object)
    results = list(
        dataset_io._unroll_column(
            "feat", src, allow_unroll=True, expect_unroll_info=None
        )
    )
    self.assertLen(results, 2)

    # Expecting feat.0_of_2 with [1.0, np.nan, 3.0]
    self.assertEqual(results[0][0], "feat.0_of_2")
    npt.assert_equal(results[0][1], np.array([1.0, np.nan, 3.0]))
    self.assertTrue(results[0][2])

    # Expecting feat.1_of_2 with [2.0, np.nan, 4.0]
    self.assertEqual(results[1][0], "feat.1_of_2")
    npt.assert_equal(results[1][1], np.array([2.0, np.nan, 4.0]))
    self.assertTrue(results[1][2])

  def test_unroll_column_ndarrays_with_empty_strings(self):
    # All elements are ndarrays of size 2, except one empty string
    src = np.array(
        [np.array([1.0, 2.0]), "", np.array([3.0, 4.0])], dtype=object
    )
    results = list(
        dataset_io._unroll_column(
            "feat", src, allow_unroll=True, expect_unroll_info=None
        )
    )
    self.assertLen(results, 2)
    self.assertEqual(results[0][0], "feat.0_of_2")
    npt.assert_equal(results[0][1], np.array([1.0, np.nan, 3.0]))
    self.assertTrue(results[0][2])

  def test_unroll_column_unequal_lists_mixed_with_empty_strings(self):
    # Unequal length lists: one of size 2, one empty string, one of size 3
    src = np.array([[1.0, 2.0], "", [3.0, 4.0, 5.0]], dtype=object)
    results = list(
        dataset_io._unroll_column(
            "feat", src, allow_unroll=True, expect_unroll_info=None
        )
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0][0], "feat")
    self.assertIs(results[0][1], src)
    self.assertFalse(results[0][2])


if __name__ == "__main__":
  absltest.main()
