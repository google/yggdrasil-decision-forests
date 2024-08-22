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


if __name__ == "__main__":
  absltest.main()
