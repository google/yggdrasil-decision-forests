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

from ydf.utils import paths


class PathsTest(absltest.TestCase):

  def test_normalize_list_of_paths_all_paths_typed(self):
    all_paths_typed = ["csv:/foo/bar", "csv:bar/foo", "csv:asdf"]
    self.assertEqual(
        paths.normalize_list_of_paths(all_paths_typed),
        "csv:/foo/bar,bar/foo,asdf",
    )

  def test_normalize_list_of_paths_fails_with_different_types(self):
    all_paths_typed_different_types = ["csv:/foo/bar", "prefix:/bar/foo"]
    with self.assertRaises(ValueError):
      _ = paths.normalize_list_of_paths(all_paths_typed_different_types)

  def test_normalize_list_of_paths_fails_with_some_missing_types(self):
    not_all_paths_typed = ["csv:/foo/bar", "bar.txt"]
    with self.assertRaises(ValueError):
      _ = paths.normalize_list_of_paths(not_all_paths_typed)

  def test_normalize_list_of_paths_fails_with_all_missing_types(self):
    not_all_paths_typed = ["/foo/bar", "bar.txt"]
    with self.assertRaises(ValueError):
      _ = paths.normalize_list_of_paths(not_all_paths_typed)

  def test_normalize_list_of_paths_fails_with_empty_list(self):
    empty_list = []
    with self.assertRaises(ValueError):
      paths.normalize_list_of_paths(empty_list)


if __name__ == "__main__":
  absltest.main()
