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

"""Test of the filesystem wrappers."""

from absl.testing import parameterized

from ydf.utils import filesystem


class FilesystemTest(parameterized.TestCase):

  def test_create_file_and_check_existance(self):
    tmpdir = self.create_tempdir("files")
    file_path = filesystem.Path(tmpdir) / "filename"
    file_path.touch()
    self.assertTrue(file_path.exists())

  def test_write_and_read_file(self):
    tmpdir = self.create_tempdir("files")
    file_path = filesystem.Path(tmpdir) / "filename"
    with filesystem.Open(file_path, "w") as f:
      f.write("foobar")
    with filesystem.Open(file_path, "r") as f:
      file_contents = f.read()
    self.assertEqual(file_contents, "foobar")
