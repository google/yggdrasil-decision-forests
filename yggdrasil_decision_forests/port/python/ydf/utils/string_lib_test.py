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

"""Test string utilities."""

import textwrap

from absl.testing import absltest

from ydf.utils import string_lib


class StringTest(absltest.TestCase):

  def test_indent(self):
    self.assertEqual(string_lib.indent(""), "    ")
    self.assertEqual(string_lib.indent("hello"), "    hello")
    self.assertEqual(string_lib.indent("hello\n"), "    hello\n")
    self.assertEqual(string_lib.indent("hello\nworld"), "    hello\n    world")
    self.assertEqual(
        string_lib.indent("hello\nworld\n"), "    hello\n    world\n"
    )

  def test_table(self):
    self.assertEqual(
        string_lib.table(
            content=[["a", "b"], [5.12345678, 7.0]],
        ),
        textwrap.dedent("""\
        +---------+---------+
        |       a |       b |
        +---------+---------+
        | 5.12346 |       7 |
        +---------+---------+
        """),
    )

  def test_table_with_col(self):
    self.assertEqual(
        string_lib.table(
            content=[["a", "b"], [5.12345678, 7.0]],
            column_labels=["X", "Y"],
        ),
        textwrap.dedent("""\
        +---------+---------+
        |       X |       Y |
        +---------+---------+
        |       a |       b |
        +---------+---------+
        | 5.12346 |       7 |
        +---------+---------+
        """),
    )

  def test_table_with_col_no_data_row_separator(self):
    self.assertEqual(
        string_lib.table(
            content=[["a", "b"], [5.12345678, 7.0]],
            column_labels=["X", "Y"],
            data_row_separator=False,
        ),
        textwrap.dedent("""\
        +---------+---------+
        |       X |       Y |
        +---------+---------+
        |       a |       b |
        | 5.12346 |       7 |
        +---------+---------+
        """),
    )

  def test_table_squeezed_column(self):
    self.assertEqual(
        string_lib.table(
            content=[["a", "b"], [5.12345678, 7.0]],
            squeeze_column=True,
        ),
        textwrap.dedent("""\
        +---------+---+
        |       a | b |
        +---------+---+
        | 5.12346 | 7 |
        +---------+---+
        """),
    )

  def test_table_squeezed_column_and_no_data_row_separator(self):
    self.assertEqual(
        string_lib.table(
            content=[["a", "b"], [5.12345678, 7.0]],
            column_labels=["X", "Y"],
            data_row_separator=False,
            squeeze_column=True,
        ),
        textwrap.dedent("""\
        +---------+---+
        |       X | Y |
        +---------+---+
        |       a | b |
        | 5.12346 | 7 |
        +---------+---+
        """),
    )

  def test_table_with_row(self):
    self.assertEqual(
        string_lib.table(
            content=[["a", "b"], [5.12345678, 7.0]],
            row_labels=["A", "B"],
        ),
        textwrap.dedent("""\
        +---------+---------+---------+
        |       A |       a |       b |
        +---------+---------+---------+
        |       B | 5.12346 |       7 |
        +---------+---------+---------+
        """),
    )

  def test_table_with_col_and_row(self):
    self.assertEqual(
        string_lib.table(
            content=[["a", "b"], [5.12345678, 7.0]],
            column_labels=["X", "Y"],
            row_labels=["A", "B"],
        ),
        textwrap.dedent("""\
        +---------+---------+---------+
        |         |       X |       Y |
        +---------+---------+---------+
        |       A |       a |       b |
        +---------+---------+---------+
        |       B | 5.12346 |       7 |
        +---------+---------+---------+
        """),
    )

  def test_table_issue_empty_content(self):
    with self.assertRaisesRegex(ValueError, "Content is empty"):
      string_lib.table(content=[])

  def test_table_issue_empty_row(self):
    with self.assertRaisesRegex(ValueError, "Content is empty"):
      string_lib.table(content=[[]])

  def test_table_issue_inconsistent_content(self):
    with self.assertRaisesRegex(
        ValueError, "All rows should have the same number of values"
    ):
      string_lib.table(content=[[1], [2, 3]])

  def test_table_issue_inconsistent_col(self):
    with self.assertRaisesRegex(
        ValueError, "`column_labels` inconsistent with `content`"
    ):
      string_lib.table(content=[[1], [2]], column_labels=["A", "B"])

  def test_table_issue_inconsistent_row(self):
    with self.assertRaisesRegex(
        ValueError, "`row_labels` inconsistent with `content`"
    ):
      string_lib.table(content=[[1], [2]], row_labels=["A"])

  def test_table_issue_multiline(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot print table with multi-line values"
    ):
      string_lib.table(content=[["a\nb"]], row_labels=["A"])


if __name__ == "__main__":
  absltest.main()
