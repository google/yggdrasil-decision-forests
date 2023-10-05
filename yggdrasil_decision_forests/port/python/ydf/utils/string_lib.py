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

"""String utilities."""


from typing import Any, List, Optional, Sequence


def indent(text: str, num_spaces: int = 4) -> str:
  r"""Indents a possibly multi-line string.

  Example:

  ```python
    indent("Hello\nWorld\n")
    >> "    Hello\n    World\n"
  ```

  Args:
    text: String to indent.
    num_spaces: Number of spaces of indentation.

  Returns:
    Intended string.
  """

  # TODO: @gbm - Replace with "textwrap.indent".
  block = " " * num_spaces
  return block + block.join(text.splitlines(keepends=True))


def table(
    content: Sequence[Sequence[Any]],
    row_labels: Optional[Sequence[str]] = None,
    column_labels: Optional[Sequence[str]] = None,
) -> str:
  """Returns a string representation of the table.

  Example:

  ```python
    table(
      content=[["a", "b"],
              [5.12345678, 7.0]],
      column_labels=["X", "Y"],
      row_labels=["A", "B"],
      )

  >> +------------+------------+------------+
  >> |            |          X |          Y |
  >> +------------+------------+------------+
  >> |          A |          a |          b |
  >> +------------+------------+------------+
  >> |          B | 5.12345678 |          7 |
  >> +------------+------------+------------+
  ```

  Floating point without decimals are printed without the final dot. For
  example 5.0 is printed as 5.

  The table cannot be empty i.e., there should be at least one row and one
  column.

  The string representation of table elements should not include line breaks.

  Args:
    content: Content of the table. `content[i][j]` is the cell value at the i-th
      row and j-th column.
    row_labels: Row names. If set, `content` should have `len(row_labels)` rows.
    column_labels: Column names. If set, `content` should have
      `len(column_labels)` columns.
  """

  if not content or not content[0]:
    raise ValueError("Content is empty")

  num_columns = len(content[0])
  num_rows = len(content)

  if any(num_columns != len(row) for row in content):
    raise ValueError("All rows should have the same number of values")

  def format_cell(cell: Any) -> str:
    if isinstance(cell, float):
      if round(cell) == cell:
        # Print 7.0 as "7" instead of "7.0".
        str_cell = str(int(cell))
      else:
        str_cell = f"{cell:g}"
    else:
      str_cell = str(cell)

    if "\n" in str_cell:
      raise ValueError(f"Cannot print table with multi-line values: {str_cell}")
    return str_cell

  # Enforce preconditions (all rows with the same size, rows and column labels
  # with the expected sizes, no newlines, etc.)
  # From here on out, there's no more error-checking logic.

  str_content = []
  if column_labels is not None:
    if len(column_labels) != num_columns:
      raise ValueError("`column_labels` inconsistent with `content`")
    row_label = [""] if row_labels is not None else []
    str_content.append(row_label + list(column_labels))

  if row_labels is not None:
    if len(row_labels) != num_rows:
      raise ValueError("`row_labels` inconsistent with `content`")
    num_columns += 1

  for row_idx, row in enumerate(content):
    row_label = [row_labels[row_idx]] if row_labels is not None else []
    str_content.append(row_label + [format_cell(cell) for cell in row])

  max_length = 0
  for col_idx in range(num_columns):
    max_length = max(max_length, max(len(row[col_idx]) for row in str_content))

  # One space before and after each cell content.
  cell_length = max_length + 2

  vertical_separator = "|"
  horizontal_separator = "-"
  dot_separator = "+"

  row_separator = (
      (dot_separator + horizontal_separator * cell_length) * num_columns
      + dot_separator
      + "\n"
  )

  output = [row_separator]
  for row in str_content:
    for col in row:
      output.append(f"{vertical_separator}{col.rjust(cell_length - 1)} ")
    output.append(f"{vertical_separator}\n{row_separator}")
  return "".join(output)
