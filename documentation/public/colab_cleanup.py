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

"""MkDocs hook removing Colab-only markup from the rendered notebooks.

Notebooks saved from Colab contain, next to each DataFrame output, a
"Convert to interactive table" button with a script calling the Colab API
(`google.colab...`). Outside of Colab, the buttons are non-functional and the
scripts raise JavaScript errors.
"""

import re

_BUTTONS_START = re.compile(r'<div class="colab-df-buttons">')
_DIV_TAG = re.compile(r"<div\b|</div>")
_COLAB_SCRIPT = re.compile(
    r"<script\b[^>]*>(?:(?!</script>).)*?google\.colab.*?</script>", re.S
)


def _remove_colab_df_buttons(html: str) -> str:
  """Removes the `<div class="colab-df-buttons">` elements."""
  while match := _BUTTONS_START.search(html):
    depth = 0
    for tag in _DIV_TAG.finditer(html, match.start()):
      depth += 1 if tag.group(0) == "<div" else -1
      if depth == 0:
        html = html[: match.start()] + html[tag.end() :]
        break
    else:
      break  # Unbalanced HTML; leave the rest untouched.
  return html


def on_post_page(output: str, page, config) -> str:
  del config  # Unused.
  if not page.file.src_uri.endswith(".ipynb"):
    return output
  output = _remove_colab_df_buttons(output)
  return _COLAB_SCRIPT.sub("", output)
