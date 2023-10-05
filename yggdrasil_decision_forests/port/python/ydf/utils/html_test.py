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

"""Test html utilities."""

import textwrap

from absl.testing import absltest

from ydf.utils import html


class HtmlTest(absltest.TestCase):

  def test_base(self):
    doc, root = html.create_doc()
    root.appendChild(html.bold(doc, "Bold text"))
    root.appendChild(html.italic(doc, "Italic text"))

    div = doc.createElement("div")
    div.appendChild(doc.createTextNode("div content"))
    root.appendChild(html.bold(doc, div))

    link1 = html.link(doc, "url")
    link1.appendChild(doc.createTextNode("link"))
    root.appendChild(link1)

    self.assertEqual(
        root.toprettyxml(indent="  "),
        textwrap.dedent("""\
        <div>
          <span style="font-weight:bold">Bold text</span>
          <span style="font-style:italic">Italic text</span>
          <span style="font-weight:bold">
            <div>div content</div>
          </span>
          <a href="url" target="_blank">link</a>
        </div>
        """),
    )


if __name__ == "__main__":
  absltest.main()
