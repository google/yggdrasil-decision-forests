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

"""Html utilities."""

from typing import Any, Dict, Tuple, Union
from xml.dom import minidom

Doc = minidom.Document
Elem = minidom.Element


class HtmlNotebookDisplay:
  """An object printed as html in a notebook."""

  def __init__(self, html: str):
    self._html = html

  def _repr_html_(self) -> str:
    return self._html

  def _repr_(self) -> str:
    return self._html


def create_doc() -> Tuple[Doc, Elem]:
  """Creates a html document.

  Example:

  ```python
  doc, root = create_doc()
  root.appendChild(bold(doc, "Some text"))
  print(root.toxml())
  ```

  Returns:
    A html document and its root node.
  """

  impl = minidom.getDOMImplementation()
  assert impl is not None
  doc = impl.createDocument(None, "div", None)
  return doc, doc.documentElement


def with_style(doc: Doc, item: Union[Elem, str], style: Dict[str, Any]) -> Elem:
  """Creates html element with given style.

  Example:

  ```python
  doc, root = create_doc()
  root.appendChild(with_style(doc, "My text", {"font-weight": "bold"}))
  print(root.toxml())
  ```

  Args:
    doc: Html document.
    item: Item to display. Can be a string or another Html element.
    style: style.

  Returns:
    Html element.
  """

  if isinstance(item, minidom.Element):
    raw_item = item
    item = doc.createElement("span")
    item.appendChild(raw_item)
  elif isinstance(item, str):
    raw_item = item
    item = doc.createElement("span")
    item.appendChild(doc.createTextNode(raw_item))
  else:
    raise ValueError(f"Non supported element {item}")

  style_key = "style"

  # Get existing style, if any.
  style_text = (
      item.getAttribute(style_key) + "; "
      if item.hasAttribute(style_key)
      else ""
  )

  # Add new style
  style_text += "; ".join([f"{k}:{v}" for k, v in style.items()])
  item.setAttribute("style", style_text)
  return item


def bold(doc: Doc, item: Union[Elem, str]) -> Elem:
  return with_style(doc, item, {"font-weight": "bold"})


def italic(doc: Doc, item: Union[Elem, str]) -> Elem:
  return with_style(doc, item, {"font-style": "italic"})


def link(doc: Doc, url: str) -> Elem:
  """Creates a link (i.e., <a>) element."""

  html_a = doc.createElement("a")
  html_a.setAttribute("href", url)
  html_a.setAttribute("target", "_blank")
  return html_a
