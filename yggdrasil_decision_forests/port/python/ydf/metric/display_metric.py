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

"""Utilities to display metrics."""

import base64
import io
import textwrap
from typing import Any, Optional, Tuple
from xml.dom import minidom

from ydf.cc import ydf
from ydf.metric import metric
from ydf.utils import documentation
from ydf.utils import html
from ydf.utils import string_lib


class _UnescappedString(minidom.Text):
  """Html element able to print unescaped string."""

  def set_data(self, data: str) -> None:
    """Sets content to print."""

    self._data = data

  def writexml(
      self,
      writer,
      indent="",
      addindent="",
      newl="",
      encoding=None,
      standalone=None,
  ) -> None:
    writer.write(self._data)


def css_style(doc: html.Doc) -> html.Elem:
  """Css style for metric display."""

  raw_style = textwrap.dedent("""\
  .metric_box {
  }

  .metric_box a {
    text-decoration:none;
    color: darkblue;
  }

  .metric_box .title {
    font-weight: bold;
  }

  .metric_box .section {
    margin: 5px 5px 5px 5px;
  }

  .metric_box .grid {
    display: grid;
    grid-template-columns: max-content max-content max-content max-content;
    grid-template-rows: auto;
    border-bottom: 1px solid lightgray;
    padding: 10px;
  }
    
  .metric_box .grid > div {
  }

  .metric_box .grid > div:nth-child(odd) {
    font-weight: bold;
    padding-right: 5px;
    text-align: right;
  }

  .metric_box .grid > div:nth-child(even) {
    padding-right: 20px;
  }

  .metric_box .complex {
    display: inline-block;
    margin: 15px 20px 0px 0px;
    vertical-align: top;
  }

  .metric_box .complex .key {
    font-weight: bold;
  }

  .metric_box .complex .value {
  }

  .metric_box .confusion_matrix {
    border-collapse: collapse;
    margin: 15px 15px;
    border: 1px solid lightgray;
  }

  .metric_box .confusion_matrix th {
    background-color: #ededed;
    font-weight: bold;
    text-align: left;
    padding: 5px;
    border: 1px solid lightgray;
  }

  .metric_box .confusion_matrix td {
    text-align: right;
    padding: 3px;
    border: 1px solid lightgray;
  }
  """)

  style = doc.createElement("style")
  raw_node = _UnescappedString()
  raw_node.set_data(raw_style)
  style.appendChild(raw_node)
  return style


def evaluation_to_str(e: metric.Evaluation) -> str:
  """String representation of an evaluation."""

  text = ""

  # Classification
  text += _field_to_str("accuracy", e.accuracy)
  text += _field_to_str("confusion matrix", e.confusion_matrix)
  if e.characteristics is not None:
    text += "characteristics:"
    for characteristic in e.characteristics:
      text += "\n" + string_lib.indent(str(characteristic))

  # Regression
  text += _field_to_str("RMSE", e.rmse)
  text += _field_to_str("RMSE 95% CI [B]", e.rmse_ci95_bootstrap)

  # Ranking
  text += _field_to_str("NDCG", e.ndcg)

  # Uplifting
  text += _field_to_str("QINI", e.qini)
  text += _field_to_str("AUUC", e.auuc)

  # Custom
  if e.custom_metrics:
    for k, v in e.custom_metrics.items():
      text += _field_to_str(k, v)

  # Generic
  text += _field_to_str("loss", e.loss)
  text += _field_to_str("num examples", e.num_examples)
  text += _field_to_str("num examples (weighted)", e.num_examples_weighted)

  if not text:
    text = "No metrics"

  return text


def evaluation_to_html_str(e: metric.Evaluation, add_style: bool = True) -> str:
  """Html representation of an evaluation."""

  doc, root = html.create_doc()

  if add_style:
    root.appendChild(css_style(doc))

  html_metric_box = doc.createElement("div")
  html_metric_box.setAttribute("class", "metric_box")
  root.appendChild(html_metric_box)

  html_metric_grid = doc.createElement("div")
  html_metric_grid.setAttribute("class", "grid section")
  html_metric_box.appendChild(html_metric_grid)

  # Metrics

  # Classification
  _field_to_html(doc, html_metric_grid, "accuracy", e.accuracy)

  if e.characteristics is not None:
    for characteristic in e.characteristics:
      _field_to_html(
          doc,
          html_metric_grid,
          "AUC: " + characteristic.name,
          characteristic.roc_auc,
      )

      _field_to_html(
          doc,
          html_metric_grid,
          "PR-AUC: " + characteristic.name,
          characteristic.pr_auc,
      )

  # Regression
  _field_to_html(doc, html_metric_grid, "RMSE", e.rmse)
  _field_to_html(
      doc, html_metric_grid, "RMSE 95% CI [B]", e.rmse_ci95_bootstrap
  )

  # Ranking
  _field_to_html(doc, html_metric_grid, "NDCG", e.ndcg)

  # Uplifting
  _field_to_html(doc, html_metric_grid, "QINI", e.qini)
  _field_to_html(doc, html_metric_grid, "AUUC", e.auuc)

  # Custom
  if e.custom_metrics:
    for k, v in e.custom_metrics.items():
      _field_to_html(doc, html_metric_grid, k, v)

  # Generic
  _field_to_html(doc, html_metric_grid, "loss", e.loss)
  _field_to_html(
      doc,
      html_metric_grid,
      "num examples",
      e.num_examples,
      documentation_url=documentation.URL_NUM_EXAMPLES,
  )
  _field_to_html(
      doc,
      html_metric_grid,
      "num examples (weighted)",
      e.num_examples_weighted,
      documentation_url=documentation.URL_WEIGHTED_NUM_EXAMPLES,
  )

  if e.confusion_matrix is not None:
    _object_to_html(
        doc,
        html_metric_box,
        "Confusion matrix",
        confusion_matrix_to_html_str(doc, e.confusion_matrix),
        documentation_url=documentation.URL_CONFUSION_MATRIX,
    )

  # Curves
  plot_html = ydf.EvaluationPlotToHtml(e._evaluation_proto)
  _object_to_html(doc, html_metric_box, None, plot_html, raw_html=True)

  return root.toprettyxml(indent="  ")


def confusion_matrix_to_html_str(
    doc: html.Doc, confusion: metric.ConfusionMatrix
) -> html.Elem:
  """Html representation of a confusion matrix."""

  html_table = doc.createElement("table")
  html_table.setAttribute("class", "confusion_matrix")

  # First line
  tr = doc.createElement("tr")
  html_table.appendChild(tr)

  th = doc.createElement("th")
  tr.appendChild(th)
  th.appendChild(doc.createTextNode("Label \\ Pred"))

  for label in confusion.classes:
    th = doc.createElement("th")
    tr.appendChild(th)
    th.appendChild(doc.createTextNode(label))

  for prediction_idx, prediction in enumerate(confusion.classes):
    tr = doc.createElement("tr")
    html_table.appendChild(tr)

    th = doc.createElement("th")
    tr.appendChild(th)
    th.appendChild(doc.createTextNode(prediction))

    for label_idx in range(len(confusion.classes)):
      value = confusion.value(
          label_idx=label_idx, prediction_idx=prediction_idx
      )
      td = doc.createElement("td")
      tr.appendChild(td)
      td.appendChild(doc.createTextNode(f"{value:g}"))

  return html_table


def _field_value_to_str(value: Any) -> Tuple[str, bool]:
  """Friendly text print a "value".

  Operations:
    - Remove decimal points in float e.g. 4.0 => "4".
    - Round digits to precision e.g 2.99999999 => "3
    - Remove any trailing line return.

  Args:
    value: The value to print.

  Returns:
    The string value, and a boolean indicating if the string value is
    multi-lines.
  """

  if value is None:
    return "", False

  if isinstance(value, float):
    if round(value) == value:
      # Remove decimale point
      str_value = str(int(value))
    else:
      # Compact print
      str_value = f"{value:g}"
  else:
    str_value = str(value)

  if "\n" not in str_value:
    return str_value, False

  # Indent the value from the key
  str_value = string_lib.indent(str_value)
  if str_value and str_value[-1] == "\n":
    # Remove final line return if any.
    str_value = str_value[:-1]
  return str_value, True


def _field_to_str(key: str, value: Any) -> str:
  """Friendly text print a "key:value".

  Operations:
    - Operations from "_field_value_to_str"

  Args:
    key: Name of the field.
    value: Value of the field.

  Returns:
    Formated key:value.
  """

  if value is None:
    return ""

  str_value, is_multi_lines = _field_value_to_str(value)

  if is_multi_lines:
    return f"{key}:\n{str_value}\n"
  else:
    return f"{key}: {str_value}\n"


def _field_to_html(
    doc: html.Doc,
    parent,
    key: str,
    value: Any,
    documentation_url: Optional[str] = None,
) -> None:
  """Friendly html print a "key:value".

  Operations:
    - All the operations of "_field_value_to_str"
    - Wrap multi-line values into <pre>.

  Args:
    doc: Html document.
    parent: Html element.
    key: Name of the field.
    value: Value of the field.
    documentation_url: Url to the documentation of this field.
  """

  if value is None:
    return

  str_value, is_multi_lines = _field_value_to_str(value)

  html_key = doc.createElement("div")
  parent.appendChild(html_key)
  html_key.setAttribute("class", "key")

  if documentation_url is not None:
    link = html.link(doc, documentation_url)
    html_key.appendChild(link)
    html_key = link

  html_key.appendChild(doc.createTextNode(key + ":"))

  html_value = doc.createElement("div")
  html_value.setAttribute("class", "value")
  parent.appendChild(html_value)

  if is_multi_lines:
    html_pre_value = doc.createElement("pre")
    html_value.appendChild(html_pre_value)
    html_pre_value.appendChild(doc.createTextNode(str_value))
  else:
    html_value.appendChild(doc.createTextNode(str_value))


def _object_to_html(
    doc: html.Doc,
    parent,
    key: Optional[str],
    value: Any,
    documentation_url: Optional[str] = None,
    raw_html: bool = False,
) -> None:
  """Friendly html print a "key" and a complex element.

  The complex element can be a multi-line string (or equivalent) or a Dom
  object.

  Args:
    doc: Html document.
    parent: Html element.
    key: Name of the field.
    value: Complex object to display.
    documentation_url: Url to the documentation of this field.
    raw_html: If true, "value" is interpreted as raw html.
  """

  if value is None:
    return

  str_value, _ = _field_value_to_str(value)

  html_container = doc.createElement("div")
  html_container.setAttribute("class", "complex")
  parent.appendChild(html_container)

  html_key = doc.createElement("div")
  html_container.appendChild(html_key)
  html_key.setAttribute("class", "key")

  if documentation_url is not None:
    link = html.link(doc, documentation_url)
    html_key.appendChild(link)
    html_key = link

  if key:
    html_key.appendChild(doc.createTextNode(key))

  html_value = doc.createElement("div")
  html_value.setAttribute("class", "value")
  html_container.appendChild(html_value)

  if isinstance(value, minidom.Element):
    html_value.appendChild(value)
  else:
    if raw_html:
      node = _RawXMLNode(value, doc)
      html_value.appendChild(node)
    else:
      html_pre_value = doc.createElement("pre")
      html_value.appendChild(html_pre_value)
      html_pre_value.appendChild(doc.createTextNode(str_value))


class _RawXMLNode(minidom.Node):
  # Required by Minidom
  nodeType = 1

  def __init__(self, data, parent):
    self.data = data
    self.ownerDocument = parent

  def writexml(self, writer, indent, addindent, newl):
    del indent
    del addindent
    del newl
    writer.write(self.data)
