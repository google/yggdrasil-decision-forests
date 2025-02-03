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

import math
import textwrap
from typing import Any, Optional, Tuple
import uuid
from xml.dom import minidom

from yggdrasil_decision_forests.model import abstract_model_pb2
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


def js_functionality(doc: html.Doc) -> html.Elem:
  """Javascript functions for metric display."""
  raw_js = textwrap.dedent("""\
  function ydfShowTab(block_id, item) {
    const block = document.getElementById(block_id);
    console.log("HIDE first of:",block.getElementsByClassName("tab selected"));
    console.log("HIDE first of:",block.getElementsByClassName("tab_content selected"));
    block.getElementsByClassName("tab selected")[0].classList.remove("selected");
    block.getElementsByClassName("tab_content selected")[0].classList.remove("selected");
    document.getElementById(block_id + "_" + item).classList.add("selected");
    document.getElementById(block_id + "_body_" + item).classList.add("selected");
  }
  """)

  js = doc.createElement("script")
  raw_node = _UnescappedString()
  raw_node.set_data(raw_js)
  js.appendChild(raw_node)
  return js


def glossary(doc: html.Doc, e: metric.Evaluation) -> html.Elem:
  """Glossary for evaluations."""

  raw_glossary = _UnescappedString()
  if e._evaluation_proto.task == abstract_model_pb2.CLASSIFICATION:
    raw_glossary.set_data(textwrap.dedent("""\
<h2>Evaluation of classification models</h2>
<dl>
  <dt><b>Accuracy</b></dt>
  <dd>The simplest metric. It's the percentage of predictions that are correct (matching the ground truth).
      <br><i>Example:</i> If a model correctly identifies 90 out of 100 images as cat or dog, the accuracy is 90%.</dd>

  <dt><b>Confusion Matrix</b></dt>
  <dd>A table that shows the counts of:
    <ul>
      <li><b>True Positives (TP):</b> Model correctly predicted positive.</li>
      <li><b>True Negatives (TN):</b> Model correctly predicted negative.</li>
      <li><b>False Positives (FP):</b> Model incorrectly predicted positive (a "false alarm").</li>
      <li><b>False Negatives (FN):</b> Model incorrectly predicted negative (a "miss").</li>
    </ul>
  </dd>

  <dt><b>Threshold</b></dt>
  <dd>YDF classification models predict a probability for each class. A threshold determines the cutoff for classifying something as positive or negative.
    <br><i>Example:</i> If the threshold is 0.5, any prediction above 0.5 might be classified as "spam," and anything below as "not spam."
  </dd>

  <dt><b>ROC Curve (Receiver Operating Characteristic Curve)</b></dt>
  <dd>A graph that plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various thresholds.
    <ul>
      <li><b>TPR (Sensitivity or Recall):</b> TP / (TP + FN) - How many of the actual positives did the model catch?</li>
      <li><b>FPR:</b> FP / (FP + TN) - How many negatives were incorrectly classified as positives?</li>
    </ul>
    <br><i>Interpretation:</i> A good model has an ROC curve that hugs the top-left corner (high TPR, low FPR).
  </dd>

  <dt><b>AUC (Area Under the ROC Curve)</b></dt>
  <dd>A single number that summarizes the overall performance shown by the ROC curve. The AUC is a more stable metric than the accuracy. Multi-class classification models evaluate one class against all other classes.
    <br><i>Interpretation:</i> Ranges from 0 to 1. A perfect model has an AUC of 1, while a random model has an AUC of 0.5. Higher is better.
  </dd>

  <dt><b>Precision-Recall Curve</b></dt>
  <dd>A graph that plots Precision against Recall at various thresholds.
    <ul>
      <li><b>Precision:</b> TP / (TP + FP) - Out of all the predictions the model labeled as positive, how many were actually positive?</li>
      <li><b>Recall (same as TPR):</b> TP / (TP + FN) - Out of all the actual positive cases, how many did the model correctly identify?</li>
    </ul>
    <br><i>Interpretation:</i> A good model has a curve that stays high (both high precision and high recall). It is especially useful when dealing with imbalanced datasets (e.g., when one class is much rarer than the other).
  </dd>

  <dt><b>PR-AUC (Area Under the Precision-Recall Curve)</b></dt>
  <dd>Similar to AUC, but for the Precision-Recall curve. A single number summarizing performance. Multi-class classification models evaluate one class against all other classes. Higher is better.</dd>

  <dt><b>Threshold / Accuracy Curve</b></dt>
  <dd>A graph that shows how the model's accuracy changes as you vary the classification threshold.</dd>

  <dt><b>Threshold / Volume Curve</b></dt>
  <dd>A graph showing how the number of data points classified as positive changes as you vary the threshold.</dd>
</dl>
    """))
  elif e._evaluation_proto.task == abstract_model_pb2.REGRESSION:
    raw_glossary.set_data(textwrap.dedent("""\
<h2>Evaluation of regression models</h2>
<dl>
  <dt><b>RMSE (Root Mean Squared Error)</b></dt>
  <dd>The square root of the average squared difference between predictions and ground truth values.
    <br><i>Interpretation:</i> Lower RMSE is better. It has the same units as the target variable, making it somewhat interpretable.
  </dd>

  <dt><b>Residual</b></dt>
  <dd>The difference between a prediction and the ground truth per example (Prediction - Ground Truth).</dd>

  <dt><b>Residual Histogram</b></dt>
  <dd>A histogram showing the distribution of the residuals.
    <br><i>Interpretation:</i> Ideally, you want a roughly symmetrical, bell-shaped distribution centered around zero, indicating that the errors are random and not biased.
  </dd>

  <dt><b>Ground Truth Histogram</b></dt>
  <dd>A histogram showing the distribution of the actual target values in your dataset.</dd>

  <dt><b>Prediction Histogram</b></dt>
  <dd>A histogram showing the distribution of the model's predictions.</dd>

  <dt><b>Ground Truth vs Predictions Curve</b></dt>
  <dd>A scatter plot where each point represents a data point. The x-axis is the ground truth value, and the y-axis is the model's prediction.
    <br><i>Interpretation:</i> A perfect model would have all points falling on a diagonal line (where prediction = ground truth). Deviations from this line show errors.
  </dd>

  <dt><b>Predictions vs Residual Curve</b></dt>
  <dd>A scatter plot where the x-axis is the model's prediction, and the y-axis is the residual.
    <br><i>Interpretation:</i> Ideally, you want to see a random scatter of points around the horizontal line at zero. Patterns (e.g., a funnel shape) might indicate problems with the model.
  </dd>

  <dt><b>Predictions vs Ground Truth Curve</b></dt>
  <dd>Sometimes this will plot a fitted curve through the points on the Ground Truth vs Predictions scatter plot to visualize the trend. It can help to see if the model is systematically over- or under-predicting in certain ranges.</dd>
</dl>
    """))
  else:
    raise ValueError(f"Unsupported evaluation task: {e._evaluation_proto.task}")
  glossary_node = doc.createElement("div")
  glossary_node.appendChild(raw_glossary)
  return glossary_node


def css_style(doc: html.Doc, add_glossary_style: bool) -> html.Elem:
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

  if add_glossary_style:
    raw_style += textwrap.dedent("""\

  .tab_block .header {
      flex-direction: row;
      display: flex;
  }

  .tab_block .header .tab {
      cursor: pointer;
      background-color: #F6F5F5;
      text-decoration: none;
      text-align: center;
      padding: 4px 12px;
      color: black;
  }

  .tab_block .header .tab.selected {
      border-bottom: 2px solid #2F80ED;
  }

  .tab_block .header .tab:hover {
      text-decoration: none;
      background-color: #DCDCDC;
  }

  .tab_block .body .tab_content {
      display: none;
      padding: 5px;
  }

  .tab_block .body .tab_content.selected {
      display: block;
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
  text += _field_to_str("MRR", e.mrr)

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
  show_glossary = e._evaluation_proto.task in [
      abstract_model_pb2.CLASSIFICATION,
      abstract_model_pb2.REGRESSION,
  ]

  doc, root = html.create_doc()
  effective_root = root

  if add_style:
    effective_root.appendChild(css_style(doc, show_glossary))
  if show_glossary:
    effective_root.appendChild(js_functionality(doc))
    tab_id = str(uuid.uuid4())
    tab_block = doc.createElement("div")
    tab_block.setAttribute("id", tab_id)
    tab_block.setIdAttribute("id")
    tab_block.setAttribute("class", "tab_block")
    tab_header = doc.createElement("div")
    tab_header.setAttribute("class", "header")

    eval_link = doc.createElement("a")
    eval_link.setAttribute("id", f"{tab_id}_eval")
    eval_link.setIdAttribute("id")
    eval_link.setAttribute("class", "tab selected")
    eval_link.setAttribute("onclick", f"ydfShowTab('{tab_id}', 'eval')")
    eval_link.appendChild(doc.createTextNode("Evalution"))

    glossary_link = doc.createElement("a")
    glossary_link.setAttribute("id", f"{tab_id}_glossary")
    glossary_link.setIdAttribute("id")
    glossary_link.setAttribute("class", "tab")
    glossary_link.setAttribute("onclick", f"ydfShowTab('{tab_id}', 'glossary')")
    glossary_link.appendChild(doc.createTextNode("Glossary"))

    tab_header.appendChild(eval_link)
    tab_header.appendChild(glossary_link)
    tab_block.appendChild(tab_header)

    tab_body = doc.createElement("div")
    tab_body.setAttribute("class", "body")
    tab_block.appendChild(tab_body)

    eval_content = doc.createElement("div")
    eval_content.setAttribute("id", f"{tab_id}_body_eval")
    eval_content.setIdAttribute("id")
    eval_content.setAttribute("class", "tab_content selected")

    glossary_content = doc.createElement("div")
    glossary_content.setAttribute("id", f"{tab_id}_body_glossary")
    glossary_content.setIdAttribute("id")
    glossary_content.setAttribute("class", "tab_content")
    glossary_content.appendChild(glossary(doc, e))

    tab_body.appendChild(glossary_content)
    tab_body.appendChild(eval_content)

    root.appendChild(tab_block)
    effective_root = eval_content

  html_metric_box = doc.createElement("div")
  html_metric_box.setAttribute("class", "metric_box")
  effective_root.appendChild(html_metric_box)

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
  _field_to_html(doc, html_metric_grid, "MRR", e.mrr)

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
    if not math.isnan(value) and round(value) == value:
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
