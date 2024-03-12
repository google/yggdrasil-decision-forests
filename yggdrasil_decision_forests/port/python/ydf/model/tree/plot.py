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

"""Utilities for plotting a tree."""

import dataclasses
import json
import string
from typing import Any, Dict, Optional, Sequence
import uuid


@dataclasses.dataclass(frozen=True)
class PlotOptions:
  """Options for plotting trees.

  All the values are expressed in pixel.
  """

  # Margin around the entire plot.
  margin: Optional[int] = 10

  # Size of a tree node.
  node_x_size: Optional[int] = 160
  node_y_size: Optional[int] = 12 * 2 + 4

  # Space between tree nodes.
  node_x_offset: Optional[int] = 160 + 20
  node_y_offset: Optional[int] = 12 * 2 + 4 + 5

  # Text size in px.
  font_size: Optional[int] = 10

  # Rounding effect of the edges.
  # This value is the distance (in pixel) of the Bezier control anchor from
  # the source point.
  edge_rounding: Optional[int] = 20

  # Padding inside nodes.
  node_padding: Optional[int] = 2

  # Show a bb box around the plot. For debugging only.
  show_plot_bounding_box: Optional[bool] = False


class TreePlot:
  """Class for plotting a tree in IPython / Colab and as string."""

  def __init__(
      self,
      tree_json: Dict[str, Any],
      label_classes: Optional[Sequence[str]],
      options: PlotOptions,
      d3js_url: str,
  ):
    """Initializes the instance based on the tree and its options.

    Args:
      tree_json: A JSON-serializable dictionary of the tree.
      label_classes: For classification problems, the names of the label
        classes, None otherwise.
      options: Options for the visual presentation of the plot.
      d3js_url: The URL to load d3.js from.
    """
    self._tree_json = tree_json
    self._d3js_url = d3js_url
    self._options = dataclasses.asdict(options)
    if label_classes is not None:
      self._options["labels"] = label_classes

  def __str__(self) -> str:
    """Returns an explanation how to display the plot."""
    return (
        "A plot of a tree. Use a notebook cell to display the plot."
        " Alternatively, export the plot with"
        ' `plot.to_file("plot.html")` or print the html source with'
        " `print(plot.html())`."
    )

  def html(self) -> str:
    """Returns HTML plot of the tree."""
    return self._repr_html_()

  def _repr_html_(self) -> str:
    """Returns HTML plot of the tree."""
    # Plotting library.
    import pkgutil

    plotter_js = pkgutil.get_data(__name__, "plotter.js").decode()

    container_id = "tree_plot_" + uuid.uuid4().hex

    html_content = string.Template("""
  <script src='${d3js_url}'></script>
  <div id="${container_id}"></div>
  <script>
  ${plotter_js}
  display_tree(${options}, ${json_tree_content}, "#${container_id}")
  </script>
  """).substitute(
        d3js_url=self._d3js_url,
        options=json.dumps(self._options),
        plotter_js=plotter_js,
        container_id=container_id,
        json_tree_content=json.dumps(self._tree_json),
    )
    return html_content

  def to_file(self, path: str) -> None:
    """Exports the plot to a html file."""
    with open(path, "w") as f:
      f.write(self.html())
