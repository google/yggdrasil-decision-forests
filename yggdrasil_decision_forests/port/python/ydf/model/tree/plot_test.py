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

import os

from absl.testing import absltest

from ydf.model.tree import plot as plot_lib
from ydf.utils import test_utils


def plotter_js_path() -> str:
  return os.path.join(test_utils.data_root_path(), "ydf/model/tree/plotter.js")


class PlotTest(absltest.TestCase):

  def test_plotter_js_included(self):
    tree_plot = plot_lib.TreePlot(
        {}, None, plot_lib.PlotOptions(), "https://google.com"
    )
    plotter_js_data = open(plotter_js_path(), "r").read()
    self.assertIn(plotter_js_data, tree_plot.html())

  def test_url(self):
    tree_plot = plot_lib.TreePlot(
        {}, None, plot_lib.PlotOptions(), "https://google.com/bla/d3js.js"
    )
    self.assertIn(
        "https://google.com/bla/d3js.js",
        tree_plot.html(),
    )

  def test_options(self):
    options = plot_lib.PlotOptions(margin=12345, show_plot_bounding_box=True)
    tree_plot = plot_lib.TreePlot({}, None, options, "https://google.com")
    self.assertIn(
        '"margin": 12345',
        tree_plot.html(),
    )
    self.assertIn(
        '"show_plot_bounding_box": true',
        tree_plot.html(),
    )

  def test_label_classes(self):
    tree_plot = plot_lib.TreePlot(
        {},
        ["class_1", "class_2", "class_3"],
        plot_lib.PlotOptions(),
        "https://google.com",
    )
    self.assertIn(
        '"labels": ["class_1", "class_2", "class_3"]',
        tree_plot.html(),
    )

  def test_tree(self):
    tree_plot = plot_lib.TreePlot(
        {"value": 654321},
        None,
        plot_lib.PlotOptions(),
        "https://google.com",
    )
    self.assertIn(
        '{"value": 654321}',
        tree_plot.html(),
    )


if __name__ == "__main__":
  absltest.main()
