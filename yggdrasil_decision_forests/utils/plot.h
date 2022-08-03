/*
 * Copyright 2021 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Plot is a library to create simple plots, and export them in other plotting
// libraries (currently, only plotly).
//
// This library does not aim to be a complete plotting library. Instead, it
// should only implement what is needed by the yggdrasil library. For a more
// complete library, use SimplePlot (go/simpleplot).
//
// Usage example:
//
// Plot plot;
// plot.title = "the title of my plot";
//
// // Create a new curve.
// auto curve = absl::make_unique<Curve>();
// curve->label = "curve 1";
// curve->xs = {1, 2, 3};
// curve->ys = {2, 0.5, 4};
// plot.items.push_back(std::move(curve));
//
// // Export to html.
// ASSIGN_OR_RETURN(const auto html_plot, ExportToHtml(plot));
// RETURN_IF_ERROR(file::SetContent(path, html_plot));
//
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_PLOT_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_PLOT_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/histogram.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace plot {

namespace internal {

// Function related to the export to the plotly library.
namespace plotly {

// Plot in construction.
struct ExportAccumulator {
  // The name of those field match the plotly fields.
  std::string data;
};

}  // namespace plotly
}  // namespace internal

// What library to use to export the plots.
enum class TargetLibrary {
  kPlotly,  // See plotly.com/javascript
};

// Configuration to export the plot to html.
struct ExportOptions {
  // Set to true iff. this is the first plot to export in this html.
  bool first_export = true;

  // If true, check that the plot is correct before the export.
  bool run_checks = true;

  // The library to use to export the plot.
  TargetLibrary target_library = TargetLibrary::kPlotly;

  // Dimension of the plot in px.
  int width = 600;
  int height = 400;
};

// A plot item is the base class for plot content.
struct PlotItem {
  virtual absl::Status Check() const = 0;
  virtual ~PlotItem() = default;
};

// Line drawing style.
enum class LineStyle { SOLID, DOTTED };

// A curve is a set of connected points.
struct Curve : PlotItem {
  absl::Status Check() const override;

  // Display label. Used in the legend.
  // Leave empty for default label.
  std::string label;

  // List of y values.
  std::vector<double> ys;

  // List of x values. If empty, the xs are taken in [0, ys.length). If not
  // empty, should be of the same size as "ys".
  std::vector<double> xs;

  LineStyle style = LineStyle::SOLID;
};

// A curve is a set of connected points.
struct Bars : PlotItem {
  absl::Status Check() const override;

  absl::Status FromHistogram(const utils::histogram::Histogram<float>& hist);

  // Display label. Used in the legend.
  // Leave empty for default label.
  std::string label;

  std::vector<double> heights;
  std::vector<double> centers;
};

// An axis of the plot.
struct Axis {
  // Label of the axis. Leave empty for no label.
  std::string label;
};

// A set of geometrical object in a 2d space.
struct Plot {
  absl::Status Check() const;

  // Title of the plot. Leave empty for no title.
  std::string title;

  // Id of the chart in the html page dom.
  // If empty, generates a random id.
  std::string chart_id;

  // The items of the plot.
  std::vector<std::unique_ptr<PlotItem>> items;

  // The x (horizontal) and y (vertical) axis.
  Axis x_axis;
  Axis y_axis;

  // Show the plot legend.
  bool show_legend = true;
};

struct MultiPlotItem {
  Plot plot;

  // The following field are the coordinate of the plot inside of the
  // multi-plot. The coordinates of a MultiPlotItem should be contained in the
  // MultiPlot.

  // Column coordinate of the plot.
  int col = 0;
  // Row coordinate of the plot.
  int row = 0;
  // Width of the plot.
  int num_cols = 1;
  // Height of the plot.
  int num_rows = 1;
};

// A set of plots.
struct MultiPlot {
  absl::Status Check() const;

  std::vector<MultiPlotItem> items;

  // Size of the multi-plot.
  int num_cols = 1;
  int num_rows = 1;
};

// Exports a multi-plot to a html file.
utils::StatusOr<std::string> ExportToHtml(const MultiPlot& multiplot,
                                          const ExportOptions& options = {});

// Exports a plot to a html file.
utils::StatusOr<std::string> ExportToHtml(const Plot& plot,
                                          const ExportOptions& options = {});

namespace internal {
namespace plotly {

// Specialization of ExportToHtml to plotly.
utils::StatusOr<std::string> ExportToHtml(const Plot& plot,
                                          const ExportOptions& options);

}  // namespace plotly
}  // namespace internal

}  // namespace plot
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_PLOT_H_
