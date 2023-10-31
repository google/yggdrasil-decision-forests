/*
 * Copyright 2022 Google LLC.
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
#include "absl/status/statusor.h"
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

  // Show a menu in the plot. Disable it when creating small plots.
  bool show_interactive_menu = false;

  // Prefix added to the id of the generated html elements. If not set,
  // generates ids randomly.
  std::string html_id_prefix;
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

// The scale of a numerical axis.
enum class AxisScale {
  UNIFORM,
  LOG,
};

// An axis of the plot.
struct Axis {
  // Label of the axis. Leave empty for no label.
  std::string label;

  // Scale of the axis.
  AxisScale scale = AxisScale::UNIFORM;

  // List of tick values. If not set, ticks are set automatically.
  absl::optional<std::vector<double>> manual_tick_values;

  // List of tick texts. If not set, "manual_tick_values" should also be set and
  // both "manual_tick_values" and "manual_tick_texts" should have the same
  // number of examples.
  absl::optional<std::vector<std::string>> manual_tick_texts;
};

// A set of geometrical object in a 2d space.
struct Plot {
  absl::Status Check() const;

  Curve* AddCurve() {
    auto curve = absl::make_unique<Curve>();
    auto* raw_curve = curve.get();
    items.push_back(std::move(curve));
    return raw_curve;
  }

  Bars* AddBars() {
    auto bars = absl::make_unique<Bars>();
    auto* raw_bars = bars.get();
    items.push_back(std::move(bars));
    return raw_bars;
  }

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
  MultiPlotItem() = default;
  MultiPlotItem(Plot&& _plot, int _col = 0, int _row = 0, int _num_cols = 1,
                int _num_rows = 1)
      : plot(std::move(_plot)),
        col(_col),
        row(_row),
        num_cols(_num_cols),
        num_rows(_num_rows) {}

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

  MultiPlotItem* AddItem() {
    items.push_back(absl::make_unique<MultiPlotItem>());
    return items.back().get();
  }

  std::vector<std::unique_ptr<MultiPlotItem>> items;

  // Size of the multi-plot.
  int num_cols = 1;
  int num_rows = 1;
};

// Exports a multi-plot to a html file.
absl::StatusOr<std::string> ExportToHtml(const MultiPlot& multiplot,
                                         const ExportOptions& options = {});

// Exports a plot to a html file.
absl::StatusOr<std::string> ExportToHtml(const Plot& plot,
                                         const ExportOptions& options = {});

// Utility to place plots in a multi-plot. This class sets automatically all the
// num_{cols,rows}, col and row fields in a multiplot and its sub-plots. The
// plots are organized in a column-minor, row-major way.
//
// Usage example:
//
//   MultiPlot multiplot;
//   ASSIGN_OR_RETURN(auto placer, PlotPlacer::Create(4, 2, &multiplot));
//   ASSIGN_OR_RETURN(auto* plot_1, placer.NewPlot());
//   ASSIGN_OR_RETURN(auto* plot_2, placer.NewPlot());
//   ASSIGN_OR_RETURN(auto* plot_3, placer.NewPlot());
//   ASSIGN_OR_RETURN(auto* plot_4, placer.NewPlot());
//   RETURN_IF_ERROR(placer.Finalize());
//
class PlotPlacer {
 public:
  // Create a placer.
  //
  // Args:
  //  multiplot: A non-owning pointer to the multi-plot that will contain the
  //    plots. The multiplot object should outlive the PlotPlacer.
  //  num_plots: Number of plots to add in the multiplot.
  //  max_num_cols: Maximum number of columns in the multi-plot.
  static absl::StatusOr<PlotPlacer> Create(int num_plots, int max_num_cols,
                                           MultiPlot* multiplot);

  // Adds and returns a new plot. Returns a non-owning pointer to the plot.
  absl::StatusOr<Plot*> NewPlot();

  // To be called after all the calls to "NewPlot".
  // This method ensures that "NewPlot" was called "num_plots" times.
  absl::Status Finalize();

 private:
  PlotPlacer(int num_plots, int num_cols, int num_rows, MultiPlot* multiplot);

  // Constructor arguments.
  int num_plots_;
  MultiPlot* multiplot_;

  // True iff. "Finalize" was called.
  bool finalize_called_ = false;

  // Number of times "NewPlot" was called.
  int num_new_plots_ = 0;
};

namespace internal {
namespace plotly {

// Specialization of ExportToHtml to plotly.
absl::StatusOr<std::string> ExportToHtml(const Plot& plot,
                                         const ExportOptions& options);

}  // namespace plotly
}  // namespace internal

}  // namespace plot
}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_PLOT_H_
