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

#include "yggdrasil_decision_forests/utils/plot.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/utils/html.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/uid.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace plot {

absl::Status Curve::Check() const {
  STATUS_CHECK(xs.empty() || xs.size() == ys.size());
  return absl::OkStatus();
}

absl::Status Bars::Check() const {
  STATUS_CHECK(centers.size() == heights.size());
  return absl::OkStatus();
}

absl::Status Plot::Check() const {
  for (const auto& item : items) {
    RETURN_IF_ERROR(item->Check());
  }
  return absl::OkStatus();
}

absl::Status MultiPlot::Check() const {
  STATUS_CHECK_GE(num_cols, 0);
  STATUS_CHECK_GE(num_rows, 0);

  for (const auto& item : items) {
    STATUS_CHECK_GE(item.col, 0);
    STATUS_CHECK_GE(item.row, 0);
    STATUS_CHECK_GE(item.num_cols, 1);
    STATUS_CHECK_GE(item.num_rows, 1);
    STATUS_CHECK_LE(item.col + item.num_cols, num_cols);
    STATUS_CHECK_LE(item.row + item.num_rows, num_rows);

    RETURN_IF_ERROR(item.plot.Check());
  }

  return absl::OkStatus();
}

absl::Status Bars::FromHistogram(
    const utils::histogram::Histogram<float>& hist) {
  centers.clear();
  heights.clear();
  for (int bucket_idx = 0; bucket_idx < hist.counts().size(); bucket_idx++) {
    heights.push_back(hist.counts()[bucket_idx]);
    centers.push_back(
        (hist.bounds()[bucket_idx] + hist.bounds()[bucket_idx + 1]) / 2);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> ExportToHtml(const Plot& plot,
                                         const ExportOptions& options) {
  if (options.run_checks) {
    RETURN_IF_ERROR(plot.Check());
  }
  switch (options.target_library) {
    case TargetLibrary::kPlotly:
      return internal::plotly::ExportToHtml(plot, options);
  }
}

absl::StatusOr<std::string> ExportToHtml(const MultiPlot& multiplot,
                                         const ExportOptions& options) {
  if (options.run_checks) {
    RETURN_IF_ERROR(multiplot.Check());
  }

  std::string html;

  // Create a grid of size num_cols x num_rows.
  absl::StrAppend(&html,
                  "<div style='display: grid; gap: 0px; "
                  "grid-auto-columns: min-content;'>");

  for (int item_idx = 0; item_idx < multiplot.items.size(); item_idx++) {
    const auto& item = multiplot.items[item_idx];
    auto sub_options = options;

    // The checking is already done above.
    sub_options.run_checks = false;

    if (item_idx != 0) {
      sub_options.first_export = false;
    }

    // Generates the html of the sub plot.
    ASSIGN_OR_RETURN(const auto sub_html, ExportToHtml(item.plot, sub_options));

    // Write the sub-plot into a grid cell.
    absl::SubstituteAndAppend(
        &html, "<div style='grid-row:$0 / $1; grid-column:$2 / $3;'>$4</div>",
        item.row + 1, item.row + item.num_rows + 1,
        item.col + item.num_cols + 1, item.col + 1, sub_html);
  }

  absl::StrAppend(&html, "</div>");
  return html;
}

absl::StatusOr<PlotPlacer> PlotPlacer::Create(int num_plots, int max_num_cols,
                                              MultiPlot* multiplot) {
  STATUS_CHECK_GT(num_plots, 0);
  STATUS_CHECK_GT(max_num_cols, 0);
  STATUS_CHECK(multiplot);

  const int num_cols = std::min(num_plots, max_num_cols);
  DCHECK_GT(num_cols, 0);

  // Integer ceil.
  const int num_rows = (num_plots + num_cols - 1) / num_cols;

  return PlotPlacer(num_plots, num_cols, num_rows, multiplot);
}

PlotPlacer::PlotPlacer(const int num_plots, const int num_cols,
                       const int num_rows, MultiPlot* multiplot)
    : num_plots_(num_plots), multiplot_(multiplot) {
  multiplot_->items.resize(num_plots);
  multiplot_->num_cols = num_cols;
  multiplot_->num_rows = num_rows;
}

absl::StatusOr<Plot*> PlotPlacer::NewPlot() {
  STATUS_CHECK(multiplot_);
  STATUS_CHECK_LT(num_new_plots_, num_plots_);
  STATUS_CHECK(!finalize_called_);
  auto& item = multiplot_->items[num_new_plots_];
  item.col = num_new_plots_ % multiplot_->num_cols;
  item.row = num_new_plots_ / multiplot_->num_cols;
  num_new_plots_++;
  return &item.plot;
}

absl::Status PlotPlacer::Finalize() {
  STATUS_CHECK(!finalize_called_);
  STATUS_CHECK_EQ(num_new_plots_, num_plots_);
  finalize_called_ = true;
  return absl::OkStatus();
}

namespace internal {

namespace plotly {

// Adds the plotly header.
absl::Status AppendHeader(std::string* html) {
  absl::SubstituteAndAppend(
      html, "<script src='$0'></script>",
      "https://www.gstatic.com/external_hosted/plotly/plotly.min.js");
  return absl::OkStatus();
}

absl::Status ExportCurveToHtml(const Curve& curve, const int item_idx,
                               const ExportOptions& options,
                               ExportAccumulator* export_acc) {
  absl::StrAppend(&export_acc->data, "{\n");

  // Xs.
  if (!curve.xs.empty()) {
    absl::SubstituteAndAppend(&export_acc->data, "x: [$0],\n",
                              absl::StrJoin(curve.xs, ","));
  }

  std::string line_style;
  switch (curve.style) {
    case LineStyle::SOLID:
      line_style = "solid";
      break;
    case LineStyle::DOTTED:
      line_style = "dot";
      break;
  }

  absl::SubstituteAndAppend(&export_acc->data, R"(y: [$0],
type: 'scatter',
mode: 'lines',
line: {
  dash: '$1',
  width: 1
},
)",
                            absl::StrJoin(curve.ys, ","),  // $0
                            line_style                     // $1
  );

  // Label.
  if (!curve.label.empty()) {
    absl::SubstituteAndAppend(&export_acc->data, "name: '$0',\n", curve.label);
  }

  absl::StrAppend(&export_acc->data, "},\n");
  return absl::OkStatus();
}

absl::Status ExportBarsToHtml(const Bars& bars, const int item_idx,
                              const ExportOptions& options,
                              ExportAccumulator* export_acc) {
  absl::StrAppend(&export_acc->data, "{\n");

  absl::SubstituteAndAppend(&export_acc->data, "x: [$0],\n",
                            absl::StrJoin(bars.centers, ","));

  absl::SubstituteAndAppend(&export_acc->data, R"(y: [$0],
type: 'bar',
)",
                            absl::StrJoin(bars.heights, ","));

  if (!bars.label.empty()) {
    absl::SubstituteAndAppend(&export_acc->data, "name: '$0',\n", bars.label);
  }

  absl::StrAppend(&export_acc->data, "},\n");
  return absl::OkStatus();
}

absl::Status ExportPlotItemToHtml(const PlotItem* item, const int item_idx,
                                  const ExportOptions& options,
                                  ExportAccumulator* export_acc) {
  // Note: We use reflection instead of class inheritance so the code plotting
  // code does not depends on the export libraries (e.g., c3js) code.

  const auto* curve = dynamic_cast<const Curve*>(item);
  if (curve) {
    return ExportCurveToHtml(*curve, item_idx, options, export_acc);
  }

  const auto* bars = dynamic_cast<const Bars*>(item);
  if (bars) {
    return ExportBarsToHtml(*bars, item_idx, options, export_acc);
  }

  return absl::UnimplementedError(
      "Support for this plot item not implemented in plotly");
}

// Specialization of ExportToHtml for c3js.
absl::StatusOr<std::string> ExportToHtml(const Plot& plot,
                                         const ExportOptions& options) {
  if (options.run_checks) {
    RETURN_IF_ERROR(plot.Check());
  }

  std::string html;
  const auto chart_id =
      plot.chart_id.empty()
          ? absl::StrCat("chart_",
                         absl::StrReplaceAll(GenUniqueId(), {{"-", "_"}}))
          : plot.chart_id;

  // Library header
  if (options.first_export) {
    RETURN_IF_ERROR(AppendHeader(&html));
  }

  ExportAccumulator export_acc;

  // Items in the plot.
  for (int item_idx = 0; item_idx < plot.items.size(); item_idx++) {
    const auto& item = plot.items[item_idx];
    RETURN_IF_ERROR(
        ExportPlotItemToHtml(item.get(), item_idx, options, &export_acc));
  }

  // Export the html
  absl::SubstituteAndAppend(&html,
                            R"(
<div id="$0" style="display: inline-block;" ></div>
<script>
  Plotly.newPlot(
    '$0',
    [$1],
    {
      width: $5,
      height: $6,
      title: '$2',
      showlegend: $7,
      xaxis: {
        ticks: 'outside',
        showgrid: true,
        zeroline: false,
        showline: true,
        title: '$3',
        },
      font: {
        size: 10,
        },
      yaxis: {
        ticks: 'outside',
        showgrid: true,
        zeroline: false,
        showline: true,
        title: '$4',
        },
      margin: {
        l: 50,
        r: 50,
        b: 50,
        t: 50,
      },
    },
    {
      modeBarButtonsToRemove: ['sendDataToCloud'],
      displaylogo: false,
    }
  );
</script>
)",
                            chart_id,                            // $0
                            export_acc.data,                     // $1
                            html::Escape(plot.title),            // $2
                            html::Escape(plot.x_axis.label),     // $3
                            html::Escape(plot.y_axis.label),     // $4
                            options.width,                       // $5
                            options.height,                      // $6
                            plot.show_legend ? "true" : "false"  // $7
  );
  return html;
}
}  // namespace plotly

}  // namespace internal
}  // namespace plot
}  // namespace utils
}  // namespace yggdrasil_decision_forests
