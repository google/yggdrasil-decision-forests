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

#include "yggdrasil_decision_forests/utils/plot.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/utils/html.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/uid.h"
#include "util/task/status_macros.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace plot {

absl::Status Curve::Check() const {
  STATUS_CHECK(xs.empty() || xs.size() == ys.size());
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

utils::StatusOr<std::string> ExportToHtml(const Plot& plot,
                                          const ExportOptions& options) {
  if (options.run_checks) {
    RETURN_IF_ERROR(plot.Check());
  }
  switch (options.target_library) {
    case TargetLibrary::kC3JS:
      return internal::c3js::ExportToHtml(plot, options);
  }
}

utils::StatusOr<std::string> ExportToHtml(const MultiPlot& multiplot,
                                          const ExportOptions& options) {
  if (options.run_checks) {
    RETURN_IF_ERROR(multiplot.Check());
  }

  std::string html;

  // Create a grid of size num_cols x num_rows.
  absl::SubstituteAndAppend(&html,
                            "<div style='display: grid; gap: 0px; "
                            "grid-template-columns: repeat($0, $1fr);'>",
                            multiplot.num_rows, multiplot.num_cols);

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

namespace internal {
namespace c3js {

// Adds the c3js header.
absl::Status AppendC3header(std::string* html) {
  // Note: There is a tight compatibility matching between the version of c3 and
  // d3.
  absl::SubstituteAndAppend(
      html,
      R"(
<link href="$0" rel="stylesheet">
<script src="$1" charset="utf-8"></script>
<script src="$2"></script>
)",
      "https://www.gstatic.com/external_hosted/c3/c3.min.css",
      "https://d3js.org/d3.v3.min.js",
      "https://www.gstatic.com/external_hosted/c3/c3.min.js");
  return absl::OkStatus();
}

// Adds a new column in "column" format.
void NewColumn(const absl::string_view key, const std::vector<double>& values,
               ExportAccumulator* export_acc) {
  absl::SubstituteAndAppend(&export_acc->data_columns, "\n['$0', ", key);
  absl::StrAppend(&export_acc->data_columns, absl::StrJoin(values, ","));
  absl::StrAppend(&export_acc->data_columns, "],");
}

absl::Status ExportCurveToHtml(const Curve& curve, const int item_idx,
                               const ExportOptions& options,
                               ExportAccumulator* export_acc) {
  // Note: The main_col_name is not displayed in the plot. Instead, it is the
  // way to reference the plot item.
  const auto main_col_name = absl::StrCat("curve_", item_idx);

  // Y values.
  NewColumn(main_col_name, curve.ys, export_acc);

  // X values.
  if (!curve.xs.empty()) {
    const std::string x_col_name = absl::StrCat("curve_", item_idx, "_x");
    NewColumn(x_col_name, curve.xs, export_acc);
    absl::SubstituteAndAppend(&export_acc->data_xs, "'$0': '$1',",
                              main_col_name, x_col_name);
  }

  // Display label.
  const auto effective_label = curve.label.empty()
                                   ? absl::StrCat("item ", item_idx)
                                   : html::Escape(curve.label);
  absl::SubstituteAndAppend(&export_acc->data_names, " $0: '$1',",
                            main_col_name, effective_label);

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

  return absl::UnimplementedError(
      "Support for this plot item not implemented in c3js");
}

// Specialization of ExportToHtml for c3js.
utils::StatusOr<std::string> ExportToHtml(const Plot& plot,
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
    RETURN_IF_ERROR(AppendC3header(&html));
  }

  ExportAccumulator export_acc;

  // Plot title
  if (!plot.title.empty()) {
    absl::SubstituteAndAppend(&export_acc.extra, "\ntitle: { text: '$0'},",
                              html::Escape(plot.title));
  }

  // Axes
  if (!plot.x_axis.label.empty()) {
    absl::SubstituteAndAppend(
        &export_acc.axis,
        "\nx: { label: { text: '$0', position: 'outer-center' } },",
        html::Escape(plot.x_axis.label));
  }
  if (!plot.y_axis.label.empty()) {
    absl::SubstituteAndAppend(
        &export_acc.axis,
        "\ny: { label: { text: '$0', position: 'outer-middle' } },",
        html::Escape(plot.y_axis.label));
  }

  // Items in the plot.
  for (int item_idx = 0; item_idx < plot.items.size(); item_idx++) {
    const auto& item = plot.items[item_idx];
    RETURN_IF_ERROR(
        ExportPlotItemToHtml(item.get(), item_idx, options, &export_acc));
  }

  // Export the html
  absl::SubstituteAndAppend(&html,
                            R"(<div id="$0"></div>
<script>
var $0 = c3.generate({
  bindto: '#$0',
  data: {
      columns: [$1
      ],
      names: {$2
      },
      xs: {$3
      },
  },
  axis: {$4
  },$5
});
</script>
)",
                            chart_id, export_acc.data_columns,
                            export_acc.data_names, export_acc.data_xs,
                            export_acc.axis, export_acc.extra);
  return html;
}

}  // namespace c3js
}  // namespace internal
}  // namespace plot
}  // namespace utils
}  // namespace yggdrasil_decision_forests
