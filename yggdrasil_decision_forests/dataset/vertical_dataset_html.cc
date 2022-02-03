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

#include "yggdrasil_decision_forests/dataset/vertical_dataset_html.h"

#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/utils/html.h"

namespace yggdrasil_decision_forests {
namespace dataset {

namespace {

// The alignment when printing a value in a html table.
std::string HtmlAlignment(proto::ColumnType type) {
  switch (type) {
    case proto::ColumnType::NUMERICAL:
    case proto::ColumnType::NUMERICAL_SET:
    case proto::ColumnType::NUMERICAL_LIST:
      return "right";
    default:
      return "left";
  }
}

}  // namespace

void AppendVerticalDatasetToHtml(const VerticalDataset& dataset,
                                 const AppendHtmlOptions& options,
                                 std::string* html) {
  namespace h = yggdrasil_decision_forests::utils::html;
  namespace a = yggdrasil_decision_forests::utils::html;

  // Table
  h::Html rows;
  // Header
  {
    h::Html row;
    for (int col_idx = 0; col_idx < dataset.ncol(); col_idx++) {
      const auto& col_spec = dataset.data_spec().columns(col_idx);
      const std::string alignment = HtmlAlignment(col_spec.type());
      row.Append(h::Th(a::Align(alignment), dataset.column(col_idx)->name()));
    }
    rows.Append(h::Tr(row));
  }
  // Body
  for (VerticalDataset::row_t example_idx = 0; example_idx < dataset.nrow();
       example_idx++) {
    h::Html row;
    for (int col_idx = 0; col_idx < dataset.ncol(); col_idx++) {
      const auto& col_spec = dataset.data_spec().columns(col_idx);
      const std::string alignment = HtmlAlignment(col_spec.type());
      std::string string_value;
      // Note: Cells containing "Non-available" values are left blank (instead
      // of containing "NA").
      if (!dataset.column(col_idx)->IsNa(example_idx)) {
        string_value = dataset.column(col_idx)->ToStringWithDigitPrecision(
            example_idx, col_spec, options.digit_precision);
      }
      row.Append(h::Td(a::Align(alignment), string_value));
    }
    rows.Append(h::Tr(row));
  }

  h::Html content;
  if (options.interactive_column_sorting) {
    if (options.id.empty()) {
      LOG(WARNING) << "sortable tables require a id.";
    }
    content.Append(h::Table(a::Class("sortable"), a::Id(options.id), rows));
  } else {
    if (!options.id.empty()) {
      content.Append(h::Table(a::Id(options.id), rows));
    } else {
      content.Append(h::Table(rows));
    }
  }

  // Column selector
  if (options.interactive_column_selection) {
    h::Html col_checkboxes;
    for (int col_idx = 0; col_idx < dataset.ncol(); col_idx++) {
      col_checkboxes.Append(
          h::Li(h::Input(a::Type("checkbox"),
                         a::DataAttr("data-column-idx", absl::StrCat(col_idx)),
                         a::Checked("")),
                dataset.column(col_idx)->name()));
    }
    const auto checkboxes = h::Ul(col_checkboxes);

    const auto html_control =
        h::P(h::Strong("Hide/Show columns"), h::Br(),
             h::Button(a::Class("show_all"), "Show all"),
             h::Button(a::Class("hide_all"), "Hide all"), checkboxes);
    content.Append(html_control);
  }
  absl::StrAppend(
      html,
      std::string(h::Div(a::Class("table-container"), content).content()));
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
