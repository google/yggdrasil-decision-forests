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

#include "yggdrasil_decision_forests/model/describe.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_forest_interface.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/hyperparameter.pb.h"
#include "yggdrasil_decision_forests/utils/documentation.h"
#include "yggdrasil_decision_forests/utils/html.h"
#include "yggdrasil_decision_forests/utils/html_content.h"
#include "yggdrasil_decision_forests/utils/plot.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::model {
namespace {

void AddKeyValue(utils::html::Html* dst, const absl::string_view key,
                 const absl::string_view value) {
  namespace h = utils::html;
  dst->Append(h::B(key));
  dst->Append(" : ");
  dst->Append(value);
  dst->Append(h::Br());
}

void AddKeyMultiLinesValue(utils::html::Html* dst, const absl::string_view key,
                           const absl::string_view value) {
  namespace h = utils::html;
  dst->Append(h::B(key));
  dst->Append(" : ");
  dst->Append(value);
  dst->Append(h::Br());
}

bool HasTuner(const AbstractModel& model) {
  return model.hyperparameter_optimizer_logs().has_value();
}

bool HasFeatureSelector(const AbstractModel& model) {
  return model.feature_selection_logs().has_value();
}

utils::html::Html Model(const model::AbstractModel& model) {
  namespace h = utils::html;
  h::Html content;
  AddKeyValue(&content, "Name", model.name());
  AddKeyValue(&content, "Task", proto::Task_Name(model.task()));
  if (model.has_label()) {
    AddKeyValue(&content, "Label", model.label());
  }

  if (model.ranking_group_col_idx() != -1) {
    AddKeyValue(
        &content, "Ranking group",
        model.data_spec().columns(model.ranking_group_col_idx()).name());
  }

  if (model.uplift_treatment_col_idx() != -1) {
    AddKeyValue(
        &content, "Uplifting treatment",
        model.data_spec().columns(model.uplift_treatment_col_idx()).name());
  }

  std::string str_input_features;
  for (int i = 0; i < model.input_features().size(); i++) {
    if (i != 0) {
      absl::StrAppend(&str_input_features, " ");
    }
    absl::StrAppend(
        &str_input_features,
        model.data_spec().columns(model.input_features()[i]).name());
  }
  AddKeyValue(&content,
              absl::StrCat("Features (", model.input_features().size(), ")"),
              str_input_features);

  if (!model.weights().has_value()) {
    AddKeyValue(&content, "Weights", "None");
  } else {
    AddKeyMultiLinesValue(
        &content, "Weights",
        utils::SerializeTextProto(*model.weights()).value_or("ERROR"));
  }

  AddKeyValue(&content, "Trained with tuner", HasTuner(model) ? "Yes" : "No");

  if (const auto model_size = model.ModelSizeInBytes();
      model_size.has_value()) {
    AddKeyValue(&content, "Model size",
                absl::StrCat(*model_size / 1000, " kB"));
  }

  return content;
}

absl::StatusOr<utils::html::Html> VariableImportances(
    const model::AbstractModel& model, const absl::string_view block_id) {
  absl::flat_hash_map<std::string, std::vector<proto::VariableImportance>>
      variable_importances;
  for (const auto& key : model.AvailableVariableImportances()) {
    ASSIGN_OR_RETURN(auto vi, model.GetVariableImportance(key));
    variable_importances[key] = vi;
  }
  ASSIGN_OR_RETURN(auto content,
                   VariableImportance(variable_importances, model.data_spec(),
                                      absl::StrCat(block_id, "_vi")));

  content.Append(utils::html::P(
      "Those variable importances are computed during training. More, and "
      "possibly more informative, variable importances are available when "
      "analyzing a model on a test dataset."));

  return content;
}

std::string FieldToString(
    const model::proto::GenericHyperParameters::Value& value) {
  switch (value.Type_case()) {
    case model::proto::GenericHyperParameters::Value::kCategorical:
      return value.categorical();
    case model::proto::GenericHyperParameters::Value::kInteger:
      return absl::StrCat(value.integer());
    case model::proto::GenericHyperParameters::Value::kReal:
      return absl::StrCat(value.real());
    case model::proto::GenericHyperParameters::Value::kCategoricalList:
      return absl::StrJoin(value.categorical_list().values().begin(),
                           value.categorical_list().values().end(), ", ");
    default:
      return "?";
  }
}

utils::html::Html TuningLogs(const model::AbstractModel& model) {
  namespace h = utils::html;
  h::Html content;

  const auto help =
      h::P("A ",
           h::A(h::Target("_blank"),
                h::HRef(utils::documentation::GlossaryTuner()), "tuner"),
           " automatically selects the hyper-parameters of a learner.");
  content.Append(help);

  content.Append(h::P(h::B("Note:"),
                      "The hyper-parameters tuning logs are accessible "
                      "programmatically in python with "
                      "`model.hyperparameter_optimizer_logs()`."));

  const auto& logs = *model.hyperparameter_optimizer_logs();

  // Index the possible fields and scores.
  absl::flat_hash_map<std::string, int> field_to_idx;
  std::vector<std::string> filed_names;

  struct Trial {
    int step_idx;
    float score;
  };
  std::vector<Trial> trials;
  trials.reserve(logs.steps_size());

  for (int step_idx = 0; step_idx < logs.steps_size(); step_idx++) {
    const auto& step = logs.steps(step_idx);
    trials.push_back({.step_idx = step_idx, .score = step.score()});

    for (const auto& field : step.hyperparameters().fields()) {
      if (field_to_idx.find(field.name()) == field_to_idx.end()) {
        const int idx = field_to_idx.size();
        field_to_idx[field.name()] = idx;
        filed_names.push_back(field.name());
      }
    }
  }
  std::sort(trials.begin(), trials.end(),
            [](const auto& a, const auto b) { return a.score > b.score; });

  // Set table header
  h::Html rows;
  {
    h::Html row;
    row.Append(h::Th("trial"));
    row.Append(h::Th("score"));
    row.Append(h::Th("duration"));
    for (const auto& name : filed_names) {
      row.Append(h::Th(name));
    }
    rows.Append(h::Tr(row));
  }

  // Fill the table
  for (const auto& trial : trials) {
    const auto& step = logs.steps(trial.step_idx);

    // Values ordered by field index.
    std::vector<std::string> values(filed_names.size(), " ");

    for (const auto& field : step.hyperparameters().fields()) {
      const int field_idx = field_to_idx.find(field.name())->second;
      values[field_idx] = FieldToString(field.value());
    }

    h::Html row;
    row.Append(h::Td(absl::StrCat(trial.step_idx)));
    row.Append(h::Td(absl::StrCat(step.score())));
    row.Append(h::Td(absl::StrCat(step.evaluation_time())));
    for (const auto& value : values) {
      row.Append(h::Td(value));
    }

    // TODO: Don't compare hps by string.
    if (step.hyperparameters().DebugString() ==
        logs.best_hyperparameters().DebugString()) {
      rows.Append(h::Tr(h::Class("best"), row));
    } else {
      rows.Append(h::Tr(row));
    }
  }

  content.Append(h::Table(h::Class("ydf_tuning_table"), rows));
  return content;
}

// Creates a plot of the feature selector logs.
absl::StatusOr<utils::plot::MultiPlot> FeatureSelectorLogsPlot(
    const proto::FeatureSelectionLogs& logs,
    const std::vector<std::string>& metric_keys) {
  utils::plot::MultiPlot multiplot;
  // Creates a plot for each metric, for the score and for the number of
  // features.
  ASSIGN_OR_RETURN(auto placer, utils::plot::PlotPlacer::Create(
                                    2 + metric_keys.size(), 2, &multiplot));

  std::vector<double> iteration_idxs(logs.iterations_size());
  std::iota(iteration_idxs.begin(), iteration_idxs.end(), 0);

  const auto update_min_max_value = [](const float value, float& min_value,
                                       float& max_value) {
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
  };

  const auto add_best_iteration_vertical_line =
      [&logs](const float min_value, const float max_value,
              utils::plot::Plot* plot) {
        if (!std::isinf(min_value)) {
          // Adds a 10% extra
          float margin = (max_value - min_value) / 10;

          auto* vertical = plot->AddCurve();
          vertical->xs.push_back(logs.best_iteration_idx());
          vertical->xs.push_back(logs.best_iteration_idx());
          vertical->ys.push_back(min_value - margin);
          vertical->ys.push_back(max_value + margin);
        }
      };

  // Score plot
  {
    ASSIGN_OR_RETURN(auto* plot, placer.NewPlot());
    plot->show_legend = false;
    plot->x_axis.label = "iteration";
    plot->y_axis.label = "score";
    auto* curve = plot->AddCurve();
    curve->xs = iteration_idxs;
    curve->ys.reserve(logs.iterations_size());
    float min_value = +std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    for (const auto& iteration : logs.iterations()) {
      curve->ys.push_back(iteration.score());
      update_min_max_value(iteration.score(), min_value, max_value);
    }
    add_best_iteration_vertical_line(min_value, max_value, plot);
  }

  // Number of features plot
  {
    ASSIGN_OR_RETURN(auto* plot, placer.NewPlot());
    plot->show_legend = false;
    plot->x_axis.label = "iteration";
    plot->y_axis.label = "number of features";
    auto* curve = plot->AddCurve();
    curve->xs = iteration_idxs;
    curve->ys.reserve(logs.iterations_size());
    float min_value = +std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    for (const auto& iteration : logs.iterations()) {
      curve->ys.push_back(iteration.features_size());
      update_min_max_value(iteration.features_size(), min_value, max_value);
    }
    add_best_iteration_vertical_line(min_value, max_value, plot);
  }

  for (const auto& metric_key : metric_keys) {
    ASSIGN_OR_RETURN(auto* plot, placer.NewPlot());
    plot->show_legend = false;
    plot->x_axis.label = "iteration";
    plot->y_axis.label = metric_key;
    auto* curve = plot->AddCurve();
    curve->xs = iteration_idxs;
    curve->ys.reserve(logs.iterations_size());
    float min_value = +std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    for (const auto& iteration : logs.iterations()) {
      const auto it_metric = iteration.metrics().find(metric_key);
      if (it_metric != iteration.metrics().end()) {
        curve->ys.push_back(it_metric->second);
        update_min_max_value(it_metric->second, min_value, max_value);
      } else {
        curve->ys.push_back(std::numeric_limits<double>::quiet_NaN());
      }
    }

    add_best_iteration_vertical_line(min_value, max_value, plot);
  }

  RETURN_IF_ERROR(placer.Finalize());
  return multiplot;
}

// Creates a table of the feature selector logs.
absl::StatusOr<utils::html::Html> FeatureSelectorLogsTable(
    const proto::FeatureSelectionLogs& logs,
    const std::vector<std::string>& metric_keys) {
  namespace h = utils::html;

  h::Html rows;
  {
    // Create header
    h::Html row;
    row.Append(h::Th("Iteration"));
    row.Append(h::Th("Score"));
    row.Append(h::Th("Metrics"));
    row.Append(h::Th("Num features"));
    row.Append(h::Th("Features"));
    rows.Append(h::Tr(row));
  }

  // Fill table content
  for (int iteration_idx = 0; iteration_idx < logs.iterations_size();
       iteration_idx++) {
    const auto& iteration = logs.iterations(iteration_idx);

    h::Html row;
    row.Append(h::Td(absl::StrCat(iteration_idx)));
    row.Append(h::Td(absl::StrCat(iteration.score())));

    std::string str_metrics;
    for (const auto& metric_key : metric_keys) {
      const auto& it_metric = iteration.metrics().find(metric_key);
      if (it_metric == iteration.metrics().end()) {
        continue;
      }
      absl::StrAppendFormat(&str_metrics, " %s:%g", it_metric->first,
                            it_metric->second);
    }
    row.Append(h::Td(str_metrics));
    row.Append(h::Td(absl::StrCat(iteration.features_size())));
    row.Append(h::Td(absl::StrJoin(iteration.features(), " ")));

    if (iteration_idx == logs.best_iteration_idx()) {
      rows.Append(h::Tr(h::Class("best"), row));
    } else {
      rows.Append(h::Tr(row));
    }
  }
  return h::Table(h::Class("ydf_tuning_table"), rows);
}

// Creates a HTML section for the feature selector logs.
absl::StatusOr<utils::html::Html> FeatureSelectorLogs(
    const model::AbstractModel& model, const absl::string_view block_id) {
  namespace h = utils::html;
  h::Html content;

  const auto help =
      h::P(h::A(h::Target("_blank"),
                h::HRef(utils::documentation::GlossaryFeatureSelection()),
                "Feature selection"),
           " automatically identify and remove unnecessary input features of "
           "the model.");
  content.Append(help);
  content.Append(h::P(
      "The plots below show the score, number of features, and metric values "
      "for each iteration of the feature selection process.  The orange "
      "vertical line marks the highest score; which defines the features "
      "are used in the final model. The table below gives the detailed logs."));

  content.Append(h::P(
      h::B("Note:"),
      "The feature selection logs are accessible "
      "programmatically in Python with "
      "`model.feature_selection_logs()`. The selected features are accessible "
      "programmatically in Python with `model.input_feature_names()`"));

  const auto& logs = *model.feature_selection_logs();

  // Find the metric keys.
  absl::flat_hash_set<std::string> metric_keys_set;
  for (const auto& metric : logs.iterations(0).metrics()) {
    metric_keys_set.insert(metric.first);
  }
  std::vector<std::string> metric_keys(metric_keys_set.begin(),
                                       metric_keys_set.end());
  std::sort(metric_keys.begin(), metric_keys.end());

  // Plot with the results
  ASSIGN_OR_RETURN(const auto plot, FeatureSelectorLogsPlot(logs, metric_keys));
  ASSIGN_OR_RETURN(
      const auto html_plot,
      utils::plot::ExportToHtml(plot, {.html_id_prefix = absl::StrCat(
                                           block_id, "feature_selector")}));
  content.Append(h::P(h::B("Plots")));
  content.AppendRaw(html_plot);

  // Table with the results.
  ASSIGN_OR_RETURN(const auto table,
                   FeatureSelectorLogsTable(logs, metric_keys));
  content.Append(h::P(h::B("Table")));
  content.Append(table);
  return content;
}

absl::StatusOr<utils::html::Html> SelfEvaluation(
    const model::AbstractModel& model, const absl::string_view block_id) {
  namespace h = utils::html;
  h::Html content;

  const auto help = h::P(
      "The following evaluation is computed on the validation "
      "or out-of-bag dataset.");
  content.Append(help);

  const auto text_report_or = metric::TextReport(model.ValidationEvaluation());
  if (text_report_or.ok()) {
    content.Append(h::Pre(h::Class("ydf_pre"), *text_report_or));
  }

  const auto plot_or = model.PlotTrainingLogs();
  if (plot_or.ok()) {
    ASSIGN_OR_RETURN(
        const auto html_plot,
        utils::plot::ExportToHtml(
            *plot_or, {.html_id_prefix = absl::StrCat(block_id, "self_eval")}));
    content.AppendRaw(html_plot);
  } else {
    content.Append(h::P(plot_or.status().message()));
  }

  return content;
}

absl::StatusOr<utils::html::Html> Structure(const model::AbstractModel& model) {
  namespace h = utils::html;
  h::Html content;

  const auto* df = dynamic_cast<const DecisionForestInterface*>(&model);
  if (!df) {
    content.Append(h::P("The model is not a decision forests"));
    return content;
  }

  // TODO: Plot the trees.
  AddKeyValue(&content, "Num trees", absl::StrCat(df->num_trees()));

  const int max_trees = 1;
  const int num_trees = df->num_trees();
  if (num_trees > max_trees) {
    content.Append(h::P("Only printing the first tree."));
  }

  std::string str_trees;
  for (int tree_idx = 0; tree_idx < num_trees; tree_idx++) {
    if (tree_idx >= max_trees) {
      break;
    }
    absl::StrAppend(&str_trees, "Tree #", tree_idx, ":\n");
    df->decision_trees()[tree_idx]->AppendModelStructure(
        model.data_spec(), model.label_col_idx(), &str_trees);
  }
  content.Append(h::Pre(h::Class("ydf_pre"), str_trees));
  return content;
}

}  // namespace

std::string Header() {
  return absl::Substitute(R"(
<style>
$0

.variable_importance {
}

.variable_importance select {
}

.variable_importance .content {
  display: none;
}

.variable_importance .content.selected {
  display: block;
}

.ydf_tuning_table {
  border-collapse: collapse;
  border: 1px solid lightgray;
}

.ydf_tuning_table th {
  background-color: #ededed;
  font-weight: bold;
  text-align: left;
  padding: 3px 4px;
  border: 1px solid lightgray;
}

.ydf_tuning_table td {
  text-align: right;
  padding: 3px 4px;
  border: 1px solid lightgray;
}

.ydf_tuning_table .best {
  background-color: khaki;
}

</style>

<script>
$1

function ydfShowVariableImportance(block_id) {
    const block = document.getElementById(block_id);
    const item = block.getElementsByTagName("select")[0].value;
    block.getElementsByClassName("content selected")[0].classList.remove("selected");
    document.getElementById(block_id + "_body_" + item).classList.add("selected");
}

</script>
  )",
                          utils::CssCommon(), utils::JsCommon());
}

absl::StatusOr<utils::html::Html> VariableImportance(
    const absl::flat_hash_map<std::string,
                              std::vector<proto::VariableImportance>>&
        variable_importances,
    const dataset::proto::DataSpecification& data_spec,
    const absl::string_view block_id) {
  if (block_id.empty()) {
    return absl::InvalidArgumentError("empty block_id");
  }

  namespace h = utils::html;

  h::Html select_options;
  h::Html select_content;

  // Adds a block of content.
  bool first_entry = true;
  const auto add_entry = [&select_content, &select_options, &block_id,
                          &first_entry](const absl::string_view key,
                                        const absl::string_view title,
                                        const h::Html& content) {
    const absl::string_view maybe_selected = first_entry ? " selected" : "";
    select_options.Append(h::Option(h::Value(key), title));
    select_content.Append(
        h::Div(h::Id(absl::StrCat(block_id, "_body_", key)),
               h::Class(absl::StrCat("content", maybe_selected)), content));
    first_entry = false;
  };

  // Sort Variable Importances by key
  std::vector<std::string> keys;
  keys.reserve(variable_importances.size());
  for (const auto& vi : variable_importances) {
    keys.push_back(vi.first);
  }
  std::sort(keys.begin(), keys.end());

  for (const auto& key : keys) {
    const auto& vi = *variable_importances.find(key);
    // Export VI plot
    // TODO: Use an html plot instead of ascii-art.
    std::string raw;
    model::AppendVariableImportanceDescription(vi.second, data_spec, 4, &raw);
    add_entry(vi.first, vi.first, h::Pre(h::Class("ydf_pre"), raw));
  }

  h::Html content;
  const auto onchange =
      absl::Substitute("ydfShowVariableImportance('$0')", block_id);

  const auto help =
      h::P(h::A(h::Target("_blank"),
                h::HRef(utils::documentation::VariableImportance()),
                "Variable importances"),
           " measure the importance of an input feature for a model.");
  content.Append(help);

  content.Append(
      h::Div(h::Id(absl::StrCat(block_id)), h::Class("variable_importance"),
             h::Select(h::OnChange(onchange), select_options), select_content));
  return content;
}

absl::StatusOr<std::string> DescribeModelHtml(
    const model::AbstractModel& model, const absl::string_view block_id) {
  if (block_id.empty()) {
    return absl::InvalidArgumentError("empty block_id");
  }

  namespace h = utils::html;
  h::Html html;
  html.AppendRaw(Header());

  utils::TabBarBuilder tabbar(block_id);

  tabbar.AddTab("model", "Model", Model(model));

  {
    h::Html content;
    content.Append(h::Pre(h::Class("ydf_pre"),
                          dataset::PrintHumanReadable(model.data_spec())));
    tabbar.AddTab("dataspec", "Dataspec", content);
  }

  if (HasTuner(model)) {
    tabbar.AddTab("tuning", "Tuning", TuningLogs(model));
  }

  if (HasFeatureSelector(model)) {
    ASSIGN_OR_RETURN(const auto feature_selector_html,
                     FeatureSelectorLogs(model, block_id));
    tabbar.AddTab("feature_selector", "Feature selection",
                  feature_selector_html);
  }

  ASSIGN_OR_RETURN(const auto str_training, SelfEvaluation(model, block_id));
  tabbar.AddTab("training", "Training", str_training);

  ASSIGN_OR_RETURN(const auto str_var_imp,
                   VariableImportances(model, block_id));
  tabbar.AddTab("variable_importance", "Variable importances", str_var_imp);

  ASSIGN_OR_RETURN(const auto str_structure, Structure(model));
  tabbar.AddTab("structure", "Structure", str_structure);

  html.Append(tabbar.Html());
  return std::string(html.content());
}

}  // namespace yggdrasil_decision_forests::model
