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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/plot_training.h"

#include "learning/lib/ami/tools/simpleplot/fig_canvas.h"
#include "learning/lib/ami/tools/simpleplot/simpleplot.pb.h"
#include "learning/lib/ami/tools/simpleplot/svg_canvas.h"
#include "yggdrasil_decision_forests/model/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

// Create a multi-plot with a log of training. The plot contains:
// - The training and validation loss according to the number of trees.
// - The secondary metric (e.g. accuracy) according to the number of trees.
simpleplot::MultiPlot CreatePlotOfLogs(
    const proto::TrainingLogs& training_logs) {
  // Labels of the plot.
  constexpr char kLabelNumTrees[] = "Number of trees";
  constexpr char kLabelTraining[] = "Training";
  constexpr char kLabelValidation[] = "Validation";
  constexpr char kLabelLoss[] = "Loss";
  constexpr char kMeanAbsPrediction[] = "Mean absolute prediction";

  // Create the sub-plots:
  //   - row 0 -> number of trees vs loss.
  //   - row 1 and above -> number of trees vs the secondary metrics.
  simpleplot::MultiPlot multi_plot;
  multi_plot.set_total_cols(1);
  multi_plot.set_total_rows(2 + training_logs.secondary_metric_names().size());

  auto* loss_sub_plot = multi_plot.add_subplots();
  loss_sub_plot->set_row(0);
  loss_sub_plot->set_col(0);
  auto* loss_plot = loss_sub_plot->mutable_plot();

  // Set the axis labels.
  loss_plot->mutable_xaxis()->set_label(kLabelNumTrees);
  loss_plot->mutable_yaxis()->set_label(kLabelLoss);

  // Create curves.
  auto* training_loss = loss_plot->add_items()->mutable_curve();
  auto* validation_loss = loss_plot->add_items()->mutable_curve();

  // Set the legend.
  training_loss->set_label(kLabelTraining);
  validation_loss->set_label(kLabelValidation);

  // Set curves' drawing style.
  training_loss->set_line_style(simpleplot::LineStyle::SOLID);
  validation_loss->set_line_style(simpleplot::LineStyle::SOLID);

  // Mean Average prediction i.e. mean(abs(model output)).
  auto* mean_abs_pred_sub_plot = multi_plot.add_subplots();
  mean_abs_pred_sub_plot->set_row(
      1 + training_logs.secondary_metric_names().size());
  mean_abs_pred_sub_plot->set_col(0);
  auto* mean_abs_pred_plot = mean_abs_pred_sub_plot->mutable_plot();

  // Set the axis labels.
  mean_abs_pred_plot->mutable_xaxis()->set_label(kLabelNumTrees);
  mean_abs_pred_plot->mutable_yaxis()->set_label(kMeanAbsPrediction);
  mean_abs_pred_plot->mutable_yaxis()->set_scale(simpleplot::AxisScale::LOG);

  // Create curves.
  auto* mean_abs_pred = mean_abs_pred_plot->add_items()->mutable_curve();
  mean_abs_pred->set_line_style(simpleplot::LineStyle::SOLID);

  std::vector<simpleplot::Curve*> training_secondary_metrics;
  std::vector<simpleplot::Curve*> validation_secondary_metrics;

  const bool has_validation =
      training_logs.entries_size() > 0 &&
      !training_logs.entries(0).validation_secondary_metrics().empty();

  for (int secondary_metric_idx = 0;
       secondary_metric_idx < training_logs.secondary_metric_names().size();
       secondary_metric_idx++) {
    // Create the plot.
    auto* secondary_metric_sub_plot = multi_plot.add_subplots();
    secondary_metric_sub_plot->set_col(0);
    secondary_metric_sub_plot->set_row(1 + secondary_metric_idx);
    auto* secondary_metric_plot = secondary_metric_sub_plot->mutable_plot();

    // Set the axis labels.
    secondary_metric_plot->mutable_xaxis()->set_label(kLabelNumTrees);
    secondary_metric_plot->mutable_yaxis()->set_label(
        training_logs.secondary_metric_names(secondary_metric_idx));

    // Create curves.
    auto* training_secondary_metric =
        secondary_metric_plot->add_items()->mutable_curve();
    auto* validation_secondary_metric =
        secondary_metric_plot->add_items()->mutable_curve();

    // Set the legend.
    training_secondary_metric->set_label(kLabelTraining);
    // Set curves' drawing style.
    training_secondary_metric->set_line_style(simpleplot::LineStyle::SOLID);
    training_secondary_metrics.push_back(training_secondary_metric);

    // Same for the validation curve.
    if (has_validation) {
      validation_secondary_metric->set_label(kLabelValidation);
      validation_secondary_metric->set_line_style(simpleplot::LineStyle::SOLID);
      validation_secondary_metrics.push_back(validation_secondary_metric);
    }
  }

  // Set the curves' x,y values.
  for (const auto& entry : training_logs.entries()) {
    // X axis
    training_loss->add_xs(entry.number_of_trees());
    validation_loss->add_xs(entry.number_of_trees());
    mean_abs_pred->add_xs(entry.number_of_trees());

    // Y axis
    training_loss->add_ys(entry.training_loss());
    validation_loss->add_ys(entry.validation_loss());
    mean_abs_pred->add_ys(entry.mean_abs_prediction());

    CHECK_EQ(training_logs.secondary_metric_names().size(),
             entry.training_secondary_metrics().size());
    if (has_validation) {
      CHECK_EQ(training_logs.secondary_metric_names().size(),
               entry.validation_secondary_metrics().size());
    }

    for (int secondary_metric_idx = 0;
         secondary_metric_idx < training_logs.secondary_metric_names().size();
         secondary_metric_idx++) {
      training_secondary_metrics[secondary_metric_idx]->add_xs(
          entry.number_of_trees());
      training_secondary_metrics[secondary_metric_idx]->add_ys(
          entry.training_secondary_metrics(secondary_metric_idx));

      if (has_validation) {
        validation_secondary_metrics[secondary_metric_idx]->add_xs(
            entry.number_of_trees());
        validation_secondary_metrics[secondary_metric_idx]->add_ys(
            entry.validation_secondary_metrics(secondary_metric_idx));
      }
    }
  }
  return multi_plot;
}

}  // namespace

void PlotAndExportTrainingLogs(const proto::TrainingLogs& training_logs,
                               absl::string_view directory) {
  QCHECK_OK(file::RecursivelyCreateDir(directory, file::Defaults()));
  const auto plots = CreatePlotOfLogs(training_logs);
  simpleplot::SVGOutputCanvas canvas(800, 400 * plots.subplots_size());
  canvas.GetFigure().DrawMultiPlot(plots);

  const auto svg_plot_path = file::JoinPath(directory, "training_logs.svg");
  canvas.SaveToFile(svg_plot_path);
  const auto pbbin_plot_path = file::JoinPath(directory, "training_logs.pbbin");
  QCHECK_OK(file::SetBinaryProto(pbbin_plot_path, plots, file::Defaults()));
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
