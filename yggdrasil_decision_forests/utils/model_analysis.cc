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

#include "yggdrasil_decision_forests/utils/model_analysis.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/describe.h"
#include "yggdrasil_decision_forests/model/model_engine_wrapper.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/feature_importance.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/html.h"
#include "yggdrasil_decision_forests/utils/html_content.h"
#include "yggdrasil_decision_forests/utils/partial_dependence_plot.h"
#include "yggdrasil_decision_forests/utils/partial_dependence_plot.pb.h"
#include "yggdrasil_decision_forests/utils/plot.h"
#include "yggdrasil_decision_forests/utils/uid.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace model_analysis {
namespace {

// Name of the main report file.
constexpr char kIndex[] = "index.html";

// Plotting labels.
constexpr char kPredictionLegendLabel[] = "Pred";
constexpr char kGroundTruthLegendLabel[] = "Label";
constexpr char kRMSELegendLabel[] = "Rmse";
constexpr char kErrorRateLegendLabel[] = "Error rate";
constexpr char kPredictionAxisLabel[] = "Prediction";
constexpr char kDensity_axisLabel[] = "Distribution";
constexpr char kKeyValueLegendSeparator[] = ":";

// Adds a curve to a plot.
plot::Curve* AddCurve(plot::Plot* plot) {
  auto curve = absl::make_unique<plot::Curve>();
  auto* curve_ptr = curve.get();
  plot->items.push_back(std::move(curve));
  return curve_ptr;
}

// Gets the first categorical label value to plot in PDP curves.
//
// In binary classification, the two label classes lead to symmetrical curves.
// Therefore, the first label class is skipped in the case of binary
// classification.
int FirstCategoricalLabelValueForPdpPlot(
    const dataset::proto::Column& label_spec) {
  return (label_spec.categorical().number_of_unique_values() == 3) ? 2 : 1;
}

// Given a pdp (or a conditional expectation), plots the 1-dimensional relation
// between the input feature and the "target" \in {prediction, ground truth,
// evaluation}.
//
// Arguments:
//   target_type: What to plot e.g. prediction, ground truth, evaluation.
//   swap_axis: If false, the feature is plotted on the x axis. If true, the
//     feature is plotted on the y axis.
//   task: Model task e.g. classification, regression.
//   label_value_idx: For classification task and prediction and ground truth
//    target only. Should be set to -1 in all other cases. Value of interest of
//    the label.
//   attribute_type: Type of the input feature.
//   curve: Output curve.
//
// The measure for evaluation is the error rate for classification, and rmse for
// regression.
enum class CurveTargetType { kPrediction, kGroundTruth, kEvaluation };

absl::Status Set1DCurveData(
    const PartialDependencePlotSet::PartialDependencePlot& pdp,
    const CurveTargetType target_type, const bool swap_axis,
    const model::proto::Task task, const int label_value_idx,
    const dataset::proto::ColumnType attribute_type, plot::Curve* curve) {
  auto* feature_dst = &curve->xs;
  auto* target_dst = &curve->ys;
  if (swap_axis) {
    std::swap(feature_dst, target_dst);
  }
  for (int bin_idx = 0; bin_idx < pdp.pdp_bins_size(); bin_idx++) {
    const auto& bin = pdp.pdp_bins(bin_idx);
    switch (attribute_type) {
      case dataset::proto::ColumnType::NUMERICAL:
        STATUS_CHECK(bin.center_input_feature_values(0).has_numerical());
        feature_dst->push_back(bin.center_input_feature_values(0).numerical());
        break;
      case dataset::proto::ColumnType::CATEGORICAL:
        STATUS_CHECK(bin.center_input_feature_values(0).has_categorical());
        feature_dst->push_back(
            bin.center_input_feature_values(0).categorical());
        break;
      case dataset::proto::ColumnType::BOOLEAN:
        STATUS_CHECK(bin.center_input_feature_values(0).has_boolean());
        feature_dst->push_back(bin.center_input_feature_values(0).boolean());
        break;
      default:
        return absl::InvalidArgumentError("Non supported attribute type.");
    }
    switch (task) {
      case model::proto::Task::CLASSIFICATION:
        switch (target_type) {
          case CurveTargetType::kPrediction:
            STATUS_CHECK_GE(label_value_idx, 0);
            target_dst->push_back(GetDensityIntegerDistributionProto(
                bin.prediction().classification_class_distribution(),
                label_value_idx));
            break;
          case CurveTargetType::kGroundTruth:
            STATUS_CHECK_GE(label_value_idx, 0);
            target_dst->push_back(GetDensityIntegerDistributionProto(
                bin.ground_truth().classification_class_distribution(),
                label_value_idx));
            break;
          case CurveTargetType::kEvaluation:
            STATUS_CHECK_EQ(label_value_idx, -1);
            target_dst->push_back(
                1 -
                bin.evaluation().num_correct_predictions() /
                    bin.prediction().classification_class_distribution().sum());
            break;
        }
        break;
      case model::proto::Task::REGRESSION:
        STATUS_CHECK_EQ(label_value_idx, -1);
        switch (target_type) {
          case CurveTargetType::kPrediction:
            target_dst->push_back(
                bin.prediction().sum_of_regression_predictions() /
                pdp.num_observations());

            break;
          case CurveTargetType::kGroundTruth:
            target_dst->push_back(
                bin.ground_truth().sum_of_regression_predictions() /
                pdp.num_observations());

            break;
          case CurveTargetType::kEvaluation:
            target_dst->push_back(sqrt(bin.evaluation().sum_squared_error() /
                                       pdp.num_observations()));
            break;
        }
        break;

      case model::proto::Task::RANKING:
        STATUS_CHECK_EQ(label_value_idx, -1);
        switch (target_type) {
          case CurveTargetType::kPrediction:
            target_dst->push_back(
                bin.prediction().sum_of_ranking_predictions() /
                pdp.num_observations());
            break;
          default:
            return absl::InvalidArgumentError("Not implemented.");
        }
        break;

      default:
        return absl::InvalidArgumentError("Not implemented.");
    }
  }
  return absl::OkStatus();
}

// Plots the 1D pdp for a numerical variable.
absl::Status PlotPartialDependencePlot1DNumerical(
    const dataset::proto::DataSpecification& data_spec,
    const PartialDependencePlotSet::PartialDependencePlot& pdp,
    const int attribute_idx, const model::proto::Task task,
    const int label_col_idx, const bool plot_ground_truth, plot::Plot* pdp_plot,
    plot::Plot* density_plot) {
  STATUS_CHECK_EQ(pdp.attribute_info_size(), 1);
  STATUS_CHECK_EQ(pdp.attribute_info(0).num_bins_per_input_feature(),
                  pdp.pdp_bins_size());

  const auto& attr_spec = data_spec.columns(attribute_idx);
  const auto& label_spec = data_spec.columns(label_col_idx);

  STATUS_CHECK_EQ(attr_spec.type(), dataset::proto::ColumnType::NUMERICAL);

  pdp_plot->title = attr_spec.name();
  density_plot->title = attr_spec.name();

  pdp_plot->x_axis.label = attr_spec.name();
  pdp_plot->y_axis.label = kPredictionAxisLabel;

  if (pdp.attribute_info(0).scale() ==
      PartialDependencePlotSet::PartialDependencePlot::AttributeInfo::LOG) {
    pdp_plot->x_axis.scale = plot::AxisScale::LOG;
    density_plot->x_axis.scale = plot::AxisScale::LOG;
  }

  // PDP
  switch (task) {
    case model::proto::Task::CLASSIFICATION: {
      for (int label_value_idx =
               FirstCategoricalLabelValueForPdpPlot(label_spec);
           label_value_idx < label_spec.categorical().number_of_unique_values();
           label_value_idx++) {
        auto* prediction_curve = AddCurve(pdp_plot);

        prediction_curve->label =
            absl::StrCat(kPredictionLegendLabel, kKeyValueLegendSeparator,
                         dataset::CategoricalIdxToRepresentation(
                             label_spec, label_value_idx));
        prediction_curve->style = plot::LineStyle::SOLID;
        RETURN_IF_ERROR(Set1DCurveData(
            pdp, CurveTargetType::kPrediction, false,
            model::proto::Task::CLASSIFICATION, label_value_idx,
            dataset::proto::ColumnType::NUMERICAL, prediction_curve));

        if (plot_ground_truth) {
          auto* ground_truth_curve = AddCurve(pdp_plot);

          ground_truth_curve->style = plot::LineStyle::SOLID;
          ground_truth_curve->label =
              absl::StrCat(kGroundTruthLegendLabel, kKeyValueLegendSeparator,
                           dataset::CategoricalIdxToRepresentation(
                               label_spec, label_value_idx));
          RETURN_IF_ERROR(Set1DCurveData(
              pdp, CurveTargetType::kGroundTruth, false,
              model::proto::Task::CLASSIFICATION, label_value_idx,
              dataset::proto::ColumnType::NUMERICAL, ground_truth_curve));
        }
      }
      if (plot_ground_truth) {
        auto* error_rate_curve = AddCurve(pdp_plot);

        error_rate_curve->label = kErrorRateLegendLabel;
        error_rate_curve->style = plot::LineStyle::DOTTED;
        RETURN_IF_ERROR(Set1DCurveData(pdp, CurveTargetType::kEvaluation, false,
                                       model::proto::Task::CLASSIFICATION, -1,
                                       dataset::proto::ColumnType::NUMERICAL,
                                       error_rate_curve));
      }
    } break;

    case model::proto::Task::REGRESSION: {
      auto* prediction_curve = AddCurve(pdp_plot);

      prediction_curve->style = plot::LineStyle::SOLID;
      if (plot_ground_truth) {
        prediction_curve->label = kPredictionLegendLabel;
      }
      RETURN_IF_ERROR(Set1DCurveData(pdp, CurveTargetType::kPrediction, false,
                                     model::proto::Task::REGRESSION, -1,
                                     dataset::proto::ColumnType::NUMERICAL,
                                     prediction_curve));
      if (plot_ground_truth) {
        auto* ground_truth_curve = AddCurve(pdp_plot);

        ground_truth_curve->label = kGroundTruthLegendLabel;
        ground_truth_curve->style = plot::LineStyle::SOLID;
        RETURN_IF_ERROR(Set1DCurveData(
            pdp, CurveTargetType::kGroundTruth, false,
            model::proto::Task::REGRESSION, -1,
            dataset::proto::ColumnType::NUMERICAL, ground_truth_curve));

        auto* rmse_curve = AddCurve(pdp_plot);
        rmse_curve->label = kRMSELegendLabel;
        rmse_curve->style = plot::LineStyle::DOTTED;
        RETURN_IF_ERROR(Set1DCurveData(pdp, CurveTargetType::kEvaluation, false,
                                       model::proto::Task::REGRESSION, -1,
                                       dataset::proto::ColumnType::NUMERICAL,
                                       rmse_curve));
      }
    } break;

    case model::proto::Task::RANKING: {
      auto* prediction_curve = AddCurve(pdp_plot);

      prediction_curve->style = plot::LineStyle::SOLID;
      RETURN_IF_ERROR(Set1DCurveData(
          pdp, CurveTargetType::kPrediction, false, model::proto::Task::RANKING,
          -1, dataset::proto::ColumnType::NUMERICAL, prediction_curve));
    } break;

    default:
      return absl::InvalidArgumentError("Not implemented");
  }

  // Density
  density_plot->x_axis.label = attr_spec.name();
  density_plot->y_axis.label = kDensity_axisLabel;
  auto* density_curve = AddCurve(density_plot);
  auto* cumulative_density_curve = AddCurve(density_plot);
  density_curve->label = "density";
  cumulative_density_curve->label = "cdf";
  density_curve->style = plot::LineStyle::SOLID;
  cumulative_density_curve->style = plot::LineStyle::DOTTED;

  float max_y = 0;
  float sum_y = 0;
  for (int bin_idx = 0; bin_idx < pdp.pdp_bins_size(); bin_idx++) {
    const auto y = pdp.attribute_info(0).num_observations_per_bins(bin_idx);
    sum_y += y;
    if (y > max_y) {
      max_y = y;
    }
  }

  float cum_y = 0;
  for (int bin_idx = 0; bin_idx < pdp.pdp_bins_size(); bin_idx++) {
    const auto& bin = pdp.pdp_bins(bin_idx);
    const auto x = bin.center_input_feature_values(0).numerical();
    const auto y = pdp.attribute_info(0).num_observations_per_bins(bin_idx);

    density_curve->xs.push_back(x);
    density_curve->ys.push_back(y / max_y);

    if (bin_idx == 0) {
      cumulative_density_curve->xs.push_back(x);
      cumulative_density_curve->ys.push_back(0);
    }
    cum_y += y;
    cumulative_density_curve->xs.push_back(x);
    cumulative_density_curve->ys.push_back(cum_y / sum_y);
  }
  return absl::OkStatus();
}

absl::Status PlotPartialDependencePlot1DCategories(
    const dataset::proto::DataSpecification& data_spec,
    const PartialDependencePlotSet::PartialDependencePlot& pdp,
    const int attribute_idx, const model::proto::Task task,
    const int label_col_idx, const bool plot_ground_truth, plot::Plot* pdp_plot,
    bool swap_axes) {
  STATUS_CHECK_EQ(pdp.attribute_info_size(), 1);
  STATUS_CHECK_EQ(pdp.attribute_info(0).num_bins_per_input_feature(),
                  pdp.pdp_bins_size());

  const auto& attr_spec = data_spec.columns(attribute_idx);
  const auto& label_spec = data_spec.columns(label_col_idx);

  pdp_plot->title = absl::StrCat(attr_spec.name(), " (cat)");
  if (swap_axes) {
    pdp_plot->x_axis.label = kPredictionAxisLabel;
  } else {
    pdp_plot->y_axis.label = kPredictionAxisLabel;
  }

  // PDP
  switch (task) {
    case model::proto::Task::CLASSIFICATION: {
      for (int label_value_idx =
               FirstCategoricalLabelValueForPdpPlot(label_spec);
           label_value_idx < label_spec.categorical().number_of_unique_values();
           label_value_idx++) {
        auto* prediction_curve = AddCurve(pdp_plot);
        prediction_curve->label =
            absl::StrCat(kPredictionLegendLabel, kKeyValueLegendSeparator,
                         dataset::CategoricalIdxToRepresentation(
                             label_spec, label_value_idx));
        prediction_curve->style = plot::LineStyle::SOLID;
        RETURN_IF_ERROR(
            Set1DCurveData(pdp, CurveTargetType::kPrediction, swap_axes,
                           model::proto::Task::CLASSIFICATION, label_value_idx,
                           attr_spec.type(), prediction_curve));

        if (plot_ground_truth) {
          auto* ground_truth_curve = AddCurve(pdp_plot);
          ground_truth_curve->style = plot::LineStyle::SOLID;
          ground_truth_curve->label =
              absl::StrCat(kGroundTruthLegendLabel, kKeyValueLegendSeparator,
                           dataset::CategoricalIdxToRepresentation(
                               label_spec, label_value_idx));
          RETURN_IF_ERROR(Set1DCurveData(
              pdp, CurveTargetType::kGroundTruth, swap_axes,
              model::proto::Task::CLASSIFICATION, label_value_idx,
              attr_spec.type(), ground_truth_curve));
        }
      }
      if (plot_ground_truth) {
        auto* error_rate_curve = AddCurve(pdp_plot);
        error_rate_curve->label = "error rate";
        error_rate_curve->style = plot::LineStyle::DOTTED;
        RETURN_IF_ERROR(Set1DCurveData(pdp, CurveTargetType::kEvaluation,
                                       swap_axes,
                                       model::proto::Task::CLASSIFICATION, -1,
                                       attr_spec.type(), error_rate_curve));
      }
    } break;

    case model::proto::Task::REGRESSION: {
      auto* prediction_curve = AddCurve(pdp_plot);
      prediction_curve->style = plot::LineStyle::SOLID;
      if (plot_ground_truth) {
        prediction_curve->label = kPredictionLegendLabel;
      }
      RETURN_IF_ERROR(Set1DCurveData(pdp, CurveTargetType::kPrediction,
                                     swap_axes, model::proto::Task::REGRESSION,
                                     -1, attr_spec.type(), prediction_curve));
      if (plot_ground_truth) {
        auto* ground_truth_curve = AddCurve(pdp_plot);
        ground_truth_curve->style = plot::LineStyle::SOLID;
        ground_truth_curve->label = kGroundTruthLegendLabel;
        RETURN_IF_ERROR(Set1DCurveData(pdp, CurveTargetType::kGroundTruth,
                                       swap_axes,
                                       model::proto::Task::REGRESSION, -1,
                                       attr_spec.type(), ground_truth_curve));

        auto* rmse_curve = AddCurve(pdp_plot);
        rmse_curve->label = kRMSELegendLabel;
        rmse_curve->style = plot::LineStyle::DOTTED;
        RETURN_IF_ERROR(Set1DCurveData(
            pdp, CurveTargetType::kEvaluation, swap_axes,
            model::proto::Task::REGRESSION, -1, attr_spec.type(), rmse_curve));
      }
    } break;

    case model::proto::Task::RANKING: {
      auto* prediction_curve = AddCurve(pdp_plot);
      prediction_curve->style = plot::LineStyle::SOLID;
      RETURN_IF_ERROR(Set1DCurveData(pdp, CurveTargetType::kPrediction,
                                     swap_axes, model::proto::Task::RANKING, -1,
                                     attr_spec.type(), prediction_curve));
    } break;

    default:
      return absl::InvalidArgumentError("Not implemented");
  }
  return absl::OkStatus();
}

void SetCategoricalTicks(const dataset::proto::Column& column,
                         plot::Axis* axis) {
  axis->manual_tick_values = std::vector<double>();
  axis->manual_tick_texts = std::vector<std::string>();
  for (int categorical_attr_value_idx = 0;
       categorical_attr_value_idx <
       column.categorical().number_of_unique_values();
       categorical_attr_value_idx++) {
    axis->manual_tick_values->push_back(categorical_attr_value_idx);
    auto value_label = dataset::CategoricalIdxToRepresentation(
        column, categorical_attr_value_idx);
    if (value_label.empty()) {
      value_label = "NA";
    }
    axis->manual_tick_texts->push_back(value_label);
  }
}

// Plots the 1D pdp for a categorical variable.
absl::Status PlotPartialDependencePlot1DCategorical(
    const dataset::proto::DataSpecification& data_spec,
    const PartialDependencePlotSet::PartialDependencePlot& pdp,
    const int attribute_idx, const model::proto::Task task,
    const int label_col_idx, const bool plot_ground_truth, plot::Plot* pdp_plot,
    plot::Plot* density_plot) {
  const auto& attr_spec = data_spec.columns(attribute_idx);

  STATUS_CHECK_EQ(attr_spec.type(), dataset::proto::ColumnType::CATEGORICAL);

  // Set the y-axis tick labels with the possible values of the attribute.
  SetCategoricalTicks(attr_spec, &pdp_plot->y_axis);
  SetCategoricalTicks(attr_spec, &density_plot->y_axis);

  RETURN_IF_ERROR(PlotPartialDependencePlot1DCategories(
      data_spec, pdp, attribute_idx, task, label_col_idx, plot_ground_truth,
      pdp_plot, /* swap_axes= */ true));

  // Density
  density_plot->title = attr_spec.name();
  density_plot->x_axis.label = kDensity_axisLabel;
  auto* density_curve = AddCurve(density_plot);
  density_curve->style = plot::LineStyle::SOLID;

  for (int bin_idx = 0; bin_idx < pdp.pdp_bins_size(); bin_idx++) {
    const auto& bin = pdp.pdp_bins(bin_idx);
    density_curve->ys.push_back(
        bin.center_input_feature_values(0).categorical());
    density_curve->xs.push_back(
        pdp.attribute_info(0).num_observations_per_bins(bin_idx));
  }
  return absl::OkStatus();
}

void SetBooleanTicks(const dataset::proto::Column& column, plot::Axis* axis) {
  axis->manual_tick_values = std::vector<double>();
  axis->manual_tick_values->push_back(0);
  axis->manual_tick_values->push_back(1);

  axis->manual_tick_texts = std::vector<std::string>();
  axis->manual_tick_texts->push_back("false");
  axis->manual_tick_texts->push_back("true");
}

// Plot the 1D pdp for a boolean variable.
absl::Status PlotPartialDependencePlot1DBoolean(
    const dataset::proto::DataSpecification& data_spec,
    const PartialDependencePlotSet::PartialDependencePlot& pdp,
    const int attribute_idx, const model::proto::Task task,
    const int label_col_idx, const bool plot_ground_truth, plot::Plot* pdp_plot,
    plot::Plot* density_plot) {
  const auto& attr_spec = data_spec.columns(attribute_idx);

  STATUS_CHECK_EQ(attr_spec.type(), dataset::proto::ColumnType::BOOLEAN);

  SetBooleanTicks(attr_spec, &pdp_plot->x_axis);
  SetBooleanTicks(attr_spec, &density_plot->x_axis);

  RETURN_IF_ERROR(PlotPartialDependencePlot1DCategories(
      data_spec, pdp, attribute_idx, task, label_col_idx, plot_ground_truth,
      pdp_plot, /* swap_axes= */ false));

  // Density
  density_plot->title = absl::StrCat(attr_spec.name(), " (bool)");
  density_plot->y_axis.label = kDensity_axisLabel;
  auto* density_curve = AddCurve(density_plot);
  density_curve->style = plot::LineStyle::SOLID;

  for (int bin_idx = 0; bin_idx < pdp.pdp_bins_size(); bin_idx++) {
    const auto& bin = pdp.pdp_bins(bin_idx);
    density_curve->xs.push_back(bin.center_input_feature_values(0).boolean());
    density_curve->ys.push_back(
        pdp.attribute_info(0).num_observations_per_bins(bin_idx));
  }
  return absl::OkStatus();
}

absl::Status PlotPartialDependencePlot(
    const dataset::proto::DataSpecification& data_spec,
    const PartialDependencePlotSet::PartialDependencePlot& pdp,
    const model::proto::Task task, const int label_col_idx,
    const bool plot_ground_truth, plot::Plot* pdp_plot,
    plot::Plot* density_plot) {
  if (pdp.attribute_info_size() == 1) {
    // 1D pdp.
    const int attribute_idx = pdp.attribute_info(0).attribute_idx();
    const auto attribute_type = data_spec.columns(attribute_idx).type();
    if (attribute_type == dataset::proto::ColumnType::NUMERICAL) {
      RETURN_IF_ERROR(PlotPartialDependencePlot1DNumerical(
          data_spec, pdp, attribute_idx, task, label_col_idx, plot_ground_truth,
          pdp_plot, density_plot));
    } else if (attribute_type == dataset::proto::ColumnType::CATEGORICAL) {
      RETURN_IF_ERROR(PlotPartialDependencePlot1DCategorical(
          data_spec, pdp, attribute_idx, task, label_col_idx, plot_ground_truth,
          pdp_plot, density_plot));
    } else if (attribute_type == dataset::proto::ColumnType::BOOLEAN) {
      RETURN_IF_ERROR(PlotPartialDependencePlot1DBoolean(
          data_spec, pdp, attribute_idx, task, label_col_idx, plot_ground_truth,
          pdp_plot, density_plot));
    } else {
      return absl::InvalidArgumentError("Not implemented pdp");
    }
  } else if (pdp.attribute_info_size() == 2) {
    // 2D pdp.
    return absl::InvalidArgumentError(
        "Pdp not implemented for this 2-dimensional combination of "
        "attribute types.");
  } else {
    return absl::InvalidArgumentError(
        "Pdp not implemented for more than 2 attributes.");
  }
  return absl::OkStatus();
}

absl::StatusOr<utils::plot::MultiPlot> Plot(
    const dataset::proto::DataSpecification& data_spec,
    const PartialDependencePlotSet& pdp_set, const model::proto::Task task,
    const int label_col_idx, const bool plot_ground_truth,
    const proto::Options& options, int* recommended_width_px,
    int* recommended_height_px) {
  // Size of the individual plots on screen (in px).
  const int sub_plot_width = options.plot_width();
  const int sub_plot_height = options.plot_height();

  // Each attributes takes two rows of plots: one for the pdp plot, and one
  // for the density plot.
  const int num_rows_per_columns = 2;

  const int num_cols = std::max(1, options.figure_width() / sub_plot_width);
  const int num_rows =
      (pdp_set.pdps_size() + num_cols - 1) / num_cols * num_rows_per_columns;
  *recommended_width_px = sub_plot_width;
  *recommended_height_px = sub_plot_height;

  utils::plot::MultiPlot multiplot;
  multiplot.num_cols = num_cols;
  multiplot.num_rows = num_rows;

  for (int pdp_idx = 0; pdp_idx < pdp_set.pdps_size(); pdp_idx++) {
    multiplot.items.push_back(absl::make_unique<plot::MultiPlotItem>());
    auto* pdp_plot = multiplot.items.back().get();
    pdp_plot->col = pdp_idx % num_cols;
    pdp_plot->row = (pdp_idx / num_cols) * num_rows_per_columns;

    multiplot.items.push_back(absl::make_unique<plot::MultiPlotItem>());
    auto* density_plot = multiplot.items.back().get();
    density_plot->col = pdp_plot->col;
    density_plot->row = pdp_plot->row + 1;

    RETURN_IF_ERROR(PlotPartialDependencePlot(
        data_spec, pdp_set.pdps(pdp_idx), task, label_col_idx,
        plot_ground_truth, &pdp_plot->plot, &density_plot->plot));
  }
  RETURN_IF_ERROR(multiplot.Check());
  return multiplot;
}

}  // namespace

namespace internal {
absl::StatusOr<utils::plot::MultiPlot> PlotConditionalExpectationPlotSet(
    const dataset::proto::DataSpecification& data_spec,
    const ConditionalExpectationPlotSet& cond_set,
    const model::proto::Task task, const int label_col_idx,
    const proto::Options& options, int* recommended_width_px,
    int* recommended_height_px) {
  return Plot(data_spec, cond_set, task, label_col_idx, true, options,
              recommended_width_px, recommended_height_px);
}

absl::StatusOr<utils::plot::MultiPlot> PlotPartialDependencePlotSet(
    const dataset::proto::DataSpecification& data_spec,
    const PartialDependencePlotSet& pdp_set, const model::proto::Task task,
    const int label_col_idx, const proto::Options& options,
    int* recommended_width_px, int* recommended_height_px) {
  return Plot(data_spec, pdp_set, task, label_col_idx, false, options,
              recommended_width_px, recommended_height_px);
}

}  // namespace internal

absl::Status AnalyseAndCreateHtmlReport(const model::AbstractModel& model,
                                        const dataset::VerticalDataset& dataset,
                                        absl::string_view model_path,
                                        absl::string_view dataset_path,
                                        absl::string_view output_directory,
                                        const proto::Options& options) {
  ASSIGN_OR_RETURN(const auto analysis, Analyse(model, dataset, options));
  return CreateHtmlReport(model, dataset, model_path, dataset_path, analysis,
                          output_directory, options);
}

absl::Status AnalyseAndCreateHtmlReport(const model::AbstractModel& model,
                                        absl::string_view model_path,
                                        absl::string_view dataset_path,
                                        absl::string_view output_directory,
                                        const proto::Options& options) {
  YDF_LOG(INFO) << "Load dataset";
  dataset::VerticalDataset dataset;
  RETURN_IF_ERROR(dataset::LoadVerticalDataset(
      dataset_path, model.data_spec(), &dataset,
      /*ensure_non_missing=*/model.input_features()));

  return AnalyseAndCreateHtmlReport(model, dataset, model_path, dataset_path,
                                    output_directory, options);
}

absl::StatusOr<proto::AnalysisResult> Analyse(
    const model::AbstractModel& model, const dataset::VerticalDataset& dataset,
    const proto::Options& options) {
  proto::AnalysisResult analysis;

  if (dataset.nrow() == 0) {
    return absl::InvalidArgumentError("The dataset is empty.");
  }

  // Try to create a fast engine.
  const model::AbstractModel* effective_model = &model;
  auto engine_or = model.BuildFastEngine();
  std::unique_ptr<model::EngineWrapperModel> engine_wrapper;
  if (engine_or.ok()) {
    YDF_LOG(INFO) << "Run the model with the fast engine wrapper";
    engine_wrapper = std::make_unique<model::EngineWrapperModel>(
        &model, std::move(engine_or.value()));
    effective_model = engine_wrapper.get();
  } else {
    YDF_LOG(INFO)
        << "Run the model with the slow engine. No fast engine could be found: "
        << engine_or.status().message();
  }

  // Partial Dependence Plots
  if (options.pdp().enabled()) {
    ASSIGN_OR_RETURN(const auto attribute_idxs,
                     GenerateAttributesCombinations(
                         *effective_model, /*flag_1d=*/true,
                         /*flag_2d=*/false,
                         /*flag_2d_categorical_numerical=*/false));

    ASSIGN_OR_RETURN(*analysis.mutable_pdp_set(),
                     utils::ComputePartialDependencePlotSet(
                         dataset, *effective_model, attribute_idxs,
                         options.pdp().num_numerical_bins(),
                         options.pdp().example_sampling()));
  }

  // Conditional Expectation Plot
  if (options.cep().enabled()) {
    ASSIGN_OR_RETURN(const auto attribute_idxs,
                     GenerateAttributesCombinations(
                         *effective_model, /*flag_1d=*/true,
                         /*flag_2d=*/false,
                         /*flag_2d_categorical_numerical=*/false));

    ASSIGN_OR_RETURN(*analysis.mutable_cep_set(),
                     utils::ComputeConditionalExpectationPlotSet(
                         dataset, *effective_model, attribute_idxs,
                         options.cep().num_numerical_bins(),
                         options.cep().example_sampling()));
  }

  if (options.permuted_variable_importance().enabled()) {
    RETURN_IF_ERROR(ComputePermutationFeatureImportance(
        dataset, &model, analysis.mutable_variable_importances(),
        {options.num_threads(),
         options.permuted_variable_importance().num_rounds()}));
  }

  if (options.include_model_structural_variable_importances()) {
    auto& dst_map = *analysis.mutable_variable_importances();
    for (const auto& key : model.AvailableVariableImportances()) {
      ASSIGN_OR_RETURN(const auto src_values, model.GetVariableImportance(key));
      model::proto::VariableImportanceSet dst_values;
      *dst_values.mutable_variable_importances() = {src_values.begin(),
                                                    src_values.end()};
      dst_map[absl::StrCat("[In model] ", key)] = dst_values;
    }
  }

  return analysis;
}

absl::Status CreateHtmlReport(const model::AbstractModel& model,
                              const dataset::VerticalDataset& dataset,
                              absl::string_view model_path,
                              absl::string_view dataset_path,
                              const proto::AnalysisResult& analysis,
                              absl::string_view output_directory,
                              const proto::Options& options) {
  RETURN_IF_ERROR(
      file::RecursivelyCreateDir(output_directory, file::Defaults()));

  ASSIGN_OR_RETURN(const auto html_content,
                   CreateHtmlReport(model, dataset, model_path, dataset_path,
                                    analysis, options));

  RETURN_IF_ERROR(
      file::SetContent(file::JoinPath(output_directory, kIndex), html_content));
  YDF_LOG(INFO) << "Report written to " << output_directory;
  return absl::OkStatus();
}

absl::StatusOr<std::string> CreateHtmlReport(
    const model::AbstractModel& model, const dataset::VerticalDataset& dataset,
    const absl::string_view model_path, const absl::string_view dataset_path,
    const proto::AnalysisResult& analysis, const proto::Options& options) {
  const auto standalone = CreateStandaloneAnalysis(model, dataset, model_path,
                                                   dataset_path, analysis);
  return CreateHtmlReport(standalone, options);
}

proto::StandaloneAnalysisResult CreateStandaloneAnalysis(
    const model::AbstractModel& model, const dataset::VerticalDataset& dataset,
    const absl::string_view model_path, const absl::string_view dataset_path,
    const proto::AnalysisResult& analysis) {
  proto::StandaloneAnalysisResult standalone;
  *standalone.mutable_core_analysis() = analysis;
  standalone.set_dataset_path(std::string(model_path));
  standalone.set_model_path(std::string(dataset_path));
  *standalone.mutable_data_spec() = model.data_spec();
  standalone.set_label_col_idx(model.label_col_idx());
  standalone.set_task(model.task());
  standalone.set_model_description(model.DescriptionAndStatistics(false));
  return standalone;
}

absl::StatusOr<utils::html::Html> CreateHtmlReportPermutationVariableImportance(
    const proto::StandaloneAnalysisResult& analysis,
    const proto::Options& options, const absl::string_view block_id) {
  absl::flat_hash_map<std::string,
                      std::vector<model::proto::VariableImportance>>
      variable_importances;
  const auto& vis = analysis.core_analysis().variable_importances();
  for (const auto& vi : vis) {
    variable_importances[vi.first] = {vi.second.variable_importances().begin(),
                                      vi.second.variable_importances().end()};
  }
  return model::VariableImportance(variable_importances, analysis.data_spec(),
                                   absl::StrCat(block_id, "_vi"));
}

absl::StatusOr<std::string> CreateHtmlReport(
    const proto::StandaloneAnalysisResult& analysis,
    const proto::Options& options) {
  namespace h = utils::html;
  const auto block_id = utils::GenUniqueId();

  h::Html html;
  html.AppendRaw(model::Header());

  // Report header.
  if (options.report_header().enabled()) {
    h::Html report_header;
    report_header.Append(h::H1("Model Analysis"));
    report_header.Append(h::P(absl::FormatTime(absl::RFC3339_sec, absl::Now(),
                                               absl::LocalTimeZone())));
    report_header.Append(
        h::P("Report generated by Yggdrasil Decision Forests"));
    html.Append(report_header);
  }

  h::Html tab_header;
  h::Html tab_content;

  // Adds a tab to the page.
  utils::TabBarBuilder tabbar;

  // Setup
  if (options.report_setup().enabled()) {
    h::Html content;
    content.Append(h::P(h::B("Analyse dataset: "), analysis.dataset_path()));
    content.Append(h::P(h::B("Model: "), analysis.model_path()));
    tabbar.AddTab("setup", "Setup", content);
  }

  // Dataset Specification
  if (options.report_setup().enabled() && analysis.has_data_spec()) {
    h::Html content;
    content.Append(
        h::Pre(dataset::PrintHumanReadable(analysis.data_spec(), false)));
    tabbar.AddTab("dataset", "Dataset", content);
  }

  // Partial Dependence Plot
  if (options.pdp().enabled() && analysis.core_analysis().has_pdp_set()) {
    utils::plot::ExportOptions plot_options;
    plot_options.show_interactive_menu = options.plot().show_interactive_menu();
    ASSIGN_OR_RETURN(
        const auto plot,
        internal::PlotPartialDependencePlotSet(
            analysis.data_spec(), analysis.core_analysis().pdp_set(),
            analysis.task(), analysis.label_col_idx(), options,
            &plot_options.width, &plot_options.height));
    ASSIGN_OR_RETURN(const auto multiplot_html,
                     utils::plot::ExportToHtml(plot, plot_options));

    h::Html content;
    content.AppendRaw(multiplot_html);
    tabbar.AddTab("pdp", "Partial Dependence Plot", content);
  }

  // Conditional Expectation Plot
  if (options.cep().enabled() && analysis.core_analysis().has_cep_set()) {
    utils::plot::ExportOptions plot_options;
    plot_options.show_interactive_menu = options.plot().show_interactive_menu();
    ASSIGN_OR_RETURN(
        const auto plot,
        internal::PlotConditionalExpectationPlotSet(
            analysis.data_spec(), analysis.core_analysis().cep_set(),
            analysis.task(), analysis.label_col_idx(), options,
            &plot_options.width, &plot_options.height));
    ASSIGN_OR_RETURN(const auto multiplot_html,
                     utils::plot::ExportToHtml(plot, plot_options));

    h::Html content;
    content.AppendRaw(multiplot_html);
    tabbar.AddTab("cep", "Conditional Expectation Plot", content);
  }

  // Permutation Variable Importance
  if (options.permuted_variable_importance().enabled() &&
      analysis.core_analysis().variable_importances_size() > 0) {
    ASSIGN_OR_RETURN(
        const auto content,
        CreateHtmlReportPermutationVariableImportance(
            analysis, options, absl::StrCat(block_id, "_variable_importance")));
    tabbar.AddTab("pva", "Variable Importances", content);
  }

  // Model Description
  if (options.model_description().enabled() &&
      analysis.has_model_description()) {
    h::Html content;
    content.Append(h::Pre(analysis.model_description()));
    tabbar.AddTab("model", "Model Description", content);
  }

  html.Append(tabbar.Html());
  return std::string(html.content());
}

}  // namespace model_analysis
}  // namespace utils
}  // namespace yggdrasil_decision_forests
