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

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/builder.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/model_testing.h"
#include "yggdrasil_decision_forests/model/random_forest/random_forest.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/model_analysis.pb.h"
#include "yggdrasil_decision_forests/utils/plot.h"
#include "yggdrasil_decision_forests/utils/test.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace model_analysis {
namespace {

using test::ApproximatelyEqualsProto;

std::string DatasetDir() {
  return file::JoinPath(
      test::DataRootDirectory(),
      "yggdrasil_decision_forests/test_data/dataset");
}

std::string ModelDir() {
  return file::JoinPath(
      test::DataRootDirectory(),
      "yggdrasil_decision_forests/test_data/model");
}

// Note: The old golden models don't contain the necessary meta-data (the number
// of training examples per nodes) to compute SHAP values. For this reason,
// tests like "ClassificationGodenModel" don't show SHAP variable importances.
TEST(ModelAnalysis, ClassificationGodenModel) {
  const std::string dataset_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "adult_test.csv"));
  const std::string model_path =
      file::JoinPath(ModelDir(), "adult_binary_class_gbdt");

  LOG(INFO) << "Load model";
  std::unique_ptr<model::AbstractModel> model;
  CHECK_OK(model::LoadModel(model_path, &model));

  LOG(INFO) << "Load dataset";
  dataset::VerticalDataset dataset;
  CHECK_OK(dataset::LoadVerticalDataset(
      dataset_path, model->data_spec(), &dataset,
      /*required_columns=*/model->input_features()));

  proto::Options options;
  options.mutable_pdp()->set_example_sampling(0.01f);
  options.mutable_cep()->set_example_sampling(0.1f);
  options.set_num_threads(1);
  options.set_html_id_prefix("my_report");
  options.mutable_report_header()->set_enabled(false);
  const auto report_path = file::JoinPath(test::TmpDirectory(), "analysis");

  ASSERT_OK_AND_ASSIGN(const auto analysis,
                       Analyse(*model.get(), dataset, options));
  ASSERT_OK_AND_ASSIGN(const auto report,
                       CreateHtmlReport(*model.get(), dataset, "MODEL_PATH",
                                        "DATASET_PATH", analysis, options));
}

// Train a model and compute the full analysis. This test includes SHAP variable
// importances.
TEST(ModelAnalysis, ClassificationWithTraining) {
  LOG(INFO) << "Load dataset";
  const std::string dataset_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "adult_test.csv"));
  const auto dataspec = dataset::CreateDataSpec(dataset_path).value();
  dataset::VerticalDataset dataset;
  CHECK_OK(dataset::LoadVerticalDataset(dataset_path, dataspec, &dataset));

  LOG(INFO) << "Train model";
  auto model = model::GetLearner(PARSE_TEST_PROTO(R"pb(
                 task: CLASSIFICATION
                 label: "income"
                 learner: "GRADIENT_BOOSTED_TREES"
                 [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
                      .gradient_boosted_trees_config] { num_trees: 20 }
               )pb"))
                   .value()
                   ->TrainWithStatus(dataset)
                   .value();

  proto::Options options;
  options.mutable_pdp()->set_example_sampling(0.01f);
  options.mutable_cep()->set_example_sampling(0.1f);
  options.set_num_threads(1);
  options.set_html_id_prefix("my_report");
  options.mutable_report_header()->set_enabled(false);
  options.mutable_shap_variable_importance()->set_example_sampling(0.1);
  const auto report_path = file::JoinPath(test::TmpDirectory(), "analysis");

  ASSERT_OK_AND_ASSIGN(const auto analysis,
                       Analyse(*model.get(), dataset, options));
  ASSERT_OK_AND_ASSIGN(const auto report,
                       CreateHtmlReport(*model.get(), dataset, "MODEL_PATH",
                                        "DATASET_PATH", analysis, options));
}

TEST(ModelAnalysis, Regression) {
  const std::string dataset_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "abalone.csv"));
  const std::string model_path =
      file::JoinPath(ModelDir(), "abalone_regression_gbdt");

  std::unique_ptr<model::AbstractModel> model;
  ASSERT_OK(model::LoadModel(model_path, &model));

  dataset::VerticalDataset dataset;
  ASSERT_OK(dataset::LoadVerticalDataset(
      dataset_path, model->data_spec(), &dataset,
      /*required_columns=*/model->input_features()));

  proto::Options options;
  options.mutable_pdp()->set_example_sampling(0.01f);
  options.mutable_cep()->set_example_sampling(0.1f);
  options.set_num_threads(1);
  options.set_html_id_prefix("my_report");
  options.mutable_report_header()->set_enabled(false);
  const auto report_path = file::JoinPath(test::TmpDirectory(), "analysis");

  ASSERT_OK_AND_ASSIGN(const auto analysis,
                       Analyse(*model.get(), dataset, options));
  ASSERT_OK_AND_ASSIGN(const auto report,
                       CreateHtmlReport(*model.get(), dataset, "MODEL_PATH",
                                        "DATASET_PATH", analysis, options));
}

TEST(ModelAnalysis, AnomalyDetection) {
  const std::string dataset_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "gaussians_test.csv"));
  const std::string model_path =
      file::JoinPath(ModelDir(), "gaussians_anomaly_if");

  ASSERT_OK_AND_ASSIGN(const auto model, model::LoadModel(model_path));

  dataset::VerticalDataset dataset;
  ASSERT_OK(dataset::LoadVerticalDataset(
      dataset_path, model->data_spec(), &dataset,
      /*required_columns=*/model->input_features()));

  proto::Options options;
  options.mutable_pdp()->set_example_sampling(0.01f);
  options.mutable_cep()->set_example_sampling(0.1f);
  options.set_num_threads(1);
  options.set_html_id_prefix("my_report");
  options.mutable_report_header()->set_enabled(false);
  options.mutable_permuted_variable_importance()->set_enabled(false);
  const auto report_path = file::JoinPath(test::TmpDirectory(), "analysis");

  ASSERT_OK_AND_ASSIGN(const auto analysis,
                       Analyse(*model.get(), dataset, options));

  ASSERT_OK_AND_ASSIGN(const auto report,
                       CreateHtmlReport(*model.get(), dataset, "MODEL_PATH",
                                        "DATASET_PATH", analysis, options));
}

TEST(ModelAnalysis, FailsWithEmptyDataset) {
  const std::string dataset_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "adult_test.csv"));
  const std::string model_path =
      file::JoinPath(ModelDir(), "adult_binary_class_gbdt");
  std::unique_ptr<model::AbstractModel> model;
  CHECK_OK(model::LoadModel(model_path, &model));
  dataset::VerticalDataset dataset;

  proto::Options options;
  const auto report_path = file::JoinPath(test::TmpDirectory(), "analysis");
  EXPECT_FALSE(AnalyseAndCreateHtmlReport(*model.get(), dataset, model_path,
                                          dataset_path, report_path, options)
                   .ok());
}

TEST(ModelAnalysis, PDPPlot) {
  dataset::proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"pb(
        columns {
          name: "f"
          type: NUMERICAL
          numerical { min_value: 0 max_value: 1000 mean: 500 }
        }
        columns {
          type: NUMERICAL
          name: "l"
          numerical { min_value: 0 max_value: 1000 mean: 500 }
        }
        columns {
          name: "f2"
          type: NUMERICAL
          numerical { min_value: 0 max_value: 1000 mean: 500 }
        }
      )pb");

  dataset::VerticalDataset dataset;
  *dataset.mutable_data_spec() = data_spec;
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  for (int i = 0; i < 1000; i++) {
    std::unordered_map<std::string, std::string> example{
        {"l", absl::StrCat(i)},
        {"f2", absl::StrCat(i / 2)},
    };
    if ((i % 2) == 0) {
      example["f"] = absl::StrCat(i);
    }
    CHECK_OK(dataset.AppendExampleWithStatus(example));
  }

  class FakeModel : public model::FakeModel {
   public:
    void Predict(const dataset::proto::Example& example,
                 model::proto::Prediction* prediction) const override {
      prediction->mutable_regression()->set_value(
          Prediction(example.attributes(0).numerical()));
    }

    void Predict(const dataset::VerticalDataset& dataset,
                 dataset::VerticalDataset::row_t row_idx,
                 model::proto::Prediction* prediction) const override {
      const float feature_value =
          dataset
              .ColumnWithCastWithStatus<
                  dataset::VerticalDataset::NumericalColumn>(0)
              .value()
              ->values()[row_idx];
      prediction->mutable_regression()->set_value(Prediction(feature_value));
    }

   private:
    float Prediction(const float feature) const {
      return std::abs((feature - 500) / 500);
    }
  };
  FakeModel model;
  model.mutable_input_features()->push_back(0);
  model.mutable_input_features()->push_back(2);
  model.set_task(model::proto::Task::REGRESSION);
  model.set_label_col_idx(1);
  *model.mutable_data_spec() = data_spec;

  proto::Options options;
  auto analysis = Analyse(model, dataset, options).value();

  int width, height;
  const auto plot = internal::PlotPartialDependencePlotSet(
                        data_spec, analysis.pdp_set(), model.task(),
                        model.label_col_idx(), options, &width, &height)
                        .value();

  EXPECT_EQ(plot.num_cols, 3);
  EXPECT_EQ(plot.num_rows, 2);
  EXPECT_EQ(plot.items.size(), 4);

  EXPECT_EQ(plot.items.front()->col, 0);
  EXPECT_EQ(plot.items.front()->row, 0);
  EXPECT_EQ(plot.items.front()->num_cols, 1);
  EXPECT_EQ(plot.items.front()->num_rows, 1);

  EXPECT_EQ(plot.items.back()->col, 1);
  EXPECT_EQ(plot.items.back()->row, 1);
  EXPECT_EQ(plot.items.back()->num_cols, 1);
  EXPECT_EQ(plot.items.back()->num_rows, 1);

  auto& p1 = plot.items.front()->plot;
  EXPECT_EQ(p1.title, "f");
  EXPECT_EQ(p1.items.size(), 1);
  auto* c1 = dynamic_cast<plot::Curve*>(p1.items.front().get());
  EXPECT_EQ(c1->xs.size(), 50);
  EXPECT_EQ(c1->ys.size(), 50);

  // TODO: Add a more extensive unit test with a golden report.
}

TEST(ModelAnalysis, ShapValuesOnRegression) {
  // Build a dataset
  dataset::VerticalDataset dataset;
  dataset::AddNumericalColumn("f1", dataset.mutable_data_spec());
  dataset::AddNumericalColumn("f2", dataset.mutable_data_spec());
  dataset::AddNumericalColumn("l", dataset.mutable_data_spec());
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  // The dataset is large to show that sampling / max time is applied.
  for (int i = 0; i < 1'000'000; i++) {
    const float f1 = static_cast<float>(i % 1000) / 1000;
    const float f2 = static_cast<float>((i * 793) % 1000) / 1000;
    const float l = f1 - 0.5 * f2;
    CHECK_OK(dataset.AppendExampleWithStatus({
        {"l", absl::StrCat(l)},
        {"f1", absl::StrCat(f1)},
        {"f2", absl::StrCat(f2)},
    }));
  }

  auto model = model::GetLearner(PARSE_TEST_PROTO(R"pb(
                 task: REGRESSION
                 label: "l"
                 learner: "GRADIENT_BOOSTED_TREES"
                 [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
                      .gradient_boosted_trees_config] {
                   num_trees: 20
                   subsample: 0.001
                   validation_set_ratio: 0.001
                 }
               )pb"))
                   .value()
                   ->TrainWithStatus(dataset)
                   .value();

  proto::Options options;
  options.mutable_pdp()->set_enabled(false);
  options.mutable_cep()->set_enabled(false);
  options.mutable_permuted_variable_importance()->set_enabled(false);
  options.mutable_shap_variable_importance()->set_example_sampling(0.5);
  options.set_maximum_duration_seconds(5);
  auto analysis = Analyse(*model, dataset, options).value();

  // The model output predict := f1 - 0.5 x f2, so the VI of f1 (attribute 0) is
  // twice the VI of f2 (attribute 1).
  const auto& vas = analysis.variable_importances().find("SHAP_VALUE")->second;
  ASSERT_EQ(vas.variable_importances_size(), 2);
  EXPECT_EQ(vas.variable_importances(0).attribute_idx(), 0);
  EXPECT_NEAR(vas.variable_importances(0).importance(), 0.2, 0.03);
  EXPECT_EQ(vas.variable_importances(1).attribute_idx(), 1);
  EXPECT_NEAR(vas.variable_importances(1).importance(), 0.1, 0.03);
}

TEST(ModelAnalysis, ShapValuesOnMultiClassClassification) {
  // Build a dataset
  dataset::VerticalDataset dataset;
  dataset::AddNumericalColumn("f1", dataset.mutable_data_spec());
  dataset::AddNumericalColumn("f2", dataset.mutable_data_spec());
  dataset::AddCategoricalColumn("l", {"X", "Y", "Z"},
                                dataset.mutable_data_spec());
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  // The dataset is large to show that sampling / max time is applied.
  for (int i = 0; i < 2000; i++) {
    const float f1 = static_cast<float>(i % 1000) / 1000;
    const float f2 = static_cast<float>((i * 793) % 1000) / 1000;
    std::string label;
    if (f2 >= 0.5f) {
      label = "X";
    } else if (f1 >= 0.5f) {
      label = "Y";
    } else {
      label = "Z";
    }
    CHECK_OK(dataset.AppendExampleWithStatus({
        {"l", label},
        {"f1", absl::StrCat(f1)},
        {"f2", absl::StrCat(f2)},
    }));
  }

  auto model = model::GetLearner(PARSE_TEST_PROTO(R"pb(
                 task: CLASSIFICATION
                 label: "l"
                 learner: "GRADIENT_BOOSTED_TREES"
                 [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
                      .gradient_boosted_trees_config] { num_trees: 20 }
               )pb"))
                   .value()
                   ->TrainWithStatus(dataset)
                   .value();

  proto::Options options;
  options.mutable_pdp()->set_enabled(false);
  options.mutable_cep()->set_enabled(false);
  options.mutable_permuted_variable_importance()->set_enabled(false);
  options.set_maximum_duration_seconds(5);
  auto analysis = Analyse(*model, dataset, options).value();

  // Make sure that both features are important.
  const auto& vas = analysis.variable_importances().find("SHAP_VALUE")->second;
  ASSERT_EQ(vas.variable_importances_size(), 2);
  EXPECT_EQ(vas.variable_importances(0).attribute_idx(), 1);
  EXPECT_NEAR(vas.variable_importances(0).importance(), 4.937, 0.03);
  EXPECT_EQ(vas.variable_importances(1).attribute_idx(), 0);
  EXPECT_NEAR(vas.variable_importances(1).importance(), 2.483, 0.03);
}

TEST(PredictionAnalysis, Classification) {
  const std::string dataset_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "adult_test.csv"));
  const std::string model_path =
      file::JoinPath(ModelDir(), "adult_binary_class_gbdt");

  LOG(INFO) << "Load model";
  std::unique_ptr<model::AbstractModel> model;
  ASSERT_OK(model::LoadModel(model_path, &model));

  LOG(INFO) << "Load dataset";
  dataset::VerticalDataset dataset;
  ASSERT_OK(dataset::LoadVerticalDataset(
      dataset_path, model->data_spec(), &dataset,
      /*required_columns=*/model->input_features()));

  proto::PredictionAnalysisOptions options;
  options.set_html_id_prefix("my_prefix");
  dataset::proto::Example example;
  dataset.ExtractExample(0, &example);
  ASSERT_OK_AND_ASSIGN(const auto analysis,
                       AnalyzePrediction(*model, example, options));
  ASSERT_OK_AND_ASSIGN(const auto report, CreateHtmlReport(analysis, options));
}

TEST(PredictionAnalysis, ClassificationWithTraining) {
  LOG(INFO) << "Load dataset";
  const std::string dataset_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "adult_train.csv"));
  const auto dataspec = dataset::CreateDataSpec(dataset_path).value();
  dataset::VerticalDataset dataset;
  CHECK_OK(dataset::LoadVerticalDataset(dataset_path, dataspec, &dataset));

  LOG(INFO) << "Train model";
  auto model = model::GetLearner(PARSE_TEST_PROTO(R"pb(
                 task: CLASSIFICATION
                 label: "income"
                 learner: "GRADIENT_BOOSTED_TREES"
                 [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
                      .gradient_boosted_trees_config] { num_trees: 20 }
               )pb"))
                   .value()
                   ->TrainWithStatus(dataset)
                   .value();

  proto::PredictionAnalysisOptions options;
  options.set_html_id_prefix("my_prefix");
  dataset::proto::Example example;
  dataset.ExtractExample(0, &example);
  ASSERT_OK_AND_ASSIGN(const auto analysis,
                       AnalyzePrediction(*model, example, options));
  ASSERT_OK_AND_ASSIGN(const auto report, CreateHtmlReport(analysis, options));

  EXPECT_TRUE(absl::StrContains(report, "SHAP values"));
  EXPECT_TRUE(absl::StrContains(report, "      |+++++"));
}

TEST(PredictionAnalysis, ClassificationMultiClassWithTraining) {
  LOG(INFO) << "Load dataset";
  const std::string dataset_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "iris.csv"));
  const auto dataspec = dataset::CreateDataSpec(dataset_path).value();
  dataset::VerticalDataset dataset;
  CHECK_OK(dataset::LoadVerticalDataset(dataset_path, dataspec, &dataset));

  LOG(INFO) << "Train model";
  auto model = model::GetLearner(PARSE_TEST_PROTO(R"pb(
                 task: CLASSIFICATION
                 label: "class"
                 learner: "GRADIENT_BOOSTED_TREES"
                 [yggdrasil_decision_forests.model.gradient_boosted_trees.proto
                      .gradient_boosted_trees_config] { num_trees: 20 }
               )pb"))
                   .value()
                   ->TrainWithStatus(dataset)
                   .value();

  proto::PredictionAnalysisOptions options;
  options.set_html_id_prefix("my_prefix");
  dataset::proto::Example example;
  dataset.ExtractExample(0, &example);
  ASSERT_OK_AND_ASSIGN(const auto analysis,
                       AnalyzePrediction(*model, example, options));
  ASSERT_OK_AND_ASSIGN(const auto report, CreateHtmlReport(analysis, options));

  EXPECT_TRUE(absl::StrContains(report, "SHAP values"));
  EXPECT_TRUE(absl::StrContains(report, "      |+++++"));
}

TEST(PredictionAnalysis, AnomalyDetection) {
  const std::string dataset_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "gaussians_test.csv"));
  const std::string model_path =
      file::JoinPath(ModelDir(), "gaussians_anomaly_if");

  ASSERT_OK_AND_ASSIGN(const auto model, model::LoadModel(model_path));

  dataset::VerticalDataset dataset;
  ASSERT_OK(dataset::LoadVerticalDataset(
      dataset_path, model->data_spec(), &dataset,
      /*required_columns=*/model->input_features()));

  proto::PredictionAnalysisOptions options;
  options.set_html_id_prefix("my_prefix");
  dataset::proto::Example example;
  dataset.ExtractExample(0, &example);
  ASSERT_OK_AND_ASSIGN(const auto analysis,
                       AnalyzePrediction(*model, example, options));
  ASSERT_OK_AND_ASSIGN(const auto report, CreateHtmlReport(analysis, options));
}

TEST(PredictionAnalysis, ToyModel) {
  // Create a model.
  dataset::proto::DataSpecification dataspec;
  dataset::AddColumn("l", dataset::proto::ColumnType::NUMERICAL, &dataspec);
  auto* f1 =
      dataset::AddColumn("f1", dataset::proto::ColumnType::NUMERICAL, &dataspec)
          ->mutable_numerical();
  f1->set_min_value(1);
  f1->set_max_value(5);
  auto* f2 = dataset::AddColumn("f2", dataset::proto::ColumnType::CATEGORICAL,
                                &dataspec)
                 ->mutable_categorical();
  f2->set_number_of_unique_values(5);
  f2->set_is_already_integerized(true);

  dataset::proto::Example example;
  example.add_attributes()->set_numerical(1);
  example.add_attributes()->set_numerical(2);
  example.add_attributes()->set_categorical(2);

  model::random_forest::RandomForestModel model;

  {
    auto tree = std::make_unique<model::decision_tree::DecisionTree>();
    model::decision_tree::TreeBuilder root(tree.get());
    auto [l1, neg] = root.ConditionIsGreater(1, 3);
    l1.LeafRegression(1);
    auto [l2, l3] = neg.ConditionContains(2, {2, 3});
    l2.LeafRegression(2);
    l3.LeafRegression(3);
    model.AddTree(std::move(tree));
  }

  model.set_task(model::proto::Task::REGRESSION);
  model.set_label_col_idx(0);
  model.set_data_spec(dataspec);
  model.mutable_input_features()->push_back(1);
  model.mutable_input_features()->push_back(2);

  proto::PredictionAnalysisOptions options;
  options.set_html_id_prefix("my_report");
  options.set_numerical_num_bins(4);
  ASSERT_OK_AND_ASSIGN(const auto analysis,
                       AnalyzePrediction(model, example, options));

  EXPECT_THAT(analysis,
              ApproximatelyEqualsProto(PARSE_TEST_PROTO_WITH_TYPE(
                  proto::PredictionAnalysisResult, R"pb(
                    data_spec {
                      columns { type: NUMERICAL name: "l" }
                      columns {
                        type: NUMERICAL
                        name: "f1"
                        numerical { min_value: 1 max_value: 5 }
                      }
                      columns {
                        type: CATEGORICAL
                        name: "f2"
                        categorical {
                          number_of_unique_values: 5
                          is_already_integerized: true
                        }
                      }
                    }
                    label_col_idx: 0
                    task: REGRESSION
                    feature_variation {
                      items {
                        bins { prediction { regression { value: 2 } } }
                        bins { prediction { regression { value: 2 } } }
                        bins { prediction { regression { value: 1 } } }
                        bins { prediction { regression { value: 1 } } }
                        attributes {
                          column_idx: 1
                          numerical {
                            values: 1
                            values: 2.3333335
                            values: 3.6666667
                            values: 5
                          }
                        }
                      }
                      items {
                        bins { prediction { regression { value: 3 } } }
                        bins { prediction { regression { value: 3 } } }
                        bins { prediction { regression { value: 2 } } }
                        bins { prediction { regression { value: 2 } } }
                        bins { prediction { regression { value: 3 } } }
                        attributes {
                          column_idx: 2
                          categorical { num_values: 5 }
                        }
                      }
                    }
                    example {
                      attributes { numerical: 1 }
                      attributes { numerical: 2 }
                      attributes { categorical: 2 }
                    }
                    prediction { regression { value: 2 } }
                  )pb")));

  ASSERT_OK_AND_ASSIGN(const auto report, CreateHtmlReport(analysis, options));
}

TEST(PredictionAnalysis, ShapRepr) {
  EXPECT_EQ(internal::ShapRepr(4, 0, 4), "    |    ");
  EXPECT_EQ(internal::ShapRepr(4, 4, 4), "    |++++");
  EXPECT_EQ(internal::ShapRepr(4, 2, 4), "    |++  ");
  EXPECT_EQ(internal::ShapRepr(4, -2, 4), "  --|    ");
  EXPECT_EQ(internal::ShapRepr(4, -4, 4), "----|    ");
  EXPECT_EQ(internal::ShapRepr(0, 0, 4), "    |    ");
}

}  // namespace
}  // namespace model_analysis
}  // namespace utils
}  // namespace yggdrasil_decision_forests
