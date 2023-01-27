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

#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/model_testing.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/plot.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace model_analysis {
namespace {

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

TEST(ModelAnalysis, Basic) {
  const std::string dataset_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "adult_test.csv"));
  const std::string model_path =
      file::JoinPath(ModelDir(), "adult_binary_class_gbdt");

  YDF_LOG(INFO) << "Load model";
  std::unique_ptr<model::AbstractModel> model;
  CHECK_OK(model::LoadModel(model_path, &model));

  YDF_LOG(INFO) << "Load dataset";
  dataset::VerticalDataset dataset;
  CHECK_OK(dataset::LoadVerticalDataset(
      dataset_path, model->data_spec(), &dataset,
      /*ensure_non_missing=*/model->input_features()));

  proto::Options options;
  options.mutable_pdp()->set_example_sampling(0.01f);
  options.mutable_cep()->set_example_sampling(0.1f);
  const auto report_path = file::JoinPath(test::TmpDirectory(), "analysis");
  CHECK_OK(AnalyseAndCreateHtmlReport(*model.get(), dataset, model_path,
                                      dataset_path, report_path, options));
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

}  // namespace
}  // namespace model_analysis
}  // namespace utils
}  // namespace yggdrasil_decision_forests
