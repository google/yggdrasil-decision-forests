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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/serving/decision_forest/decision_forest.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

#include "yggdrasil_decision_forests/serving/utils.h"

namespace yggdrasil_decision_forests {
namespace serving {
namespace {

using test::EqualsProto;

std::string TestDataDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/test_data");
}

// Load a dataset
dataset::VerticalDataset LoadDataset(
    const dataset::proto::DataSpecification& data_spec,
    const absl::string_view dataset_filename,
    const absl::string_view format = "csv") {
  const std::string ds_typed_path = absl::StrCat(
      format, ":", file::JoinPath(TestDataDir(), "dataset", dataset_filename));
  dataset::VerticalDataset dataset;
  CHECK_OK(LoadVerticalDataset(ds_typed_path, data_spec, &dataset));
  return dataset;
}

// Load a model.
std::unique_ptr<model::AbstractModel> LoadModel(
    const absl::string_view model_dirname) {
  const std::string model_dir =
      file::JoinPath(TestDataDir(), "model", model_dirname);
  std::unique_ptr<model::AbstractModel> model;
  CHECK_OK(model::LoadModel(model_dir, &model));
  return model;
}

void feature_statistics_toy_example(const ExampleFormat format) {
  dataset::VerticalDataset dataset;
  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"pb(
    created_num_rows: 10
    columns { type: NUMERICAL name: "a" }
    columns { type: NUMERICAL name: "b" }
    columns {
      type: CATEGORICAL
      name: "c"
      categorical { is_already_integerized: true number_of_unique_values: 4 }
    }
  )pb");
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  dataset.AppendExample({{"a", "1.0"}, {"b", "2.0"}, {"c", "1"}});
  dataset.AppendExample({{"a", "2.0"}, {"b", "3.0"}, {"c", "2"}});
  dataset.AppendExample({{"a", "3.0"}});

  std::vector<NumericalOrCategoricalValue> replacement_values = {
      NumericalOrCategoricalValue::Numerical(-1),
      NumericalOrCategoricalValue::Numerical(-1),
      NumericalOrCategoricalValue::Categorical(0)};
  FeatureStatistics stats(&dataset.data_spec(), {0, 1, 2}, replacement_values);

  std::vector<NumericalOrCategoricalValue> batch_1;
  CHECK_OK(decision_forest::LoadFlatBatchFromDataset(
      dataset, 0, 3, {"a", "b", "c"}, replacement_values, &batch_1, format));

  stats.Update(batch_1, 3, format);
  LOG(INFO) << "Report 1:\n" << stats.BuildReport();
  const proto::FeatureStatistics expected_1 = PARSE_TEST_PROTO(
      R"(
        features {
          num_non_missing: 3
          numerical { sum: 6 sum_squared: 14 min: 1 max: 3 }
        }
        features {
          num_non_missing: 2
          numerical { sum: 5 sum_squared: 13 min: 2 max: 3 }
        }
        features {
          num_non_missing: 2
          categorical {
            count_per_value { key: 0 value: 1 }
            count_per_value { key: 1 value: 1 }
            count_per_value { key: 2 value: 1 }
          }
        }
        num_examples: 3
      )");
  EXPECT_THAT(stats.Export(), EqualsProto(expected_1));

  stats.Update(batch_1, 3, format);
  LOG(INFO) << "Report 2:\n" << stats.BuildReport();
  const proto::FeatureStatistics expected_2 = PARSE_TEST_PROTO(
      R"(
        features {
          num_non_missing: 6
          numerical { sum: 12 sum_squared: 28 min: 1 max: 3 }
        }
        features {
          num_non_missing: 4
          numerical { sum: 10 sum_squared: 26 min: 2 max: 3 }
        }
        features {
          num_non_missing: 4
          categorical {
            count_per_value { key: 0 value: 2 }
            count_per_value { key: 1 value: 2 }
            count_per_value { key: 2 value: 2 }
          }
        }
        num_examples: 6
      )");
  EXPECT_THAT(stats.Export(), EqualsProto(expected_2));

  FeatureStatistics stats2(&dataset.data_spec(), {0, 1, 2}, replacement_values);
  LOG(INFO) << "Report 3:\n" << stats.BuildReport();
  const proto::FeatureStatistics expected_3 = PARSE_TEST_PROTO(
      R"(
        features { numerical {} }
        features { numerical {} }
        features { categorical {} }
      )");
  EXPECT_THAT(stats2.Export(), EqualsProto(expected_3));

  CHECK_OK(stats2.ImportAndAggregate(stats.Export()));
  LOG(INFO) << "Report 4:\n" << stats.BuildReport();
  EXPECT_THAT(stats.Export(), EqualsProto(stats2.Export()));

  CHECK_OK(stats2.ImportAndAggregate(stats.Export()));
  LOG(INFO) << "Report 5:\n" << stats.BuildReport();
  const proto::FeatureStatistics expected_4 = PARSE_TEST_PROTO(
      R"(
        features {
          num_non_missing: 12
          numerical { sum: 24 sum_squared: 56 min: 1 max: 3 }
        }
        features {
          num_non_missing: 8
          numerical { sum: 20 sum_squared: 52 min: 2 max: 3 }
        }
        features {
          num_non_missing: 8
          categorical {
            count_per_value { key: 0 value: 4 }
            count_per_value { key: 1 value: 4 }
            count_per_value { key: 2 value: 4 }
          }
        }
        num_examples: 12
      )");
  EXPECT_THAT(stats2.Export(), EqualsProto(expected_4));

  EXPECT_EQ(stats.BuildReport(), R"(FeatureStatistics report
========================
Total number of features:3
Model input features:3
SERVING: Number of examples:6
MODEL: Number of examples:10
Features:
	"a" [NUMERICAL] SERVING: num-missing:0 (0.00%) MODEL: num-missing:0 (0.00%)
		SERVING: mean:2 min:1 max:3 sd:0.816497
		MODEL: mean:0 min:0 max:0
	"b" [NUMERICAL] SERVING: num-missing:2 (33.33%) MODEL: num-missing:0 (0.00%)
		SERVING: mean:2.5 min:2 max:3 sd:0.5
		MODEL: mean:0 min:0 max:0
	"c" [CATEGORICAL] SERVING: num-missing:2 (33.33%) MODEL: num-missing:0 (0.00%)
		"0" SERVING: count:2 (33.3333%) MODEL: count:0 (0%)
		"1" SERVING: count:2 (33.3333%) MODEL: count:0 (0%)
		"2" SERVING: count:2 (33.3333%) MODEL: count:0 (0%)
)");
}

TEST(FeatureStatistics, ToyExample) {
  feature_statistics_toy_example(ExampleFormat::FORMAT_EXAMPLE_MAJOR);
  feature_statistics_toy_example(ExampleFormat::FORMAT_FEATURE_MAJOR);
}

void feature_statistics_adult(const ExampleFormat format) {
  const auto model = LoadModel("adult_binary_class_gbdt");
  const auto dataset = LoadDataset(model->data_spec(), "adult_test.csv");

  auto* gbt_model =
      dynamic_cast<model::gradient_boosted_trees::GradientBoostedTreesModel*>(
          model.get());
  decision_forest::GradientBoostedTreesBinaryClassification specialized_model;
  CHECK_OK(GenericToSpecializedModel(*gbt_model, &specialized_model));

  FeatureStatistics stats(specialized_model);

  const int64_t batch_size = 10;
  const int64_t num_batches = (dataset.nrow() + batch_size - 1) / batch_size;
  std::vector<NumericalOrCategoricalValue> flat_examples;
  for (int64_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    const int64_t begin_example_idx = batch_idx * batch_size;
    const int64_t end_example_idx =
        std::min(begin_example_idx + batch_size, dataset.nrow());

    CHECK_OK(decision_forest::LoadFlatBatchFromDataset(
        dataset, begin_example_idx, end_example_idx,
        FeatureNames(specialized_model.features().fixed_length_features()),
        specialized_model.features().fixed_length_na_replacement_values(),
        &flat_examples, format));

    stats.Update(flat_examples, end_example_idx - begin_example_idx, format);
  }
  LOG(INFO) << "Report:";
  LOG(INFO) << stats.BuildReport();
}

TEST(FeatureStatistics, QKMS) {
  feature_statistics_adult(ExampleFormat::FORMAT_EXAMPLE_MAJOR);
  feature_statistics_adult(ExampleFormat::FORMAT_FEATURE_MAJOR);
}

}  // namespace
}  // namespace serving
}  // namespace yggdrasil_decision_forests
