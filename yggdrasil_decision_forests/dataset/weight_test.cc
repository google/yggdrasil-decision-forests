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

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/weight.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

#include "yggdrasil_decision_forests/dataset/weight.h"

namespace yggdrasil_decision_forests {
namespace dataset {
namespace {

using test::EqualsProto;
using test::StatusIs;
using testing::ElementsAre;
using testing::SizeIs;

TEST(Weight, BadWeightLinking) {
  const proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"(
        columns {
          type: CATEGORICAL
          name: "Cat_1"
          categorical {
            number_of_unique_values: 3
            items {
              key: "a"
              value { index: 0 }
            }
            items {
              key: "b"
              value { index: 1 }
            }
            items {
              key: "c"
              value { index: 2 }
            }
          }
        }
      )");
  proto::LinkedWeightDefinition weight_link;

  const proto::WeightDefinition weight_def_1 = PARSE_TEST_PROTO(
      R"(
        attribute: "Cat_2"
        categorical {
          items { value: "a" weight: 1 }
          items { value: "b" weight: 2 }
          items { value: "c" weight: 3 }
        }
      )");
  EXPECT_THAT(GetLinkedWeightDefinition(weight_def_1, data_spec, &weight_link),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "does not match any column names"));

  const proto::WeightDefinition weight_def_2 = PARSE_TEST_PROTO(
      R"(
        attribute: "Cat_1"
        categorical {
          items { value: "a" weight: 2 }
          items { value: "c" weight: 3 }
        }
      )");
  EXPECT_THAT(GetLinkedWeightDefinition(weight_def_2, data_spec, &weight_link),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "\"b\" does not have any defined weight"));

  const proto::WeightDefinition weight_def_3 = PARSE_TEST_PROTO(
      R"(
        attribute: "Cat_1"
        categorical {
          items { value: "a" weight: 1 }
          items { value: "b" weight: 2 }
          items { value: "c" weight: 3 }
          items { value: "d" weight: 1 }
        }
      )");
  EXPECT_THAT(GetLinkedWeightDefinition(weight_def_3, data_spec, &weight_link),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "\"d\" is not defined in the column dataspec"));
}

TEST(Weight, LinkWeightDefinitionNumerical) {
  const proto::WeightDefinition weight_def = PARSE_TEST_PROTO(
      R"(
        attribute: "Num_1"
        numerical {}
      )");
  const proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"(
        columns { type: NUMERICAL name: "Num_1" }
      )");
  proto::LinkedWeightDefinition weight_link;
  EXPECT_OK(GetLinkedWeightDefinition(weight_def, data_spec, &weight_link));
  const proto::LinkedWeightDefinition expected = PARSE_TEST_PROTO(
      R"(
        attribute_idx: 0
        numerical {}
      )");
  EXPECT_THAT(weight_link, EqualsProto(expected));
}

TEST(Weight, LinkWeightDefinitionCategorical) {
  const proto::WeightDefinition weight_def = PARSE_TEST_PROTO(
      R"(
        attribute: "Cat_1"
        categorical {
          items { value: "b" weight: 2 }
          items { value: "c" weight: 3 }
        }
      )");
  const proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"(
        columns {
          type: CATEGORICAL
          name: "Cat_1"
          categorical {
            number_of_unique_values: 3
            items {
              key: "OOB"
              value { index: 0 }
            }
            items {
              key: "b"
              value { index: 1 }
            }
            items {
              key: "c"
              value { index: 2 }
            }
          }
        }
      )");
  proto::LinkedWeightDefinition weight_link;
  EXPECT_OK(GetLinkedWeightDefinition(weight_def, data_spec, &weight_link));
  const proto::LinkedWeightDefinition expected = PARSE_TEST_PROTO(
      R"(
        attribute_idx: 0
        categorical {
          categorical_value_idx_2_weight: 1
          categorical_value_idx_2_weight: 2
          categorical_value_idx_2_weight: 3
        }
      )");
  EXPECT_THAT(weight_link, EqualsProto(expected));
}

TEST(Weight, GetWeightNumerical) {
  const proto::WeightDefinition weight_def = PARSE_TEST_PROTO(
      R"(
        attribute: "Num_1"
        numerical {}
      )");
  const proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"(
        columns { type: NUMERICAL name: "Num_1" is_manual_type: true }
      )");
  proto::LinkedWeightDefinition weight_link;
  CHECK_OK(GetLinkedWeightDefinition(weight_def, data_spec, &weight_link));
  VerticalDataset dataset;
  dataset.set_data_spec(data_spec);
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  dataset.AppendExample({{"Num_1", "0"}});
  dataset.AppendExample({{"Num_1", "1"}});
  dataset.AppendExample({{"Num_1", "2"}});
  dataset.AppendExample({{"Num_1", "3"}});
  EXPECT_NEAR(GetWeight(dataset, 0, weight_link), 0.f, 0.001f);
  EXPECT_NEAR(GetWeight(dataset, 1, weight_link), 1.f, 0.001f);
  EXPECT_NEAR(GetWeight(dataset, 2, weight_link), 2.f, 0.001f);

  for (int example_idx = 0; example_idx < dataset.nrow(); example_idx++) {
    proto::Example example;
    dataset.ExtractExample(example_idx, &example);
    EXPECT_NEAR(GetWeight(dataset, 0, weight_link),
                GetWeight(dataset, 0, weight_link), 0.001f);
  }

  std::vector<float> weights;
  EXPECT_OK(GetWeights(dataset, weight_link, &weights));
  EXPECT_THAT(weights, ElementsAre(0.f, 1.f, 2.f, 3.f));

  dataset.AppendExample({{"Num_1", "NA"}});
  EXPECT_THAT(GetWeights(dataset, weight_link, &weights),
              StatusIs(absl::StatusCode::kInvalidArgument, "Found NA value"));
}

TEST(Weight, GetWeightCategorical) {
  const proto::WeightDefinition weight_def = PARSE_TEST_PROTO(
      R"(
        attribute: "Cat_1"
        categorical {
          items { value: "a" weight: 1 }
          items { value: "b" weight: 2 }
          items { value: "c" weight: 3 }
        }
      )");
  const proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"(
        columns {
          type: CATEGORICAL
          name: "Cat_1"
          categorical {
            number_of_unique_values: 3
            items {
              key: "a"
              value { index: 0 }
            }
            items {
              key: "b"
              value { index: 1 }
            }
            items {
              key: "c"
              value { index: 2 }
            }
          }
        }
      )");
  ASSERT_FALSE(HasFailure()) << "error during proto parsing";
  proto::LinkedWeightDefinition weight_link;
  CHECK_OK(GetLinkedWeightDefinition(weight_def, data_spec, &weight_link));
  VerticalDataset dataset;
  dataset.set_data_spec(data_spec);
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  dataset.AppendExample({{"Cat_1", "a"}});
  dataset.AppendExample({{"Cat_1", "b"}});
  dataset.AppendExample({{"Cat_1", "c"}});
  dataset.AppendExample({{"Cat_1", "a"}});
  EXPECT_NEAR(GetWeight(dataset, 0, weight_link), 1.f, 0.001f);
  EXPECT_NEAR(GetWeight(dataset, 1, weight_link), 2.f, 0.001f);
  EXPECT_NEAR(GetWeight(dataset, 2, weight_link), 3.f, 0.001f);

  for (int example_idx = 0; example_idx < dataset.nrow(); example_idx++) {
    proto::Example example;
    dataset.ExtractExample(example_idx, &example);
    EXPECT_NEAR(GetWeight(dataset, 0, weight_link),
                GetWeight(dataset, 0, weight_link), 0.001f);
  }

  std::vector<float> weights;
  CHECK_OK(GetWeights(dataset, weight_link, &weights));
  EXPECT_THAT(weights, ElementsAre(1.f, 2.f, 3.f, 1.f));

  dataset.AppendExample({{"Cat_1", "NA"}});
  EXPECT_THAT(GetWeights(dataset, weight_link, &weights),
              StatusIs(absl::StatusCode::kInvalidArgument, "Found NA value"));
}

TEST(Weight, OptimizedUnspecifiedWeightsAreEmpty) {
  const proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"pb(
        columns { type: NUMERICAL name: "Num_1" is_manual_type: true }
      )pb");
  VerticalDataset dataset;
  dataset.set_data_spec(data_spec);
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  dataset.AppendExample({{"Num_1", "0"}});
  dataset.AppendExample({{"Num_1", "1"}});
  dataset.AppendExample({{"Num_1", "2"}});

  model::proto::TrainingConfigLinking config_link;

  std::vector<float> optimized_weights;
  EXPECT_OK(GetWeights(dataset, config_link, &optimized_weights,
                       /*use_optimized_unit_weights=*/true));
  EXPECT_THAT(optimized_weights, SizeIs(0));

  std::vector<float> non_optimized_weights;
  EXPECT_OK(GetWeights(dataset, config_link, &non_optimized_weights,
                       /*use_optimized_unit_weights=*/false));
  EXPECT_THAT(non_optimized_weights, SizeIs(3));
}

TEST(Weight, OptimizedUnitWeightsAreEmpty) {
  const proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"pb(
        columns { type: NUMERICAL name: "Num_1" is_manual_type: true }
      )pb");
  VerticalDataset dataset;
  dataset.set_data_spec(data_spec);
  CHECK_OK(dataset.CreateColumnsFromDataspec());
  dataset.AppendExample({{"Num_1", "1"}});
  dataset.AppendExample({{"Num_1", "1"}});
  dataset.AppendExample({{"Num_1", "1"}});

  model::proto::TrainingConfigLinking config_link;
  config_link.mutable_weight_definition()->set_attribute_idx(0);
  config_link.mutable_weight_definition()->mutable_numerical();

  std::vector<float> optimized_weights;
  EXPECT_OK(GetWeights(dataset, config_link, &optimized_weights,
                       /*use_optimized_unit_weights=*/true));
  EXPECT_THAT(optimized_weights, SizeIs(0));

  std::vector<float> non_optimized_weights;
  EXPECT_OK(GetWeights(dataset, config_link, &non_optimized_weights,
                       /*use_optimized_unit_weights=*/false));
  EXPECT_THAT(non_optimized_weights, SizeIs(3));
}

TEST(Weight, OptimizedNonunitWeightsAreUnchanged) {
  const proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"pb(
        columns { type: NUMERICAL name: "Num_1" is_manual_type: true }
      )pb");
  VerticalDataset dataset_nonequal_weights;
  dataset_nonequal_weights.set_data_spec(data_spec);
  CHECK_OK(dataset_nonequal_weights.CreateColumnsFromDataspec());
  dataset_nonequal_weights.AppendExample({{"Num_1", "1"}});
  dataset_nonequal_weights.AppendExample({{"Num_1", "2"}});
  dataset_nonequal_weights.AppendExample({{"Num_1", "3"}});

  model::proto::TrainingConfigLinking config_link;
  config_link.mutable_weight_definition()->set_attribute_idx(0);
  config_link.mutable_weight_definition()->mutable_numerical();

  std::vector<float> optimized_nonequal_weights;
  EXPECT_OK(GetWeights(dataset_nonequal_weights, config_link,
                       &optimized_nonequal_weights,
                       /*use_optimized_unit_weights=*/true));
  EXPECT_THAT(optimized_nonequal_weights, ElementsAre(1, 2, 3));

  VerticalDataset dataset_nonunit_weights;
  dataset_nonunit_weights.set_data_spec(data_spec);
  CHECK_OK(dataset_nonunit_weights.CreateColumnsFromDataspec());
  dataset_nonunit_weights.AppendExample({{"Num_1", "2"}});
  dataset_nonunit_weights.AppendExample({{"Num_1", "2"}});
  dataset_nonunit_weights.AppendExample({{"Num_1", "2"}});

  std::vector<float> optimized_nonunit_weights;
  EXPECT_OK(GetWeights(dataset_nonunit_weights, config_link,
                       &optimized_nonunit_weights,
                       /*use_optimized_unit_weights=*/true));
  EXPECT_THAT(optimized_nonunit_weights, ElementsAre(2, 2, 2));
}

}  // namespace
}  // namespace dataset
}  // namespace yggdrasil_decision_forests
