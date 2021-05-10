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

#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace dataset {
namespace {

using test::EqualsProto;

std::string DatasetDir() {
  return file::JoinPath(test::DataRootDirectory(),
                        "yggdrasil_decision_forests/"
                        "test_data/dataset");
}

TEST(VerticalDatasetIOTest, Load) {
  for (const auto& dataset_path :
       {absl::StrCat("csv:", file::JoinPath(DatasetDir(), "toy.csv"))}) {
    LOG(INFO) << "Create dataspec for " << dataset_path;
    proto::DataSpecificationGuide guide;
    proto::DataSpecification data_spec;
    CreateDataSpec(dataset_path, false, guide, &data_spec);
    LOG(INFO) << "Load " << dataset_path;
    VerticalDataset ds;
    EXPECT_OK(LoadVerticalDataset(dataset_path, data_spec, &ds));
    EXPECT_EQ(ds.nrow(), 4);
    EXPECT_EQ(ds.ncol(), 9);

    for (int col_idx = 0; col_idx < ds.ncol(); col_idx++) {
      EXPECT_EQ(ds.column(col_idx)->name(),
                ds.data_spec().columns(col_idx).name());
    }
  }
}

TEST(VerticalDatasetIOTest, LoadSaveLoad) {
  const std::string dataset_path = file::JoinPath(DatasetDir(), "toy.csv");
  const std::string format = "csv";
  // Paths
  const std::string typed_dataset_path =
      absl::StrCat(format, ":", dataset_path);
  const std::string dataset_saved_path =
      file::JoinPath(test::TmpDirectory(), "dataset_copy");
  const std::string typed_dataset_saved_path =
      absl::StrCat(format, ":", dataset_saved_path);

  // Infer dataset columns.
  proto::DataSpecificationGuide guide;
  guide.mutable_default_column_guide()
      ->mutable_categorial()
      ->set_min_vocab_frequency(1);
  guide.mutable_default_column_guide()
      ->mutable_tokenizer()
      ->mutable_tokenizer()
      ->set_to_lower_case(false);
  proto::DataSpecification data_spec;
  CreateDataSpec(typed_dataset_path, false, guide, &data_spec);
  // Load dataset.
  VerticalDataset ds;
  EXPECT_OK(LoadVerticalDataset(typed_dataset_path, data_spec, &ds));
  // Save dataset.
  EXPECT_OK(SaveVerticalDataset(ds, typed_dataset_saved_path));
  // Re-load dataset.
  VerticalDataset ds2;
  EXPECT_OK(LoadVerticalDataset(typed_dataset_saved_path, data_spec, &ds2));
  // Check equality between the datasets.

  for (int example_idx = 0; example_idx < ds.nrow(); example_idx++) {
    int column_idx = 0;
    EXPECT_EQ(
        ds.MutableColumnWithCast<VerticalDataset::NumericalColumn>(column_idx)
            ->ToString(example_idx, ds.data_spec().columns(column_idx)),
        ds2.MutableColumnWithCast<VerticalDataset::NumericalColumn>(column_idx)
            ->ToString(example_idx, ds.data_spec().columns(column_idx)));

    column_idx = 1;
    EXPECT_EQ(
        ds.MutableColumnWithCast<VerticalDataset::NumericalColumn>(column_idx)
            ->ToString(example_idx, ds.data_spec().columns(column_idx)),
        ds2.MutableColumnWithCast<VerticalDataset::NumericalColumn>(column_idx)
            ->ToString(example_idx, ds.data_spec().columns(column_idx)));

    column_idx = 2;
    EXPECT_EQ(
        ds.MutableColumnWithCast<VerticalDataset::CategoricalColumn>(column_idx)
            ->ToString(example_idx, ds.data_spec().columns(column_idx)),
        ds2.MutableColumnWithCast<VerticalDataset::CategoricalColumn>(
               column_idx)
            ->ToString(example_idx, ds.data_spec().columns(column_idx)));

    column_idx = 3;
    EXPECT_EQ(
        ds.MutableColumnWithCast<VerticalDataset::CategoricalColumn>(column_idx)
            ->ToString(example_idx, ds.data_spec().columns(column_idx)),
        ds2.MutableColumnWithCast<VerticalDataset::CategoricalColumn>(
               column_idx)
            ->ToString(example_idx, ds.data_spec().columns(column_idx)));

    column_idx = 4;
    EXPECT_EQ(ds.MutableColumnWithCast<VerticalDataset::CategoricalSetColumn>(
                    column_idx)
                  ->ToString(example_idx, ds.data_spec().columns(column_idx)),
              ds2.MutableColumnWithCast<VerticalDataset::CategoricalSetColumn>(
                     column_idx)
                  ->ToString(example_idx, ds.data_spec().columns(column_idx)));

    column_idx = 5;
    EXPECT_EQ(ds.MutableColumnWithCast<VerticalDataset::CategoricalSetColumn>(
                    column_idx)
                  ->ToString(example_idx, ds.data_spec().columns(column_idx)),
              ds2.MutableColumnWithCast<VerticalDataset::CategoricalSetColumn>(
                     column_idx)
                  ->ToString(example_idx, ds.data_spec().columns(column_idx)));

    column_idx = 6;
    EXPECT_EQ(
        ds.MutableColumnWithCast<VerticalDataset::BooleanColumn>(column_idx)
            ->ToString(example_idx, ds.data_spec().columns(column_idx)),
        ds2.MutableColumnWithCast<VerticalDataset::BooleanColumn>(column_idx)
            ->ToString(example_idx, ds.data_spec().columns(column_idx)));
  }
}

TEST(Dataset, TokenizeTfExample) {
  // Contains the two following examples:
  // {f1: ["Hello the world"]
  // {f1: ["Bonjour le monde"]}
  const auto path = absl::StrCat(
      "tfrecord+tfe:", file::JoinPath(DatasetDir(), "sentences.tfe-tfrecord"));

  const proto::DataSpecificationGuide guide = PARSE_TEST_PROTO(
      R"(
        column_guides {
          column_name_pattern: ".*"
          type: CATEGORICAL_LIST
          tokenizer {}
          categorial { min_vocab_frequency: 1 }
        }
      )");
  proto::DataSpecification data_spec;
  CreateDataSpec(path, false, guide, &data_spec);

  VerticalDataset dataset;
  EXPECT_OK(LoadVerticalDataset(path, data_spec, &dataset));

  EXPECT_EQ(dataset.nrow(), 2);

  EXPECT_EQ(dataset.ValueToString(0, 0), "hello, world");
  EXPECT_EQ(dataset.ValueToString(1, 0), "bonjour, le, monde");

  proto::Example example;
  dataset.ExtractExample(0, &example);
  const proto::Example expected_example_1 = PARSE_TEST_PROTO(R"(
    attributes { categorical_list { values: 4 values: 1 } }
  )");
  EXPECT_THAT(example, EqualsProto(expected_example_1));
  dataset.ExtractExample(1, &example);
  const proto::Example expected_example_2 = PARSE_TEST_PROTO(R"(
    attributes { categorical_list { values: 5 values: 3 values: 2 } }
  )");
  EXPECT_THAT(example, EqualsProto(expected_example_2));
}

}  // namespace
}  // namespace dataset
}  // namespace yggdrasil_decision_forests
