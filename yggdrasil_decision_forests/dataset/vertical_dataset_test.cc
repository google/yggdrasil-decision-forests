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

#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
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

TEST(VerticalDataset, ExtractAndAppend) {
  const auto dataset_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "toy.csv"));
  proto::DataSpecificationGuide guide;
  guide.mutable_default_column_guide()
      ->mutable_categorial()
      ->set_min_vocab_frequency(0);
  proto::DataSpecification data_spec;
  CreateDataSpec(dataset_path, false, guide, &data_spec);
  VerticalDataset dataset;
  EXPECT_OK(LoadVerticalDataset(dataset_path, data_spec, &dataset));
  std::vector<dataset::VerticalDataset::row_t> indices{1, 3};
  const auto extracted_dataset = dataset.Extract(indices).value();
  EXPECT_EQ(extracted_dataset.nrow(), 2);
  EXPECT_EQ(extracted_dataset.ncol(), 9);
  EXPECT_EQ("2", extracted_dataset.column(0)->ToString(
                     0, extracted_dataset.data_spec().columns(0)));
  EXPECT_EQ("4", extracted_dataset.column(0)->ToString(
                     1, extracted_dataset.data_spec().columns(0)));
  EXPECT_EQ("x", extracted_dataset.column(5)->ToString(
                     0, extracted_dataset.data_spec().columns(5)));
  EXPECT_EQ("x, y, z", extracted_dataset.column(5)->ToString(
                           1, extracted_dataset.data_spec().columns(5)));
}

TEST(VerticalDataset, ConvertToGivenDataspec) {
  const auto dataset_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "toy.csv"));
  proto::DataSpecificationGuide guide;
  guide.mutable_default_column_guide()
      ->mutable_categorial()
      ->set_min_vocab_frequency(0);
  proto::DataSpecification data_spec;
  CreateDataSpec(dataset_path, false, guide, &data_spec);
  VerticalDataset dataset;
  EXPECT_OK(LoadVerticalDataset(dataset_path, data_spec, &dataset));

  proto::DataSpecification new_data_spec;
  AddColumn("Num_1", proto::ColumnType::NUMERICAL, &new_data_spec);

  AddColumn("Num_2", proto::ColumnType::NUMERICAL, &new_data_spec);

  auto* cat_1_new_spec =
      AddColumn("Cat_1", proto::ColumnType::CATEGORICAL, &new_data_spec);
  cat_1_new_spec->mutable_categorical()->set_max_number_of_unique_values(3);
  (*cat_1_new_spec->mutable_categorical()->mutable_items())["OOV"].set_index(0);
  (*cat_1_new_spec->mutable_categorical()->mutable_items())["B"].set_index(1);
  (*cat_1_new_spec->mutable_categorical()->mutable_items())["A"].set_index(2);

  auto* cat_2_new_spec =
      AddColumn("Cat_2", proto::ColumnType::CATEGORICAL, &new_data_spec);
  cat_2_new_spec->mutable_categorical()->set_max_number_of_unique_values(2);
  (*cat_2_new_spec->mutable_categorical()->mutable_items())["OOV"].set_index(0);
  (*cat_2_new_spec->mutable_categorical()->mutable_items())["A"].set_index(1);

  auto* cat_set_1_new_spec = AddColumn(
      "Cat_set_1", proto::ColumnType::CATEGORICAL_SET, &new_data_spec);
  cat_set_1_new_spec->mutable_categorical()->set_max_number_of_unique_values(2);
  (*cat_set_1_new_spec->mutable_categorical()->mutable_items())["OOV"]
      .set_index(0);
  (*cat_set_1_new_spec->mutable_categorical()->mutable_items())["x"].set_index(
      1);

  auto* cat_set_2_new_spec = AddColumn(
      "Cat_set_2", proto::ColumnType::CATEGORICAL_SET, &new_data_spec);
  cat_set_2_new_spec->mutable_categorical()->set_max_number_of_unique_values(2);
  (*cat_set_2_new_spec->mutable_categorical()->mutable_items())["OOV"]
      .set_index(0);
  (*cat_set_2_new_spec->mutable_categorical()->mutable_items())["x"].set_index(
      1);

  const auto new_dataset =
      dataset.ConvertToGivenDataspec(new_data_spec, {}).value();

  EXPECT_EQ(new_dataset.nrow(), 4);
  EXPECT_EQ(new_dataset.ncol(), 6);

  EXPECT_EQ("1", new_dataset.column(0)->ToString(
                     0, new_dataset.data_spec().columns(0)));
  EXPECT_EQ("2", new_dataset.column(0)->ToString(
                     1, new_dataset.data_spec().columns(0)));

  EXPECT_EQ("nan", new_dataset.column(1)->ToString(
                       0, new_dataset.data_spec().columns(1)));
  EXPECT_EQ("2", new_dataset.column(1)->ToString(
                     1, new_dataset.data_spec().columns(1)));

  EXPECT_EQ("A", new_dataset.column(2)->ToString(
                     0, new_dataset.data_spec().columns(2)));
  EXPECT_EQ("B", new_dataset.column(2)->ToString(
                     1, new_dataset.data_spec().columns(2)));

  EXPECT_EQ("A", new_dataset.column(3)->ToString(
                     0, new_dataset.data_spec().columns(3)));
  EXPECT_EQ("NA", new_dataset.column(3)->ToString(
                      1, new_dataset.data_spec().columns(3)));
  // Note: "B" is not in the new dataspec dictionary.
  EXPECT_EQ("OOV", new_dataset.column(3)->ToString(
                       2, new_dataset.data_spec().columns(3)));

  EXPECT_EQ("x", new_dataset.column(4)->ToString(
                     0, new_dataset.data_spec().columns(4)));
  // Note: "y" is not in the new dataspec dictionary.
  EXPECT_EQ("x, OOV", new_dataset.column(4)->ToString(
                          1, new_dataset.data_spec().columns(4)));
}

TEST(VerticalDataset, ColumnWithCast) {
  const auto dataset_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "toy.csv"));
  proto::DataSpecificationGuide guide;
  guide.mutable_default_column_guide()
      ->mutable_categorial()
      ->set_min_vocab_frequency(0);
  proto::DataSpecification data_spec;
  CreateDataSpec(dataset_path, false, guide, &data_spec);
  VerticalDataset dataset;
  EXPECT_OK(LoadVerticalDataset(dataset_path, data_spec, &dataset));

  EXPECT_NE(
      dataset.MutableColumnWithCast<dataset::VerticalDataset::NumericalColumn>(
          0),
      nullptr);
  EXPECT_NE(
      dataset
          .MutableColumnWithCast<dataset::VerticalDataset::CategoricalColumn>(
              2),
      nullptr);
  EXPECT_NE(dataset.MutableColumnWithCast<
                dataset::VerticalDataset::CategoricalSetColumn>(4),
            nullptr);
  EXPECT_NE(
      dataset.MutableColumnWithCast<dataset::VerticalDataset::BooleanColumn>(6),
      nullptr);
}

TEST(VerticalDataset, MapExampleToProtoExample) {
  proto::DataSpecification data_spec;
  AddColumn("a", proto::ColumnType::NUMERICAL, &data_spec);
  AddColumn("b", proto::ColumnType::STRING, &data_spec);
  auto* c_col =
      AddColumn("c", proto::ColumnType::DISCRETIZED_NUMERICAL, &data_spec);
  c_col->mutable_discretized_numerical()->mutable_boundaries()->Add(0);
  c_col->mutable_discretized_numerical()->mutable_boundaries()->Add(1);
  c_col->mutable_discretized_numerical()->mutable_boundaries()->Add(2);
  AddColumn("d", proto::ColumnType::HASH, &data_spec);

  std::unordered_map<std::string, std::string> example_map{
      {"a", "0.5"}, {"b", "test"}, {"c", "1.5"}, {"d", "hello"}};
  proto::Example example;
  MapExampleToProtoExample(example_map, data_spec, &example);

  const proto::Example expected_example = PARSE_TEST_PROTO(
      R"(
        attributes { numerical: 0.5 }
        attributes { text: "test" }
        attributes { discretized_numerical: 2 }
        attributes { hash: 13009744463427800296 }
      )");
  EXPECT_THAT(example, EqualsProto(expected_example));
}

TEST(VerticalDataset, ProtoExampleToMapExample) {
  proto::DataSpecification data_spec;
  AddColumn("a", proto::ColumnType::NUMERICAL, &data_spec);
  AddColumn("b", proto::ColumnType::STRING, &data_spec);
  auto* c_col = AddColumn("c", proto::ColumnType::CATEGORICAL_SET, &data_spec);
  c_col->mutable_categorical()->set_is_already_integerized(true);
  c_col->mutable_categorical()->set_number_of_unique_values(5);
  auto* d_col =
      AddColumn("d", proto::ColumnType::DISCRETIZED_NUMERICAL, &data_spec);
  d_col->mutable_discretized_numerical()->mutable_boundaries()->Add(0);
  d_col->mutable_discretized_numerical()->mutable_boundaries()->Add(1);
  d_col->mutable_discretized_numerical()->mutable_boundaries()->Add(2);
  AddColumn("e", proto::ColumnType::HASH, &data_spec);

  const proto::Example example = PARSE_TEST_PROTO(
      R"(
        attributes { numerical: 0.5 }
        attributes { text: "test" }
        attributes { categorical_set { values: 1 values: 2 values: 4 } }
        attributes { discretized_numerical: 2 }
        attributes { hash: 1234 }
      )");
  const auto example_map = ProtoExampleToMapExample(example, data_spec).value();
  std::unordered_map<std::string, std::string> expected_example_map{
      {"a", "0.5"},
      {"b", "test"},
      {"c", "1, 2, 4"},
      {"d", "1.5"},
      {"e", "1234"}};
  EXPECT_EQ(example_map, expected_example_map);
}

TEST(VerticalDataset, AppendExample) {
  proto::DataSpecification data_spec;
  AddColumn("a", proto::ColumnType::NUMERICAL, &data_spec);
  AddColumn("b", proto::ColumnType::STRING, &data_spec);
  auto* c_col =
      AddColumn("c", proto::ColumnType::DISCRETIZED_NUMERICAL, &data_spec);
  c_col->mutable_discretized_numerical()->mutable_boundaries()->Add(0);
  c_col->mutable_discretized_numerical()->mutable_boundaries()->Add(1);
  c_col->mutable_discretized_numerical()->mutable_boundaries()->Add(2);
  AddColumn("d", proto::ColumnType::HASH, &data_spec);
  std::unordered_map<std::string, std::string> example_map{
      {"a", "0.5"}, {"b", "test"}, {"c", "1.1"}, {"d", "hello"}};
  VerticalDataset dataset;
  dataset.set_data_spec(data_spec);
  EXPECT_OK(dataset.CreateColumnsFromDataspec());
  dataset.AppendExample(example_map);
  dataset.AppendExample(example_map);

  EXPECT_EQ(dataset.nrow(), 2);
  EXPECT_EQ(dataset.ncol(), 4);
  EXPECT_EQ("0.5",
            dataset.column(0)->ToString(0, dataset.data_spec().columns(0)));
  EXPECT_EQ("test",
            dataset.column(1)->ToString(0, dataset.data_spec().columns(1)));
  EXPECT_EQ("1.5",
            dataset.column(2)->ToString(0, dataset.data_spec().columns(2)));
  EXPECT_EQ("13009744463427800296",
            dataset.column(3)->ToString(0, dataset.data_spec().columns(3)));
}

TEST(VerticalDataset, ExtractExample) {
  const proto::DataSpecification data_spec = PARSE_TEST_PROTO(
      R"(
        columns { type: NUMERICAL name: "a" }
        columns { type: NUMERICAL_SET name: "b" }
        columns { type: NUMERICAL_LIST name: "c" }
        columns {
          type: CATEGORICAL
          name: "d"
          categorical { is_already_integerized: true }
        }
        columns {
          type: CATEGORICAL_SET
          name: "e"
          categorical { is_already_integerized: true }
        }
        columns {
          type: CATEGORICAL_LIST
          name: "f"
          categorical { is_already_integerized: true }
        }
        columns { type: BOOLEAN name: "g" }
        columns { type: STRING name: "h" }
        columns {
          type: DISCRETIZED_NUMERICAL
          name: "i"
          discretized_numerical { boundaries: 0 boundaries: 1 boundaries: 2 }
        }
      )");

  VerticalDataset dataset;
  dataset.set_data_spec(data_spec);
  EXPECT_OK(dataset.CreateColumnsFromDataspec());

  const proto::Example example_1 = PARSE_TEST_PROTO(
      R"(
        attributes { numerical: 0.5 }
        attributes { numerical_set: { values: 0 values: 1 } }
        attributes { numerical_list: { values: 0 values: 1 } }
        attributes { categorical: 1 }
        attributes { categorical_set: { values: 0 values: 1 } }
        attributes { categorical_list: { values: 0 values: 1 } }
        attributes { boolean: 1 }
        attributes { text: "hello" }
        attributes { discretized_numerical: 2 }
      )");
  dataset.AppendExample(example_1);

  const proto::Example example_2 = PARSE_TEST_PROTO(
      R"(
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
        attributes {}
      )");
  dataset.AppendExample(example_2);

  proto::Example extracted_example_1;
  dataset.ExtractExample(0, &extracted_example_1);
  EXPECT_THAT(example_1, EqualsProto(extracted_example_1));

  proto::Example extracted_example_2;
  dataset.ExtractExample(1, &extracted_example_2);
  EXPECT_THAT(example_2, EqualsProto(extracted_example_2));
}

TEST(VerticalDataset, Append) {
  const std::string dataset_path =
      absl::StrCat("csv:", file::JoinPath(DatasetDir(), "toy.csv"));
  proto::DataSpecificationGuide guide;
  guide.mutable_default_column_guide()
      ->mutable_categorial()
      ->set_min_vocab_frequency(0);
  proto::DataSpecification data_spec;
  CreateDataSpec(dataset_path, false, guide, &data_spec);

  VerticalDataset dataset_1, dataset_2;
  EXPECT_OK(LoadVerticalDataset(dataset_path, data_spec, &dataset_1));
  EXPECT_OK(LoadVerticalDataset(dataset_path, data_spec, &dataset_2));

  EXPECT_OK(dataset_1.Append(dataset_2));

  EXPECT_EQ(dataset_1.nrow(), dataset_2.nrow() * 2);

  for (int row_idx = 0; row_idx < dataset_2.nrow(); row_idx++) {
    proto::Example example_1, example_2, example_3;
    dataset_1.ExtractExample(row_idx, &example_1);
    dataset_2.ExtractExample(row_idx, &example_2);
    dataset_1.ExtractExample(row_idx + dataset_2.nrow(), &example_3);
    EXPECT_THAT(example_1, EqualsProto(example_2));
    EXPECT_THAT(example_1, EqualsProto(example_3));
  }
}

TEST(VerticalDataset, PushBackNotOwnedColumn) {
  VerticalDataset dataset;
  EXPECT_EQ(dataset.ncol(), 0);
  VerticalDataset::NumericalColumn column;
  dataset.PushBackNotOwnedColumn(&column);
  EXPECT_EQ(dataset.column(0)->nrows(), 0);
  EXPECT_EQ(dataset.ncol(), 1);
  column.Add(5.f);
  EXPECT_EQ(dataset.column(0)->nrows(), 1);
}

TEST(VerticalDataset, PushBackOwnedColumn) {
  VerticalDataset dataset;
  EXPECT_EQ(dataset.ncol(), 0);
  dataset.PushBackOwnedColumn(
      absl::make_unique<VerticalDataset::NumericalColumn>());
  EXPECT_EQ(dataset.column(0)->nrows(), 0);
  EXPECT_EQ(dataset.ncol(), 1);
  dataset.MutableColumnWithCast<VerticalDataset::NumericalColumn>(0)->Add(5.f);
  EXPECT_EQ(dataset.column(0)->nrows(), 1);
}

TEST(VerticalDataset, ShallowNonOwningClone) {
  VerticalDataset original;
  AddColumn("a", proto::ColumnType::NUMERICAL, original.mutable_data_spec());
  AddColumn("b", proto::ColumnType::STRING, original.mutable_data_spec());
  EXPECT_OK(original.CreateColumnsFromDataspec());
  original.AppendExample({{"a", "0.1"}, {"b", "AAA"}});
  original.AppendExample({{"a", "0.2"}, {"b", "BBB"}});

  const auto clone_1 = original.ShallowNonOwningClone();
  const auto clone_2 = clone_1.ShallowNonOwningClone();

  EXPECT_EQ(original.nrow(), 2);
  EXPECT_EQ(clone_1.nrow(), 2);
  EXPECT_EQ(clone_2.nrow(), 2);

  EXPECT_EQ(original.column(0)->nrows(), 2);
  EXPECT_EQ(clone_1.column(0)->nrows(), 2);
  EXPECT_EQ(clone_2.column(0)->nrows(), 2);

  EXPECT_TRUE(original.OwnsColumn(0));
  EXPECT_FALSE(clone_1.OwnsColumn(0));
  EXPECT_FALSE(clone_2.OwnsColumn(0));
}

TEST(VerticalDataset, AddColumn) {
  VerticalDataset dataset;
  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"(
    columns { type: NUMERICAL name: "a" }
    columns { type: NUMERICAL name: "b" }
  )");
  EXPECT_OK(dataset.CreateColumnsFromDataspec());
  dataset.AppendExample({{"a", "0.1"}, {"b", "0.3"}});
  dataset.AppendExample({{"a", "0.2"}, {"b", "0.4"}});

  auto* col_c = dynamic_cast<dataset::VerticalDataset::NumericalColumn*>(
      dataset
          .AddColumn(PARSE_TEST_PROTO(R"(
            type: NUMERICAL name: "c"
          )"))
          .value());
  EXPECT_EQ(col_c->values().size(), 2);

  const proto::DataSpecification expected_dataspec = PARSE_TEST_PROTO(
      R"(
        columns { type: NUMERICAL name: "a" }
        columns { type: NUMERICAL name: "b" }
        columns { type: NUMERICAL name: "c" }
      )");
  EXPECT_THAT(dataset.data_spec(), EqualsProto(expected_dataspec));
}

TEST(VerticalDataset, ReplaceColumn) {
  VerticalDataset dataset;
  *dataset.mutable_data_spec() = PARSE_TEST_PROTO(R"(
    columns { type: NUMERICAL name: "a" }
    columns { type: NUMERICAL name: "b" }
  )");
  EXPECT_OK(dataset.CreateColumnsFromDataspec());
  dataset.AppendExample({{"a", "0.1"}, {"b", "0.3"}});
  dataset.AppendExample({{"a", "0.2"}, {"b", "0.4"}});

  auto* col_c = dynamic_cast<dataset::VerticalDataset::NumericalColumn*>(
      dataset
          .ReplaceColumn(0, PARSE_TEST_PROTO(R"(
                           type: NUMERICAL name: "c"
                         )"))
          .value());
  EXPECT_EQ(col_c->values().size(), 2);

  const proto::DataSpecification expected_dataspec = PARSE_TEST_PROTO(
      R"(
        columns { type: NUMERICAL name: "c" }
        columns { type: NUMERICAL name: "b" }
      )");
  EXPECT_THAT(dataset.data_spec(), EqualsProto(expected_dataspec));
}

TEST(VerticalDataset, Set) {
  VerticalDataset dataset;
  AddColumn("a", proto::ColumnType::NUMERICAL, dataset.mutable_data_spec());
  AddColumn("b", proto::ColumnType::STRING, dataset.mutable_data_spec());
  EXPECT_OK(dataset.CreateColumnsFromDataspec());
  dataset.AppendExample({{"a", "0.1"}, {"b", "AAA"}});
  dataset.AppendExample({{"a", "0.2"}, {"b", "BBB"}});

  dataset.Set(0, 0, PARSE_TEST_PROTO("numerical: 0.3"));
  dataset.Set(1, 0, PARSE_TEST_PROTO(""));  // Missing feature
  dataset.Set(0, 1, PARSE_TEST_PROTO("text: \"CCC\""));

  EXPECT_EQ(dataset.ValueToString(0, 0), "0.3");
  EXPECT_EQ(dataset.ValueToString(1, 0), "nan");
  EXPECT_EQ(dataset.ValueToString(0, 1), "CCC");
  EXPECT_EQ(dataset.ValueToString(1, 1), "BBB");
}

}  // namespace
}  // namespace dataset
}  // namespace yggdrasil_decision_forests
