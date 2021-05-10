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

#include "yggdrasil_decision_forests/dataset/synthetic_dataset.h"

#include "gtest/gtest.h"
#include "absl/strings/match.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/data_spec_inference.h"
#include "yggdrasil_decision_forests/dataset/synthetic_dataset.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace dataset {
namespace {

proto::DataSpecification GetDataSpec(const absl::string_view path,
                                     proto::DataSpecificationGuide guide = {}) {
  auto cat_int = guide.add_column_guides();
  cat_int->set_column_name_pattern("^cat_int_.*$");
  cat_int->set_type(dataset::proto::CATEGORICAL);

  auto cat_set_int = guide.add_column_guides();
  cat_set_int->set_column_name_pattern("^cat_set_int_.*$");
  cat_set_int->set_type(dataset::proto::CATEGORICAL_SET);

  proto::DataSpecification data_spec;
  CreateDataSpec(path, false, guide, &data_spec);
  return data_spec;
}

TEST(SyntheticDataset, BinaryClassification) {
  proto::SyntheticDatasetOptions options;
  const auto dst_path = absl::StrCat(
      "tfrecord+tfe:", file::JoinPath(test::TmpDirectory(), "dataset.tfr"));
  CHECK_OK(GenerateSyntheticDataset(options, dst_path));

  const auto data_spec = GetDataSpec(dst_path);
  std::string readable_representation = PrintHumanReadable(data_spec, true);
  LOG(INFO) << "SPEC:\n" << readable_representation;
}

TEST(SyntheticDataset, MultiClassClassification) {
  proto::SyntheticDatasetOptions options;
  options.mutable_classification()->set_num_classes(3);
  const auto dst_path = absl::StrCat(
      "tfrecord+tfe:", file::JoinPath(test::TmpDirectory(), "dataset.tfr"));
  CHECK_OK(GenerateSyntheticDataset(options, dst_path));

  const auto data_spec = GetDataSpec(dst_path);
  std::string readable_representation = PrintHumanReadable(data_spec, true);
  LOG(INFO) << "SPEC:\n" << readable_representation;
}

TEST(SyntheticDataset, Regression) {
  proto::SyntheticDatasetOptions options;
  options.mutable_regression();
  const auto dst_path = absl::StrCat(
      "tfrecord+tfe:", file::JoinPath(test::TmpDirectory(), "dataset.tfr"));
  CHECK_OK(GenerateSyntheticDataset(options, dst_path));

  const auto data_spec = GetDataSpec(dst_path);
  std::string readable_representation = PrintHumanReadable(data_spec, true);
  LOG(INFO) << "SPEC:\n" << readable_representation;
}

TEST(SyntheticDataset, MultidimensionalNumerical) {
  proto::SyntheticDatasetOptions options;
  options.set_num_numerical(0);
  options.set_num_categorical(0);
  options.set_num_categorical_set(0);
  options.set_num_boolean(0);
  options.set_num_multidimensional_numerical(2);
  options.set_num_accumulators(2);

  options.mutable_classification();
  const auto dst_path = absl::StrCat(
      "tfrecord+tfe:", file::JoinPath(test::TmpDirectory(), "dataset.tfr"));
  CHECK_OK(GenerateSyntheticDataset(options, dst_path));

  proto::DataSpecificationGuide guide;
  guide.set_unstack_numerical_set_as_numericals(true);
  const auto data_spec = GetDataSpec(dst_path, guide);
  std::string readable_representation = PrintHumanReadable(data_spec, true);
  LOG(INFO) << "SPEC:\n" << readable_representation;
}

TEST(SyntheticDataset, MultidimensionalNumericalInt) {
  proto::SyntheticDatasetOptions options;
  options.set_num_numerical(0);
  options.set_num_categorical(0);
  options.set_num_categorical_set(0);
  options.set_num_boolean(0);
  options.set_num_multidimensional_numerical(2);
  options.set_represent_numerical_as_integer(true);
  options.set_num_accumulators(2);

  options.mutable_classification();
  const auto dst_path = absl::StrCat(
      "tfrecord+tfe:", file::JoinPath(test::TmpDirectory(), "dataset.tfr"));
  CHECK_OK(GenerateSyntheticDataset(options, dst_path));

  proto::DataSpecificationGuide guide;
  guide.set_unstack_numerical_set_as_numericals(true);
  const auto data_spec = GetDataSpec(dst_path, guide);
  std::string readable_representation = PrintHumanReadable(data_spec, true);
  LOG(INFO) << "SPEC:\n" << readable_representation;
}

TEST(SyntheticDataset, MultidimensionalNumericalDiscretized) {
  proto::SyntheticDatasetOptions options;
  options.set_num_numerical(0);
  options.set_num_categorical(0);
  options.set_num_categorical_set(0);
  options.set_num_boolean(0);
  options.set_num_multidimensional_numerical(2);
  options.set_num_accumulators(2);

  options.mutable_classification();
  const auto dst_path = absl::StrCat(
      "tfrecord+tfe:", file::JoinPath(test::TmpDirectory(), "dataset.tfr"));
  CHECK_OK(GenerateSyntheticDataset(options, dst_path));

  proto::DataSpecificationGuide guide;
  guide.set_unstack_numerical_set_as_numericals(true);
  guide.set_detect_numerical_as_discretized_numerical(true);
  const auto data_spec = GetDataSpec(dst_path, guide);
  std::string readable_representation = PrintHumanReadable(data_spec, true);
  LOG(INFO) << "SPEC:\n" << readable_representation;
}

TEST(SyntheticDataset, WriteToCsv) {
  const auto dst_path =
      absl::StrCat("csv:", file::JoinPath(test::TmpDirectory(), "dataset.csv"));

  proto::SyntheticDatasetOptions options;
  EXPECT_THAT(GenerateSyntheticDataset(options, dst_path),
              test::StatusIs(absl::StatusCode::kInvalidArgument));

  options.set_num_categorical_set(0);
  options.set_num_multidimensional_numerical(0);
  CHECK_OK(GenerateSyntheticDataset(options, dst_path));

  const auto data_spec = GetDataSpec(dst_path);
  std::string readable_representation = PrintHumanReadable(data_spec, true);
  LOG(INFO) << "SPEC:\n" << readable_representation;
}

TEST(SyntheticDataset, Ranking) {
  proto::SyntheticDatasetOptions options;
  options.mutable_ranking();
  const auto dst_path = absl::StrCat(
      "tfrecord+tfe:", file::JoinPath(test::TmpDirectory(), "dataset.tfr"));
  CHECK_OK(GenerateSyntheticDataset(options, dst_path));

  const auto data_spec = GetDataSpec(dst_path);
  std::string readable_representation = PrintHumanReadable(data_spec, true);
  LOG(INFO) << "SPEC:\n" << readable_representation;
}

}  // namespace
}  // namespace dataset
}  // namespace yggdrasil_decision_forests
