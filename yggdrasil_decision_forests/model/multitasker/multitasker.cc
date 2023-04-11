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

#include "yggdrasil_decision_forests/model/multitasker/multitasker.h"

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/multitasker/multitasker.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace multitasker {

namespace {
// Filename containing the multitask header.
constexpr char kHeaderBaseFilename[] = "multitasker.pb";
}  // namespace

constexpr char MultitaskerModel::kRegisteredName[];

absl::Status MultitaskerModel::Save(absl::string_view directory,
                                    const ModelIOOptions& io_options) const {
  RETURN_IF_ERROR(ValidateModelIOOptions(io_options));
  proto::Header header;
  std::string header_filename =
      absl::StrCat(io_options.file_prefix.value(), kHeaderBaseFilename);
  header.set_num_models(models_.size());

  RETURN_IF_ERROR(file::RecursivelyCreateDir(directory, file::Defaults()));
  RETURN_IF_ERROR(file::SetBinaryProto(
      file::JoinPath(directory, header_filename), header, file::Defaults()));

  std::string effective_file_prefix;
  if (io_options.file_prefix.has_value()) {
    effective_file_prefix = io_options.file_prefix.value();
  }

  for (int model_idx = 0; model_idx < models_.size(); model_idx++) {
    auto sub_io_options = io_options;
    sub_io_options.file_prefix =
        absl::StrCat(effective_file_prefix, "_", model_idx);
    RETURN_IF_ERROR(
        SaveModel(directory, models_[model_idx].get(), sub_io_options));
  }

  return absl::OkStatus();
}

absl::Status MultitaskerModel::Load(absl::string_view directory,
                                    const ModelIOOptions& io_options) {
  RETURN_IF_ERROR(ValidateModelIOOptions(io_options));

  proto::Header header;
  std::string header_filename =
      absl::StrCat(io_options.file_prefix.value(), kHeaderBaseFilename);
  RETURN_IF_ERROR(file::GetBinaryProto(
      file::JoinPath(directory, header_filename), &header, file::Defaults()));
  models_.resize(header.num_models());

  std::string effective_file_prefix;
  if (io_options.file_prefix.has_value()) {
    effective_file_prefix = io_options.file_prefix.value();
  }

  for (int model_idx = 0; model_idx < header.num_models(); model_idx++) {
    auto sub_io_options = io_options;
    sub_io_options.file_prefix =
        absl::StrCat(effective_file_prefix, "_", model_idx);
    RETURN_IF_ERROR(LoadModel(directory, &models_[model_idx], sub_io_options));
  }
  return absl::OkStatus();
}

absl::Status MultitaskerModel::Validate() const {
  for (auto& model : models_) {
    RETURN_IF_ERROR(model->Validate());
  }
  return absl::OkStatus();
}

void MultitaskerModel::Predict(const dataset::VerticalDataset& dataset,
                               dataset::VerticalDataset::row_t row_idx,
                               model::proto::Prediction* prediction) const {
  models_.front()->Predict(dataset, row_idx, prediction);
}

void MultitaskerModel::Predict(const dataset::proto::Example& example,
                               model::proto::Prediction* prediction) const {
  models_.front()->Predict(example, prediction);
}

void MultitaskerModel::AppendDescriptionAndStatistics(
    bool full_definition, std::string* description) const {
  AbstractModel::AppendDescriptionAndStatistics(full_definition, description);
  for (int model_idx = 0; model_idx < models_.size(); model_idx++) {
    const auto& model = models_[model_idx];
    absl::SubstituteAndAppend(description, "model #$0\n========\n", model_idx);
    model->AppendDescriptionAndStatistics(full_definition, description);
    absl::StrAppend(description, "\n");
  }
}

}  // namespace multitasker
}  // namespace model
}  // namespace yggdrasil_decision_forests
