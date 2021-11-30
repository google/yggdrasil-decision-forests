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

// Train a ML model and export it to disk.

#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/learner/learner_library.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(std::string, output, "", "Output model directory.");

ABSL_FLAG(std::string, dataset, "",
          "Typed path to training dataset i.e. [type]:[path] format. Support "
          "glob, shard and comma. Example: csv:/my/dataset.csv");

ABSL_FLAG(
    std::string, dataspec, "",
    "Path to the dataset specification (dataspec). Note: The dataspec is often "
    "created with :infer_dataspec and inspected with :show_dataspec.");

ABSL_FLAG(std::string, config, "",
          "Path to the training configuration i.e. a "
          "model::proto::TrainingConfig text proto.");

ABSL_FLAG(
    std::string, deployment, "",
    "Path to the deployment configuration for the training i.e. what computing "
    "resources to use to train the model. Text proto buffer of type "
    "model::proto::DeploymentConfig. If not specified, the training is done "
    "locally with a number of threads chosen by the training algorithm.");

ABSL_FLAG(
    std::string, valid_dataset, "",
    "Optional validation dataset specified with [type]:[path] format. If "
    "not specified and if the learning algorithm uses a validation dataset, "
    "the effective validation dataset is extracted from the training dataset.");

constexpr char kUsageMessage[] = "Train a ML model and export it to disk.";

namespace yggdrasil_decision_forests {
namespace cli {

void Train() {
  // Check required flags.
  QCHECK(!absl::GetFlag(FLAGS_output).empty());
  QCHECK(!absl::GetFlag(FLAGS_dataset).empty());
  QCHECK(!absl::GetFlag(FLAGS_dataspec).empty());
  QCHECK(!absl::GetFlag(FLAGS_config).empty());

  // Load configuration protos and the dataspec.
  dataset::proto::DataSpecification data_spec;
  model::proto::DeploymentConfig deployment;
  model::proto::TrainingConfig config;
  QCHECK_OK(file::GetTextProto(absl::GetFlag(FLAGS_dataspec), &data_spec,
                               file::Defaults()));
  QCHECK_OK(file::GetTextProto(absl::GetFlag(FLAGS_config), &config,
                               file::Defaults()));
  if (!absl::GetFlag(FLAGS_deployment).empty()) {
    QCHECK_OK(file::GetTextProto(absl::GetFlag(FLAGS_deployment), &deployment,
                                 file::Defaults()));
  }

  if (!config.metadata().has_framework()) {
    config.mutable_metadata()->set_framework("Yggdrasil cli");
  }

  LOG(INFO) << "Configuration:\n" << config.DebugString();
  LOG(INFO) << "Deployment:\n" << deployment.DebugString();

  std::unique_ptr<model::AbstractLearner> learner;
  QCHECK_OK(model::GetLearner(config, &learner));
  *learner->mutable_deployment() = deployment;
  learner->set_log_directory(
      file::JoinPath(absl::GetFlag(FLAGS_output), "train_logs"));

  absl::optional<std::string> valid_dataset;
  if (!absl::GetFlag(FLAGS_valid_dataset).empty()) {
    valid_dataset = absl::GetFlag(FLAGS_valid_dataset);
  }

  LOG(INFO) << "Start training model.";
  auto model = learner
                   ->TrainWithStatus(absl::GetFlag(FLAGS_dataset), data_spec,
                                     valid_dataset)
                   .value();

  LOG(INFO) << "Save model.";
  QCHECK_OK(model::SaveModel(absl::GetFlag(FLAGS_output), model.get()));
}

}  // namespace cli
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(kUsageMessage, &argc, &argv, true);
  yggdrasil_decision_forests::cli::Train();
  return 0;
}
