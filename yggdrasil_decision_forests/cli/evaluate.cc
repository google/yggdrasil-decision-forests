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

// Evaluates a model.
//
// Usage example:
//   bazel run -c opt :evaluate -- \
//     --alsologtostderr \
//     --model=/path/to/my/model
//     --dataset=csv:/path/to/dataset.csv
//
#include <memory>

#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/dataset/weight.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/metric/report.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/protobuf.h"
#include "yggdrasil_decision_forests/utils/random.h"

ABSL_FLAG(std::string, model, "", "Model directory.");

ABSL_FLAG(std::string, dataset, "",
          "Typed path to dataset i.e. [type]:[path] format.");

ABSL_FLAG(std::string, options, "",
          "Path to optional evaluation configuration. proto::EvaluationOptions "
          "Text proto.");

constexpr char kUsageMessage[] = "Evaluates a model.";

namespace yggdrasil_decision_forests {
namespace cli {

void Evaluate() {
  // Check required flags.
  QCHECK(!absl::GetFlag(FLAGS_dataset).empty());
  QCHECK(!absl::GetFlag(FLAGS_model).empty());

  // Load the model
  std::unique_ptr<model::AbstractModel> model;
  QCHECK_OK(model::LoadModel(absl::GetFlag(FLAGS_model), &model));

  metric::proto::EvaluationOptions options =
      utils::ParseTextProto<metric::proto::EvaluationOptions>(
          absl::GetFlag(FLAGS_options))
          .value();
  utils::RandomEngine rnd;
  metric::proto::EvaluationResults evaluation;

  // Evaluation weighting.
  if (model->weights().has_value() && !options.has_weights()) {
    *options.mutable_weights() =
        dataset::GetUnlinkedWeightDefinition(model->weights().value(),
                                             model->data_spec())
            .value();
  }

  if (!options.has_task()) {
    options.set_task(model->task());
  }
  // evaluate model.
  evaluation = model->Evaluate(absl::GetFlag(FLAGS_dataset), options, &rnd);

  std::string text_report;
  metric::AppendTextReport(evaluation, &text_report);
  std::cout << "Evaluation:" << std::endl << text_report;
}

}  // namespace cli
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(kUsageMessage, &argc, &argv, true);
  yggdrasil_decision_forests::cli::Evaluate();
  return 0;
}
