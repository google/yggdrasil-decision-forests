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

// Display the statistics and structure of a model.

#include "absl/flags/flag.h"
#include "yggdrasil_decision_forests/model/model_library.h"
#include "yggdrasil_decision_forests/utils/logging.h"

ABSL_FLAG(std::string, model, "", "Model directory.");

ABSL_FLAG(bool, full_definition, false,
          "Show the full details of the model. For decision forest models, "
          "show the tree structure.");

ABSL_FLAG(
    bool, engines, false,
    "List and test the fast engines compatible with the model. Note: "
    "Engines needs to be linked to the binary. Some engines depend on "
    "the platform e.g. if you don't have AVX2, AVX2 engines won't be listed.");

ABSL_FLAG(bool, dataspec, false,
          "Show the dataspec contained in the model. This is similar as "
          "running :show_dataspec on the "
          "data_spec.pb file in the model directory.");

constexpr char kUsageMessage[] =
    "Display the statistics and structure of a model.";

namespace yggdrasil_decision_forests {
namespace cli {

void ListEngines(const model::AbstractModel* model) {
  std::cout << "Fast serving engines:" << std::endl;

  for (auto& factory : model->ListCompatibleFastEngines()) {
    // Information about the engine.
    std::cout << "  " << factory->name() << std::endl;
    std::cout << "    is better than: {"
              << absl::StrJoin(factory->IsBetterThan(), ",") << "}"
              << std::endl;

    // Create the engine.
    auto engine = factory->CreateEngine(model);
    if (engine.ok()) {
      std::cout << "    The engine was created successfully." << std::endl;
    } else {
      std::cout << "    The engine could not be created: " << engine.status()
                << std::endl;
    }
  }

  // Create the best engine.
  auto best_engine = model->BuildFastEngine();
  if (best_engine.ok()) {
    std::cout << "  The best engine was created successfully." << std::endl;
  } else {
    std::cout << "  The best engine could not be created: "
              << best_engine.status() << std::endl;
  }
}

void ShowModel() {
  // Check required flags.
  QCHECK(!absl::GetFlag(FLAGS_model).empty());

  // Loads the model.
  std::unique_ptr<model::AbstractModel> model;
  CHECK_OK(model::LoadModel(absl::GetFlag(FLAGS_model), &model));

  // Description of the model.
  std::string description;
  model->AppendDescriptionAndStatistics(absl::GetFlag(FLAGS_full_definition),
                                        &description);
  std::cout << description;

  // Show dataspec.
  if (absl::GetFlag(FLAGS_dataspec)) {
    std::cout << std::endl << "Dataspec" << std::endl;
    std::cout << "========" << std::endl << std::endl;
    std::cout << dataset::PrintHumanReadable(model->data_spec(), false);
    std::cout << std::endl;
  }

  // List the engines supported by the model.
  if (absl::GetFlag(FLAGS_engines)) {
    std::cout << std::endl << "Engines" << std::endl;
    std::cout << "========" << std::endl << std::endl;
    ListEngines(model.get());
  }
}

}  // namespace cli
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(kUsageMessage, &argc, &argv, true);
  yggdrasil_decision_forests::cli::ShowModel();
  return 0;
}
