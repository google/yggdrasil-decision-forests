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
#include "yggdrasil_decision_forests/model/fast_engine_factory.h"
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

ABSL_FLAG(bool, explain_engine_incompatibility, false,
          "If true, and if --engines=true, print an explanation of why each of "
          "the available serving engine is not compatible with the model.");

constexpr char kUsageMessage[] =
    "Display the statistics and structure of a model.";

namespace yggdrasil_decision_forests {
namespace cli {

void ListEngines(const bool explain_engine_incompatibility,
                 const model::AbstractModel* model) {
  const auto compatible_engines = model->ListCompatibleFastEngines();
  std::cout << "  There are " << compatible_engines.size()
            << " compatible fast serving engine(s):" << std::endl;

  for (auto& factory : compatible_engines) {
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
  std::cout << std::endl
            << "  Create the best engine with \"BuildFastEngine()\":"
            << std::endl;
  auto best_engine = model->BuildFastEngine();
  if (best_engine.ok()) {
    std::cout << "  The best engine was created successfully." << std::endl;
  } else {
    std::cout << "  The best engine could not be created: "
              << best_engine.status() << std::endl;
  }

  // Explain why each engine is compatible or not.
  if (explain_engine_incompatibility) {
    const auto all_engine_factors = model::ListAllFastEngines();
    std::cout << "  There are " << all_engine_factors.size()
              << " registered fast serving engine(s):" << std::endl;
    for (auto& factory : all_engine_factors) {
      if (factory->IsCompatible(model)) {
        std::cout << "  engine " << factory->name() << " is compatible"
                  << std::endl;
      } else {
        const auto engine_or = factory->CreateEngine(model);
        std::cout << "  engine " << factory->name()
                  << " is not compatible:" << std::endl
                  << "    ";
        if (engine_or.ok()) {
          std::cout << "Unknown reason. Please report this error to us.";
        } else {
          std::cout << engine_or.status().message();
        }
        std::cout << std::endl;
      }
    }
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
    ListEngines(absl::GetFlag(FLAGS_explain_engine_incompatibility),
                model.get());
  }
}

}  // namespace cli
}  // namespace yggdrasil_decision_forests

int main(int argc, char** argv) {
  InitLogging(kUsageMessage, &argc, &argv, true);
  yggdrasil_decision_forests::cli::ShowModel();
  return 0;
}
