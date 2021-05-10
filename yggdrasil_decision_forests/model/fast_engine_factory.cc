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

#include "yggdrasil_decision_forests/model/fast_engine_factory.h"

namespace yggdrasil_decision_forests {
namespace model {

std::vector<std::unique_ptr<FastEngineFactory>> ListAllFastEngines() {
  std::vector<std::unique_ptr<FastEngineFactory>> factories;
  for (const auto& engine_name : FastEngineFactoryRegisterer::GetNames()) {
    auto engine_factory = FastEngineFactoryRegisterer::Create(engine_name);
    if (!engine_factory.ok()) {
      LOG(WARNING) << "Error when creating fast engine:" << engine_name << " : "
                   << engine_factory.status();
    }
    factories.emplace_back(std::move(engine_factory).value());
  }
  return factories;
}

}  // namespace model
}  // namespace yggdrasil_decision_forests
