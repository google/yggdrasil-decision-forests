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

// FastEngineFactory can create FastEngine.
#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_FAST_ENGINE_FACTORY_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_FAST_ENGINE_FACTORY_H_

#include <memory>
#include <string>
#include <vector>

#include "yggdrasil_decision_forests/serving/fast_engine.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/registration.h"

namespace yggdrasil_decision_forests {
namespace model {

class AbstractModel;

class FastEngineFactory {
 public:
  virtual ~FastEngineFactory() = default;

  // Unique name of the engine.
  virtual std::string name() const = 0;

  // Creates an engine. The model can be discarded after the call.
  virtual utils::StatusOr<std::unique_ptr<serving::FastEngine>> CreateEngine(
      const AbstractModel* const model) const = 0;

  // Checks if an engine is compatible with a model.
  virtual bool IsCompatible(const AbstractModel* const model) const = 0;

  // Returns the list of engine names that are know to be worst/slower than the
  // current engine.
  virtual std::vector<std::string> IsBetterThan() const = 0;
};

REGISTRATION_CREATE_POOL(FastEngineFactory);

#define REGISTER_FastEngineFactory(name, key) \
  REGISTRATION_REGISTER_CLASS(name, key, FastEngineFactory);

// Lists all the engines linked in the binary.
std::vector<std::unique_ptr<FastEngineFactory>> ListAllFastEngines();

}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_FAST_ENGINE_FACTORY_H_
