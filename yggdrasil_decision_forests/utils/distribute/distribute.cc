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

#include "yggdrasil_decision_forests/utils/distribute/distribute.h"

#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace distribute {

utils::StatusOr<std::unique_ptr<AbstractManager>> CreateManager(
    const proto::Config& config, const absl::string_view worker_name,
    Blob welcome_blob, int parallel_execution_per_worker) {
  ASSIGN_OR_RETURN(auto manager, AbstractManagerRegisterer::Create(
                                     config.implementation_key()));
  RETURN_IF_ERROR(manager->Initialize(config, worker_name, welcome_blob,
                                      parallel_execution_per_worker));
  return manager;
}

utils::StatusOr<int> NumWorkers(const proto::Config& config) {
  ASSIGN_OR_RETURN(auto manager, AbstractManagerRegisterer::Create(
                                     config.implementation_key()));
  const auto num_workers = manager->NumWorkersInConfiguration(config);
  RETURN_IF_ERROR(manager->Done());
  // Returning num_workers directly fails in old versions of absl currently used
  // by tensorflow.
  if (num_workers.ok()) {
    return num_workers.value();
  }
  return num_workers.status();
}

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
