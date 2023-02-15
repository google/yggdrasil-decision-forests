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

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_GRPC_GRPC_COMMON_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_GRPC_GRPC_COMMON_H_

#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"

namespace yggdrasil_decision_forests {
namespace distribute {

// Tests if a failing status is a transiant error (i.e., the recipient is
// temporarily not available) or a definitive error.
bool IsTransientError(const grpc::Status& status);

// Index of a worker.
typedef int WorkerIdx;

// Creates a client context;
void ConfigureClientContext(grpc::ClientContext* context);

}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_GRPC_GRPC_COMMON_H_
