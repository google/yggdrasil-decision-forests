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
#include "grpcpp/server_builder.h"
#include "grpcpp/support/channel_arguments.h"

namespace yggdrasil_decision_forests {
namespace distribute {

// Tests if a failing status is a transient error (i.e., the recipient is
// temporarily not available) or a definitive error.
bool IsTransientError(const grpc::Status& status);

// Index of a worker.
typedef int WorkerIdx;

// Creates a client context;
void ConfigureClientContext(grpc::ClientContext* context);

// Arguments of the channels to the workers: unlimited message sizes, a
// connection owned by the channel (a new channel opens a new connection
// instead of reusing the one of a previous channel to the same address), and,
// if "keepalive_time_ms > 0", keep-alive pings (see
// "GRPCImp.keepalive_time_ms"), also during long requests without data.
grpc::ChannelArguments WorkerChannelArguments(int keepalive_time_ms,
                                              int keepalive_timeout_ms);

// Lets the worker server accept the keep-alive pings of
// "WorkerChannelArguments" (by default, a server closes the connection of a
// client sending pings more often than every 5 minutes).
void AcceptKeepalivePings(grpc::ServerBuilder* builder);

}  // namespace distribute
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_DISTRIBUTE_IMPLEMENTATIONS_GRPC_GRPC_COMMON_H_
