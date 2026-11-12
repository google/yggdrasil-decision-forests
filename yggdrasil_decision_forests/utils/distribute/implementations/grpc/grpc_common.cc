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

#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_common.h"

#include <limits>
#include <string>

#include "grpcpp/server_builder.h"
#include "grpcpp/support/channel_arguments.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "third_party/grpc/include/grpc/grpc.h"
#include "third_party/grpc/include/grpcpp/client_context.h"
#include "third_party/grpc/include/grpcpp/server_builder.h"
#include "third_party/grpc/include/grpcpp/support/channel_arguments.h"
#include "third_party/grpc/include/grpcpp/support/status.h"

namespace yggdrasil_decision_forests {
namespace distribute {

bool IsTransientError(const grpc::Status& status) {
  const std::string& error_message = status.error_message();
  static constexpr absl::string_view kTransientErrors[] = {
      "Socket closed", "Stream removed", "Transport closed",
      "Connection reset by peer", "recvmsg:Connection reset by peer",
      "Broken pipe", "keepalive watchdog timeout",
      // Connection closed by the keep-alive pings (see
      // "WorkerChannelArguments").
      "keepalive timeout", "ping timeout", "failed to connect to all addresses",
      "empty address list",
      // The worker is being rescheduled.
      "Connection timed out", "Connection refused", "DNS resolution failed",
      "name resolution"};

  for (const auto& error : kTransientErrors) {
    if (absl::StrContainsIgnoreCase(error_message, error)) {
      return true;
    }
  }
  return false;
}

void ConfigureClientContext(grpc::ClientContext* context) {
  // Use default context creation.
  //
  // Do not set wait_for_ready=true or long deadline as fast connection failure
  // is necessary for on-the-fly change of worker address.
}

grpc::ChannelArguments WorkerChannelArguments(const int keepalive_time_ms,
                                              const int keepalive_timeout_ms) {
  grpc::ChannelArguments channel_arguments;
  channel_arguments.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
  channel_arguments.SetMaxSendMessageSize(std::numeric_limits<int>::max());
  // Re-creating the channel after a failure opens a new connection (by
  // default, the channels to the same address share their connections).
  channel_arguments.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, 1);
  if (keepalive_time_ms > 0) {
    channel_arguments.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, keepalive_time_ms);
    channel_arguments.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS,
                             keepalive_timeout_ms);
    // The HTTP/2 transport closes the connection when a ping is not answered
    // within this timeout (default: 1 minute with keep-alive pings). In the
    // current gRPC version, "GRPC_ARG_KEEPALIVE_TIMEOUT_MS" alone is not
    // enforced. Internal argument "GRPC_ARG_PING_TIMEOUT_MS" of
    // "chttp2/transport/internal_channel_arg_names.h".
    channel_arguments.SetInt("grpc.http2.ping_timeout_ms",
                             keepalive_timeout_ms);
    channel_arguments.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
    // By default, after two pings without data (e.g. during a long request),
    // the pings are limited to one per minute.
    channel_arguments.SetInt(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA, 0);
  }
  return channel_arguments;
}

void AcceptKeepalivePings(grpc::ServerBuilder* builder) {
  builder->AddChannelArgument(
      GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS, 10000);
  // Never close a connection because of too frequent pings.
  builder->AddChannelArgument(GRPC_ARG_HTTP2_MAX_PING_STRIKES, 0);
}

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
