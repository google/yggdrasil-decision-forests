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

#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_worker.h"

#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "absl/synchronization/notification.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc.grpc.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace distribute {
namespace {

grpc::Status AbslStatusToGrpcStatus(const absl::Status& src) {
  if (src.ok()) {
    return grpc::Status();
  } else {
    return grpc::Status(grpc::StatusCode::UNKNOWN, src.ToString());
  }
}

class WorkerService final : public proto::Server::Service {
 public:
  WorkerService(absl::Notification* stop_server) : stop_server_(stop_server) {}

 private:
  grpc::Status Run(grpc::ServerContext* context, const proto::Query* request,
                   proto::Answer* reply) override {
    RETURN_IF_ERROR(AbslStatusToGrpcStatus(ReadyWorker(request->manager_uid(),
                                                       request->config_path(),
                                                       request->worker_idx())));

    auto result_or = worker_->RunRequest(request->blob());

    if (!result_or.ok()) {
      reply->set_error(absl::StrCat("Worker #", request->worker_idx(), ": ",
                                    result_or.status().ToString()));
    } else {
      *reply->mutable_blob() = std::move(result_or).value();
    }
    return grpc::Status::OK;
  }

  grpc::Status Shutdown(grpc::ServerContext* context,
                        const proto::ShutdownQuery* request,
                        proto::Empty* reply) override {
    if (worker_) {
      RETURN_IF_ERROR(AbslStatusToGrpcStatus(worker_->Done()));
      old_workers_.push_back(std::move(worker_));
      worker_.reset();
    }
    if (request->kill_worker_manager() && !stop_server_->HasBeenNotified()) {
      stop_server_->Notify();
    }
    return grpc::Status::OK;
  }

  grpc::Status Ping(grpc::ServerContext* context, const proto::Empty* request,
                    proto::Empty* reply) override {
    LOG(INFO) << "Reply ping";
    return grpc::Status::OK;
  }

  // Ensures the worker is ready to answer queries.
  absl::Status ReadyWorker(uint64_t manager_uid, absl::string_view config_path,
                           const int worker_idx) {
    if (worker_ && manager_uid_ != manager_uid) {
      // The manager has changed.
      LOG(INFO) << "The manager has changed.";
      worker_.reset();
    }
    manager_uid_ = manager_uid;

    proto::WorkerConfig worker_config;
    RETURN_IF_ERROR(
        file::GetBinaryProto(config_path, &worker_config, file::Defaults()));
    if (worker_config.manager_uid() != manager_uid_) {
      return absl::InvalidArgumentError(
          "Two managers are fighting for the same worker.");
    }

    ASSIGN_OR_RETURN(
        worker_, AbstractWorkerRegisterer::Create(worker_config.worker_name()));
    RETURN_IF_ERROR(InternalInitializeWorker(worker_idx, worker_.get()));
    RETURN_IF_ERROR(worker_->Setup(worker_config.welcome_blob()));
    return absl::OkStatus();
  }

  // Non owning pointer to the notification that stops the server.
  absl::Notification* stop_server_ = nullptr;

  // Active worker.
  std::unique_ptr<AbstractWorker> worker_;

  // Inactive workers. "Done" was called on these workers, but they might still
  // be running threads.
  std::vector<std::unique_ptr<AbstractWorker>> old_workers_;

  // UID of the manager. Only valid if worker_ is set.
  uint64_t manager_uid_;
};

}  // namespace

absl::Status GRPCWorkerMainWorkerMain(const int port, bool use_loas) {
  absl::Notification stop_server;
  WorkerService service(&stop_server);

  std::shared_ptr<grpc::ServerCredentials> credential;
  if (use_loas) {
    return absl::InvalidArgumentError("Loas not available");
  } else {
    credential = grpc::InsecureServerCredentials();
  }

  grpc::ServerBuilder builder;
  std::string server_address = absl::StrCat("[::]:", port);
  LOG(INFO) << "Start worker server at address " << server_address;
  builder.AddListeningPort(server_address, credential);

  builder.RegisterService(&service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (!server) {
    return absl::UnknownError("Could not start the worker GRPC server");
  }

  utils::concurrency::Thread server_thread([&]() { server->Wait(); });
  stop_server.WaitForNotification();
  absl::SleepFor(absl::Seconds(1));
  server->Shutdown();
  server_thread.Join();
  return absl::OkStatus();
}

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
