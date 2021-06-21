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

#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc.h"

#include <random>

#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/channel_arguments.h"
#include "absl/synchronization/notification.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc.grpc.pb.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace distribute {
namespace {

constexpr int kDeadLineInHours = 24 * 40;

grpc::Status AbslStatusToGrpcStatus(const absl::Status& src) {
  if (src.ok()) {
    return grpc::Status();
  } else {
    return grpc::Status(grpc::StatusCode::UNKNOWN, src.ToString());
  }
}

absl::Status GrpcStatusToAbslStatus(const grpc::Status& src) {
  if (src.ok()) {
    return absl::Status();
  } else {
    return absl::UnknownError(src.error_message());
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

absl::Status GRPCManager::InitializeWorkers(const proto::Config& config) {
  const auto& imp_config = config.GetExtension(proto::grpc);

  std::vector<std::string> worker_addresses;
  switch (imp_config.worker_address_case()) {
    case proto::GRPCImp::kSocketAddresses:
      for (const auto& address : imp_config.socket_addresses().addresses()) {
        worker_addresses.push_back(
            absl::StrCat(address.ip(), ":", address.port()));
      }
      break;
    case proto::GRPCImp::kBns:
      for (int worker_idx = 0; worker_idx < imp_config.bns().num_workers();
           worker_idx++) {
        // const auto worker_address =
        //     BuildURI(imp_config.bns().prefix(), worker_idx);
        worker_addresses.push_back(
            absl::StrCat(imp_config.bns().prefix(), "/", worker_idx));
      }
      break;
    default:
      return absl::UnimplementedError("Unknown worker address type");
  }

  if (worker_addresses.empty()) {
    return absl::InvalidArgumentError("There should be at least one worker");
  }
  if (verbose_) {
    LOG(INFO) << "Start manager with " << worker_addresses.size()
              << " workers.";
  }

  std::shared_ptr<grpc::ChannelCredentials> credential;
  if (imp_config.use_loas()) {
    return absl::InvalidArgumentError("Loas not available");
  } else {
    credential = grpc::InsecureChannelCredentials();
  }

  for (int worker_idx = 0; worker_idx < worker_addresses.size(); worker_idx++) {
    auto worker = absl::make_unique<Worker>();
    worker->worker_idx = worker_idx;

    while (true) {
      grpc::ChannelArguments channel_arguments;
      channel_arguments.SetMaxReceiveMessageSize(
          std::numeric_limits<int>::max());
      channel_arguments.SetMaxSendMessageSize(std::numeric_limits<int>::max());

      worker->channel = grpc::CreateCustomChannel(
          worker_addresses[worker_idx], credential, channel_arguments);
      worker->stub = proto::Server::NewStub(worker->channel);

      grpc::ClientContext context;
      proto::Empty query;
      proto::Empty answer;
      const auto ping_status = worker->stub->Ping(&context, query, &answer);
      if (!ping_status.ok()) {
        if (verbose_) {
          LOG(INFO) << "Worker #" << worker_idx
                    << " is not yet available. Waiting 10s";
        }
        absl::SleepFor(absl::Seconds(10));
        continue;
      }

      break;
    }

    worker->address = worker_addresses[worker_idx];

    worker->main_thread_1 = absl::make_unique<utils::concurrency::Thread>(
        [this, worker = worker.get()]() { WorkerMain1(worker); });

    worker->main_thread_2 = absl::make_unique<utils::concurrency::Thread>(
        [this, worker = worker.get()]() { WorkerMain2(worker); });

    workers_.push_back(std::move(worker));
  }

  if (verbose_) {
    LOG(INFO) << "All the workers are available";
  }

  return absl::OkStatus();
}

void GRPCManager::WorkerRun(Blob blob, Worker* worker) {
  proto::Query query;
  *query.mutable_blob() = std::move(blob);
  query.set_config_path(worker_config_path_);
  query.set_manager_uid(manager_uid_);
  query.set_worker_idx(worker->worker_idx);

  int num_re_emitting = 0;
  proto::Answer answer;
  while (true) {
    grpc::ClientContext context;
    context.set_wait_for_ready(true);
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::hours(kDeadLineInHours));
    const auto status = worker->stub->Run(&context, query, &answer);

    if (!status.ok()) {
      if (verbose_) {
        LOG(WARNING) << "GRPC call to worker #" << worker->worker_idx
                     << " failed with error: " << status.error_message();
      }
      if (status.error_message() == "Socket closed") {
        // The worker died during the execution (e.g. rescheduling).
        // Let's try again.
        if (verbose_) {
          num_re_emitting++;
          LOG(WARNING) << "Re-emitting request (num_re_emitting:"
                       << num_re_emitting << ")";
        }
        continue;
      } else {
        // Something is not right.
        answer.set_error(status.error_message());
        async_pending_answers_.Push(std::move(answer));
        return;
      }
    } else {
      if (verbose_ && answer.has_error()) {
        LOG(WARNING) << "Worker #" << worker->worker_idx
                     << " returned an error: " << answer.error();
      }
      async_pending_answers_.Push(std::move(answer));
      return;
    }
  }
}

void GRPCManager::WorkerMain1(Worker* worker) {
  while (true) {
    auto pending_blob_or = worker->async_pending_queries_.Pop();
    if (!pending_blob_or.has_value()) {
      break;
    }
    WorkerRun(std::move(pending_blob_or.value()), worker);
  }
}

void GRPCManager::WorkerMain2(Worker* worker) {
  while (true) {
    auto pending_blob_or = async_pending_queries_.Pop();
    if (!pending_blob_or.has_value()) {
      break;
    }
    WorkerRun(std::move(pending_blob_or.value()), worker);
  }
}

absl::Status GRPCManager::InitializeConfigFile(
    const proto::Config& config, const absl::string_view worker_name,
    const Blob welcome_blob) {
  if (config.working_directory().empty()) {
    return absl::InvalidArgumentError("The worker directory cannot be empty.");
  }
  RETURN_IF_ERROR(
      file::RecursivelyCreateDir(config.working_directory(), file::Defaults()));
  worker_config_path_ =
      file::JoinPath(config.working_directory(), "config.pbbin");
  proto::WorkerConfig worker_config;
  worker_config.set_worker_name(std::string(worker_name));
  worker_config.set_welcome_blob(welcome_blob);
  worker_config.set_manager_uid(manager_uid_);
  for (const auto& worker : workers_) {
    worker_config.add_worker_addresses(worker->address);
  }
  RETURN_IF_ERROR(file::SetBinaryProto(worker_config_path_, worker_config,
                                       file::Defaults()));
  return absl::OkStatus();
}

utils::StatusOr<Blob> GRPCManager::BlockingRequest(Blob blob, int worker_idx) {
  if (verbose_) {
    LOG(INFO) << "Incoming blocking request with " << blob.size() << " bytes";
  }

  if (worker_idx < 0) {
    worker_idx = next_auto_worker_idx_.fetch_add(1) % workers_.size();
  }
  auto* worker = workers_[worker_idx].get();

  proto::Query query;
  *query.mutable_blob() = std::move(blob);
  query.set_config_path(worker_config_path_);
  query.set_manager_uid(manager_uid_);
  query.set_worker_idx(worker->worker_idx);

  proto::Answer answer;
  while (true) {
    grpc::ClientContext context;
    context.set_wait_for_ready(true);
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::hours(kDeadLineInHours));
    const auto status = worker->stub->Run(&context, query, &answer);
    if (!status.ok()) {
      if (verbose_) {
        LOG(WARNING) << "GRPC to worker #" << worker_idx
                     << " failed with error: " << status.error_message();
      }
      if (status.error_message() == "Socket closed") {
        // The worker died during the execution (e.g. rescheduling).
        // Let's try again.
        continue;
      } else {
        // Something is not right.
        return GrpcStatusToAbslStatus(status);
      }
    }
    break;
  }

  if (answer.has_error()) {
    if (verbose_) {
      LOG(WARNING) << "Worker #" << worker_idx
                   << " returned an error: " << answer.error();
    }
    return absl::InvalidArgumentError(answer.error());
  }
  return std::move(*answer.mutable_blob());
}

absl::Status GRPCManager::AsynchronousRequest(Blob blob, int worker_idx) {
  if (verbose_) {
    LOG(INFO) << "Incoming asynchronous request with " << blob.size()
              << " bytes";
  }
  if (worker_idx < 0) {
    async_pending_queries_.Push(std::move(blob));
  } else {
    workers_[worker_idx]->async_pending_queries_.Push(std::move(blob));
  }
  return absl::OkStatus();
}

utils::StatusOr<Blob> GRPCManager::NextAsynchronousAnswer() {
  auto answer_or = async_pending_answers_.Pop();
  if (!answer_or.has_value()) {
    return absl::InvalidArgumentError("No more results available");
  }
  if (answer_or.value().has_error()) {
    return absl::InvalidArgumentError(answer_or.value().error());
  }
  return std::move(*answer_or.value().mutable_blob());
}

int GRPCManager::NumWorkers() { return workers_.size(); }

absl::Status GRPCManager::Done(absl::optional<bool> kill_worker_manager) {
  if (verbose_) {
    LOG(INFO) << "Shutdown manager";
  }
  if (done_was_called_) {
    LOG(WARNING) << "Calling done twice";
    return absl::OkStatus();
  }
  done_was_called_ = true;
  async_pending_queries_.Close();
  async_pending_answers_.Close();

  for (auto& worker : workers_) {
    worker->async_pending_queries_.Close();
  }

  JoinWorkers();

  proto::ShutdownQuery query;
  if (kill_worker_manager.has_value()) {
    query.set_kill_worker_manager(kill_worker_manager.value());
  } else {
    query.set_kill_worker_manager(false);
  }

  // TODO: Run in parallel.
  for (auto& worker : workers_) {
    grpc::ClientContext context;
    proto::Empty ignored;
    auto worker_shutdown = worker->stub->Shutdown(&context, query, &ignored);
    if (!worker_shutdown.ok()) {
      // It is not a big deal if the worker crashes during shutdown.
      LOG(WARNING) << "Error when shutting down the connection:"
                   << worker_shutdown.error_message();
    }
  }

  if (verbose_) {
    LOG(INFO) << "Manager has been shutdown";
  }

  return absl::OkStatus();
}

void GRPCManager::JoinWorkers() {
  for (auto& worker : workers_) {
    worker->main_thread_1->Join();
    worker->main_thread_2->Join();
  }
}

absl::Status GRPCManager::Initialize(const proto::Config& config,
                                     const absl::string_view worker_name,
                                     Blob welcome_blob) {
  verbose_ = config.verbose();

  // Generate manager uid.  Used to distinguish between the different managers
  // controlling a same pool of workers.
  std::random_device rnd;
  manager_uid_ = std::uniform_int_distribution<uint64_t>(
      std::numeric_limits<uint64_t>::lowest(),
      std::numeric_limits<uint64_t>::max())(rnd);

  if (verbose_) {
    LOG(INFO) << "Initialize manager with " << welcome_blob.size()
              << " bytes welcome blob, uid:" << manager_uid_;
  }
  RETURN_IF_ERROR(InitializeWorkers(config));
  RETURN_IF_ERROR(
      InitializeConfigFile(config, worker_name, std::move(welcome_blob)));
  return absl::OkStatus();
}

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
