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

#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_manager.h"

#include <random>

#include "grpcpp/create_channel.h"
#include "grpcpp/support/channel_arguments.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc.grpc.pb.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_common.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests {
namespace distribute {
namespace {

constexpr int kDeadLineInHours = 24 * 40;

absl::Status GrpcStatusToAbslStatus(const grpc::Status& src) {
  if (src.ok()) {
    return absl::Status();
  } else {
    return absl::UnknownError(src.error_message());
  }
}

// Creates a connection (called "stub" in grpc).
std::unique_ptr<proto::Server::Stub> CreateStub(
    const absl::string_view address,
    std::shared_ptr<grpc::ChannelCredentials>* credential) {
  grpc::ChannelArguments channel_arguments;
  channel_arguments.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
  channel_arguments.SetMaxSendMessageSize(std::numeric_limits<int>::max());
  auto channel = grpc::CreateCustomChannel(std::string(address), *credential,
                                           channel_arguments);
  return proto::Server::NewStub(channel);
}

}  // namespace

constexpr char GRPCManager::kKey[];

absl::Status GRPCManager::InitializeWorkers(
    const proto::Config& config, const int parallel_execution_per_worker) {
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
  if (verbosity_ >= 1) {
    LOG(INFO) << "Start manager with " << worker_addresses.size()
              << " workers.";
  }

  if (imp_config.use_loas()) {
    return absl::InvalidArgumentError("Loas not available");
  } else {
    credential_ = grpc::InsecureChannelCredentials();
  }

  for (int worker_idx = 0; worker_idx < worker_addresses.size(); worker_idx++) {
    auto worker = absl::make_unique<Worker>();
    worker->worker_idx = worker_idx;

    while (true) {
      utils::concurrency::MutexLock l(&worker->mutex_address);
      worker->stub = CreateStub(worker_addresses[worker_idx], &credential_);

      grpc::ClientContext context;
      proto::Empty query;
      proto::Empty answer;
      const auto ping_status = worker->stub->Ping(&context, query, &answer);
      if (!ping_status.ok()) {
        if (verbosity_ >= 1) {
          LOG(INFO) << "Worker #" << worker_idx
                    << " is not yet available. Waiting 10s";
        }
        absl::SleepFor(absl::Seconds(10));
        continue;
      }

      break;
    }

    utils::concurrency::MutexLock l(&worker->mutex_address);
    worker->connected_address = worker_addresses[worker_idx];
    worker->expected_address = worker->connected_address;
    worker->StartThreads(parallel_execution_per_worker, this);
    workers_.push_back(std::move(worker));
  }

  if (verbosity_ >= 1) {
    LOG(INFO) << "All the workers are available";
  }

  return absl::OkStatus();
}

void GRPCManager::Worker::StartThreads(int parallel_execution_per_worker,
                                       GRPCManager* manager) {
  process_local_queries.Start(parallel_execution_per_worker, [this, manager]() {
    manager->ProcessLocalQueries(this);
  });

  process_global_queries.Start(
      parallel_execution_per_worker,
      [this, manager]() { manager->ProcessGlobalQueries(this); });
}

absl::StatusOr<int> GRPCManager::NumWorkersInConfiguration(
    const proto::Config& config) const {
  const auto& imp_config = config.GetExtension(proto::grpc);
  switch (imp_config.worker_address_case()) {
    case proto::GRPCImp::kSocketAddresses:
      return imp_config.socket_addresses().addresses_size();
    case proto::GRPCImp::kBns:
      return imp_config.bns().num_workers();
    default:
      return absl::UnimplementedError("Unknown worker address type");
  }
}

absl::StatusOr<proto::Server::Stub*> GRPCManager::UpdateWorkerConnection(
    Worker* worker) {
  utils::concurrency::MutexLock l(&worker->mutex_address);
  if (worker->expected_address != worker->connected_address) {
    // The worker has moved.
    LOG(WARNING) << "Changing worker address from " << worker->connected_address
                 << " to " << worker->expected_address;
    worker->connected_address = worker->expected_address;
    worker->discarded_stubs_.push_back(std::move(worker->stub));
    worker->stub.reset();

    worker->stub = CreateStub(worker->connected_address, &credential_);
  }
  return worker->stub.get();
}

absl::Status GRPCManager::SetParallelExecutionPerWorker(int num) {
  if (verbosity_) {
    LOG(INFO) << "Change the number of parallel execution per worker";
  }

  // Close the query channels.
  async_pending_queries_.Close();
  for (auto& worker : workers_) {
    worker->async_pending_queries_.Close();
  }

  // Wait for the threads to join
  JoinWorkers();

  // Re-open the channels and restart the threads.
  async_pending_queries_.Reopen();
  for (auto& worker : workers_) {
    worker->async_pending_queries_.Reopen();
    worker->StartThreads(num, this);
  }
  return absl::OkStatus();
}

absl::StatusOr<Blob> GRPCManager::WorkerRunImp(Blob blob, Worker* worker) {
  ASSIGN_OR_RETURN(auto stub, UpdateWorkerConnection(worker));

  proto::Query query;
  *query.mutable_blob() = std::move(blob);
  query.set_manager_uid(manager_uid_);
  query.set_worker_idx(worker->worker_idx);

  proto::Answer answer;
  while (true) {
    grpc::ClientContext context;
    context.set_wait_for_ready(true);
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::hours(kDeadLineInHours));
    const auto status = stub->Run(&context, query, &answer);
    if (!status.ok()) {
      if (status.error_message() == "UNAVAILABLE: worker config required") {
        // The worker received the request, but the worker is lacking the worker
        // configuration field. The request should be re-sent with the worker
        // configuration.
        LOG(WARNING) << "Send worker configuration to worker #"
                     << worker->worker_idx;
        *query.mutable_worker_config() = worker_config_;
        continue;
      }

      if (verbosity_ >= 1) {
        LOG(WARNING) << "GRPC to worker #" << worker->worker_idx
                     << " failed with error: " << status.error_message();
      }
      if (IsTransiantError(status)) {
        // The worker is temporarly not available.
        ASSIGN_OR_RETURN(stub, UpdateWorkerConnection(worker));
        continue;
      } else {
        // Something is not right.
        return GrpcStatusToAbslStatus(status);
      }
    }
    break;
  }

  if (answer.has_error()) {
    if (verbosity_ >= 1) {
      LOG(WARNING) << "Worker #" << worker->worker_idx
                   << " returned an error: " << answer.error();
    }
    return absl::InvalidArgumentError(answer.error());
  }
  return std::move(*answer.mutable_blob());
}

void GRPCManager::WorkerRun(Blob blob, Worker* worker) {
  auto answer_or = WorkerRunImp(std::move(blob), worker);
  if (!answer_or.ok()) {
    async_pending_answers_.Push(answer_or.status());
  } else {
    async_pending_answers_.Push(std::move(answer_or).value());
  }
}

void GRPCManager::ProcessLocalQueries(Worker* worker) {
  while (true) {
    auto pending_blob_or = worker->async_pending_queries_.Pop();
    if (!pending_blob_or.has_value()) {
      break;
    }
    WorkerRun(std::move(pending_blob_or.value()), worker);
  }
}

void GRPCManager::ProcessGlobalQueries(Worker* worker) {
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
    const int parallel_execution_per_worker, Blob welcome_blob) {
  worker_config_.set_worker_name(std::string(worker_name));
  worker_config_.set_welcome_blob(welcome_blob);
  worker_config_.set_manager_uid(manager_uid_);
  worker_config_.set_parallel_execution_per_worker(
      parallel_execution_per_worker);

  for (const auto& worker : workers_) {
    utils::concurrency::MutexLock l(&worker->mutex_address);
    worker_config_.add_worker_addresses(worker->expected_address);
  }
  return absl::OkStatus();
}

absl::StatusOr<Blob> GRPCManager::BlockingRequest(Blob blob, int worker_idx) {
  if (verbosity_ >= 2) {
    LOG(INFO) << "Emitting blocking request of " << blob.size() << " bytes";
  }

  if (worker_idx < 0) {
    worker_idx = next_auto_worker_idx_.fetch_add(1) % workers_.size();
  }
  auto* worker = workers_[worker_idx].get();

  return WorkerRunImp(std::move(blob), worker);
}

absl::Status GRPCManager::AsynchronousRequest(Blob blob, int worker_idx) {
  if (verbosity_ >= 2) {
    LOG(INFO) << "Emitting asynchronous request of " << blob.size() << " bytes";
  }
  if (worker_idx < 0) {
    async_pending_queries_.Push(std::move(blob));
  } else {
    workers_[worker_idx]->async_pending_queries_.Push(std::move(blob));
  }
  return absl::OkStatus();
}

absl::StatusOr<Blob> GRPCManager::NextAsynchronousAnswer() {
  auto answer_or = async_pending_answers_.Pop();
  if (!answer_or.has_value()) {
    return absl::OutOfRangeError("No more results available");
  }
  if (!answer_or.value().ok()) {
    return answer_or.value();
  }
  return std::move(*answer_or.value());
}

int GRPCManager::NumWorkers() { return workers_.size(); }

absl::Status GRPCManager::Done(absl::optional<bool> kill_worker_manager) {
  if (verbosity_ >= 1) {
    LOG(INFO) << "Shutdown manager";
  }
  if (done_was_called_) {
    LOG(WARNING) << "Calling done twice";
    return absl::OkStatus();
  }
  done_was_called_ = true;
  async_pending_queries_.Close();
  async_pending_answers_.Close();

  async_pending_queries_.Clear();
  async_pending_answers_.Clear();

  for (auto& worker : workers_) {
    worker->async_pending_queries_.Close();
    worker->async_pending_queries_.Clear();
  }

  JoinWorkers();
  if (verbosity_ >= 1) {
    LOG(INFO) << "Workers joined";
  }

  proto::ShutdownQuery query;
  if (kill_worker_manager.has_value()) {
    query.set_kill_worker_manager(kill_worker_manager.value());
  } else {
    query.set_kill_worker_manager(false);
  }

  // TODO: Run in parallel.
  for (auto& worker : workers_) {
    grpc::ClientContext context;
    context.set_wait_for_ready(false);
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::minutes(2));
    proto::Empty ignored;
    utils::concurrency::MutexLock l(&worker->mutex_address);
    auto worker_shutdown = worker->stub->Shutdown(&context, query, &ignored);
    if (!worker_shutdown.ok()) {
      // It is not a big deal if the worker crashes during shutdown.
      LOG(WARNING) << "Error when shutting down the connection:"
                   << worker_shutdown.error_message();
    }
  }

  if (verbosity_ >= 1) {
    LOG(INFO) << "Manager has been shutdown";
  }

  return absl::OkStatus();
}

absl::Status GRPCManager::DebugShutdownWorker(int worker_idx) {
  proto::ShutdownQuery query;
  query.set_kill_worker_manager(true);

  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  context.set_deadline(std::chrono::system_clock::now() +
                       std::chrono::minutes(2));
  proto::Empty ignored;
  auto& worker = workers_[worker_idx];
  utils::concurrency::MutexLock l(&worker->mutex_address);
  auto worker_shutdown = worker->stub->Shutdown(&context, query, &ignored);
  return GrpcStatusToAbslStatus(worker_shutdown);
}

void GRPCManager::JoinWorkers() {
  for (auto& worker : workers_) {
    worker->process_local_queries.JoinAndClear();
    worker->process_global_queries.JoinAndClear();
  }
}

absl::Status GRPCManager::Initialize(const proto::Config& config,
                                     const absl::string_view worker_name,
                                     Blob welcome_blob,
                                     const int parallel_execution_per_worker) {
  verbosity_ = config.verbosity();

  // Generate manager uid.  Used to distinguish between the different managers
  // controlling a same pool of workers.
  std::random_device rnd;
  manager_uid_ = std::uniform_int_distribution<uint64_t>(
      std::numeric_limits<uint64_t>::lowest(),
      std::numeric_limits<uint64_t>::max())(rnd);

  if (verbosity_ >= 1) {
    LOG(INFO) << "Initialize manager with " << welcome_blob.size()
              << " bytes welcome blob, uid:" << manager_uid_;
  }
  RETURN_IF_ERROR(InitializeWorkers(config, parallel_execution_per_worker));
  RETURN_IF_ERROR(InitializeConfigFile(config, worker_name,
                                       parallel_execution_per_worker,
                                       std::move(welcome_blob)));
  return absl::OkStatus();
}

absl::Status GRPCManager::UpdateWorkerAddress(
    const int worker_idx, const absl::string_view new_address) {
  DCHECK_GE(worker_idx, 0);
  DCHECK_LT(worker_idx, workers_.size());
  auto& worker = workers_[worker_idx];
  utils::concurrency::MutexLock l(&worker->mutex_address);
  worker->expected_address = std::string(new_address);
  *worker_config_.mutable_worker_addresses(worker_idx) =
      std::string(new_address);

  // Send change to all workers.

  proto::UpdateWorkerAddressQuery query;
  query.set_worker_idx(worker_idx);
  *query.mutable_new_address() = std::string(new_address);

  // TODO: Run in parallel.
  for (auto& worker : workers_) {
    if (worker->worker_idx == worker_idx) {
      continue;
    }
    grpc::ClientContext context;
    context.set_wait_for_ready(false);
    context.set_deadline(std::chrono::system_clock::now() +
                         std::chrono::minutes(2));
    proto::Empty ignored;
    auto worker_shutdown =
        worker->stub->UpdateWorkerAddress(&context, query, &ignored);
    if (!worker_shutdown.ok()) {
      // We don't care if the worker is down. It will receive the update when it
      // gets back online.
    }
  }

  return absl::OkStatus();
}

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
