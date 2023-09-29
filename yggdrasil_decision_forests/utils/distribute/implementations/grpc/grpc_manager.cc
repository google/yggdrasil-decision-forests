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

#include <memory>
#include <random>

#include "grpcpp/create_channel.h"
#include "grpcpp/support/channel_arguments.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
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

// In addition to control available though the "GRPCManager" class (e.g. the
// GRPCManager::UpdateWorkerAddress function), GRPC manager can be referenced
// and configured using the session key (optional "key" field in the manager
// configuration). This is called a "global change".
//
// For example, calling "UpdateWorkerAddress(grpc_key, address)" is equivalent
// to calling  "manager->UpdateWorkerAddress(address)" if "manager" is
// configured with the "grpc_key" key.
//
// "GetGlobalChanges" returns a control structure for manager configuration
// through keys.

// Redefinition of the address of a worker.
struct UpdateAddress {
  int worker_idx;
  std::string new_address;
};

// Changes for a given key.
struct KeyChanges {
  // List of changes made after the manager creation.
  // Currently, "UpdateAddress" is the only type of changes.
  std::vector<UpdateAddress> pending_changes;

  // List of already applied changes.
  absl::flat_hash_map<int, std::string> past_changes;
};

struct GlobalChanges {
  absl::flat_hash_map<int, KeyChanges> per_key GUARDED_BY(mutex);
  utils::concurrency::Mutex mutex;
  utils::concurrency::CondVar cond_var;  // When a change is made.
};

GlobalChanges& GetGlobalChanges() {
  static GlobalChanges all_changes;
  return all_changes;
}

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
    case proto::GRPCImp::kGrpcAddresses:
      for (const auto& address : imp_config.grpc_addresses().addresses()) {
        worker_addresses.push_back(address);
      }
      break;
    default:
      return absl::UnimplementedError("Unknown worker address type");
  }

  if (worker_addresses.empty()) {
    return absl::InvalidArgumentError("There should be at least one worker");
  }

  // Override the worker address with global changes.
  if (imp_config.has_key()) {
    auto& all_events = GetGlobalChanges();
    utils::concurrency::MutexLock l(&all_events.mutex);
    auto& per_key = all_events.per_key[imp_config.key()];
    for (const auto& change : per_key.past_changes) {
      worker_addresses[change.first] = change.second;
    }
  }

  if (verbosity_ >= 1) {
    YDF_LOG(INFO) << "Start manager with " << worker_addresses.size()
                  << " workers and key="
                  << (key_.has_value() ? key_.value() : -1);
  }

  if (imp_config.use_loas()) {
    return absl::InvalidArgumentError("Loas not available");
  } else {
    credential_ = grpc::InsecureChannelCredentials();
  }

  for (int worker_idx = 0; worker_idx < worker_addresses.size(); worker_idx++) {
    auto worker = absl::make_unique<Worker>();
    worker->worker_idx = worker_idx;
    {
      utils::concurrency::MutexLock l(&worker->mutex_address);
      worker->expected_address = worker_addresses[worker_idx];
      worker->StartThreads(parallel_execution_per_worker, this);
      workers_.push_back(std::move(worker));
    }
  }

  for (auto& worker : workers_) {
    RETURN_IF_ERROR(UpdateWorkerConnection(worker.get()).status());
  }

  return absl::OkStatus();
}

absl::Status GRPCManager::WaitForAllWorkersToBeReady() {
  for (auto& worker : workers_) {
    while (true) {
      ASSIGN_OR_RETURN(auto stub, UpdateWorkerConnection(worker.get()));
      grpc::ClientContext context;
      ConfigureClientContext(&context);
      proto::Empty query;
      proto::Empty answer;
      const auto ping_status = stub->Ping(&context, query, &answer);
      if (!ping_status.ok()) {
        if (verbosity_ >= 1) {
          YDF_LOG(INFO) << "Worker #" << worker->worker_idx
                        << " is not yet available. Waiting 10s";
        }
        absl::SleepFor(absl::Seconds(10));
        continue;
      }
      break;
    }
  }

  if (verbosity_ >= 1) {
    YDF_LOG(INFO) << "All the workers are available";
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

  peer_worker_update_thread_ = absl::make_unique<utils::concurrency::Thread>(
      [this, manager]() { manager->ProcessPeerWorkerAddressUpdate(this); });
}

absl::StatusOr<int> GRPCManager::NumWorkersInConfiguration(
    const proto::Config& config) const {
  const auto& imp_config = config.GetExtension(proto::grpc);
  switch (imp_config.worker_address_case()) {
    case proto::GRPCImp::kSocketAddresses:
      return imp_config.socket_addresses().addresses_size();
    case proto::GRPCImp::kGrpcAddresses:
      return imp_config.grpc_addresses().addresses_size();
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

    YDF_LOG(INFO) << "Update address of worker #" << worker->worker_idx
                  << " from" << worker->connected_address << " to "
                  << worker->expected_address;

    worker->connected_address = worker->expected_address;

    if (worker->stub) {
      worker->discarded_stubs_.push_back(std::move(worker->stub));
      worker->stub.reset();
    }

    DCHECK(credential_);
    worker->stub = CreateStub(worker->connected_address, &credential_);
  }

  DCHECK(worker->stub);
  return worker->stub.get();
}

absl::Status GRPCManager::SetParallelExecutionPerWorker(int num) {
  if (verbosity_) {
    YDF_LOG(INFO) << "Change the number of parallel execution per worker";
  }

  // Close the query channels.
  async_pending_queries_.Close();
  for (auto& worker : workers_) {
    worker->async_pending_queries_.Close();
    worker->peer_worker_update_workers_.Close();
  }

  // Wait for the threads to join
  JoinWorkers();

  // Re-open the channels and restart the threads.
  async_pending_queries_.Reopen();
  for (auto& worker : workers_) {
    worker->async_pending_queries_.Reopen();
    worker->peer_worker_update_workers_.Reopen();
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
    ConfigureClientContext(&context);
    const auto status = stub->Run(&context, query, &answer);
    if (!status.ok()) {
      if (status.error_message() == "UNAVAILABLE: worker config required") {
        // The worker received the request, but the worker is lacking the worker
        // configuration field. The request should be re-sent with the worker
        // configuration.
        YDF_LOG(WARNING) << "Send worker configuration to worker #"
                         << worker->worker_idx;
        utils::concurrency::MutexLock l(&mutex_worker_config_);
        *query.mutable_worker_config() = worker_config_;
        continue;
      }

      if (verbosity_ >= 1) {
        YDF_LOG(WARNING) << "GRPC to worker #" << worker->worker_idx
                         << " failed with error: " << status.error_message();
      }
      if (IsTransientError(status)) {
        // The worker is temporarily not available.
        absl::SleepFor(absl::Seconds(5));
        ASSIGN_OR_RETURN(stub, UpdateWorkerConnection(worker));
        continue;
      } else {
        // Something is not right.
        YDF_LOG(INFO)
            << "Fatal error in GRPC communication. If this is in fact a "
               "transiant error, update \"IsTransiantError\" accordingly.";
        return GrpcStatusToAbslStatus(status);
      }
    }
    break;
  }

  if (answer.has_error()) {
    if (verbosity_ >= 1) {
      YDF_LOG(WARNING) << "Worker #" << worker->worker_idx
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
  utils::concurrency::MutexLock l(&mutex_worker_config_);
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
    YDF_LOG(INFO) << "Emitting blocking request of " << blob.size() << " bytes";
  }

  if (worker_idx < 0) {
    worker_idx = next_auto_worker_idx_.fetch_add(1) % workers_.size();
  }
  auto* worker = workers_[worker_idx].get();

  return WorkerRunImp(std::move(blob), worker);
}

absl::Status GRPCManager::AsynchronousRequest(Blob blob, int worker_idx) {
  if (verbosity_ >= 2) {
    YDF_LOG(INFO) << "Emitting asynchronous request of " << blob.size()
                  << " bytes";
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
    YDF_LOG(INFO) << "Shutdown manager with key="
                  << (key_.has_value() ? key_.value() : -1);
  }
  if (done_was_called_) {
    YDF_LOG(WARNING) << "Calling done twice";
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

    worker->peer_worker_update_workers_.Close();
    worker->peer_worker_update_workers_.Clear();
  }

  JoinWorkers();
  if (verbosity_ >= 1) {
    YDF_LOG(INFO) << "Workers joined";
  }

  proto::ShutdownQuery query;
  if (kill_worker_manager.has_value()) {
    query.set_kill_worker_manager(kill_worker_manager.value());
  } else {
    query.set_kill_worker_manager(false);
  }

  // TODO: Run in parallel.
  for (auto& worker : workers_) {
    ASSIGN_OR_RETURN(auto stub, UpdateWorkerConnection(worker.get()));

    grpc::ClientContext context;
    ConfigureClientContext(&context);

    proto::Empty ignored;
    auto worker_shutdown = stub->Shutdown(&context, query, &ignored);
    if (!worker_shutdown.ok()) {
      // It is not a big deal if the worker crashes during shutdown.
      YDF_LOG(WARNING) << "Error when shutting down the connection:"
                       << worker_shutdown.error_message();
    }
  }

  if (key_.has_value()) {
    StopEventCheckingThread();
  }

  if (verbosity_ >= 1) {
    YDF_LOG(INFO) << "Manager has been shutdown";
  }

  return absl::OkStatus();
}

absl::Status GRPCManager::DebugShutdownWorker(int worker_idx) {
  proto::ShutdownQuery query;
  query.set_kill_worker_manager(true);

  grpc::ClientContext context;
  ConfigureClientContext(&context);
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
    worker->peer_worker_update_thread_->Join();
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
    YDF_LOG(INFO) << "Initialize manager with " << welcome_blob.size()
                  << " bytes welcome blob, uid:" << manager_uid_;
  }
  RETURN_IF_ERROR(InitializeWorkers(config, parallel_execution_per_worker));
  RETURN_IF_ERROR(InitializeConfigFile(config, worker_name,
                                       parallel_execution_per_worker,
                                       std::move(welcome_blob)));
  const auto& imp_config = config.GetExtension(proto::grpc);
  if (imp_config.has_key()) {
    key_ = imp_config.key();
    StartEventCheckingThread();
  }
  RETURN_IF_ERROR(WaitForAllWorkersToBeReady());
  return absl::OkStatus();
}

absl::Status GRPCManager::UpdateWorkerAddress(
    const int worker_idx, const absl::string_view new_address) {
  DCHECK_GE(worker_idx, 0);
  DCHECK_LT(worker_idx, workers_.size());
  auto& worker = workers_[worker_idx];
  {
    utils::concurrency::MutexLock l(&worker->mutex_address);
    worker->expected_address = std::string(new_address);
  }
  {
    utils::concurrency::MutexLock l(&mutex_worker_config_);
    *worker_config_.mutable_worker_addresses(worker_idx) =
        std::string(new_address);
  }
  for (auto& update_worker : workers_) {
    if (update_worker->worker_idx == worker_idx) {
      continue;
    }
    update_worker->peer_worker_update_workers_.Push(worker_idx);
  }
  return absl::OkStatus();
}

void GRPCManager::ProcessPeerWorkerAddressUpdate(Worker* worker) {
  while (true) {
    auto worker_idx_or = worker->peer_worker_update_workers_.Pop();
    if (!worker_idx_or.has_value()) {
      break;
    }

    proto::UpdateWorkerAddressQuery query;
    query.set_worker_idx(worker_idx_or.value());

    // Get the new address of those workers.
    {
      auto& target_worker = workers_[query.worker_idx()];
      utils::concurrency::MutexLock l(&target_worker->mutex_address);
      query.set_new_address(target_worker->expected_address);
    }

    // Send update.
    while (!done_was_called_) {
      auto stub_or = UpdateWorkerConnection(worker);
      if (!stub_or.ok()) {
        YDF_LOG(WARNING) << "Cannot create stub";
        continue;
      }

      grpc::ClientContext context;
      ConfigureClientContext(&context);
      proto::Empty ignored;
      auto worker_shutdown =
          stub_or.value()->UpdateWorkerAddress(&context, query, &ignored);

      if (worker_shutdown.ok()) {
        break;
      }
    }
  }
}

void UpdateWorkerAddress(int key, int worker_idx,
                         absl::string_view new_address) {
  YDF_LOG(INFO) << "Receive update of worker #" << worker_idx << " address to "
                << new_address;
  auto& all_changes = GetGlobalChanges();
  utils::concurrency::MutexLock l(&all_changes.mutex);
  auto& per_key = all_changes.per_key[key];

  per_key.pending_changes.push_back(
      UpdateAddress{/*worker_idx=*/worker_idx,
                    /*new_address=*/std::string(new_address)});
  per_key.past_changes[worker_idx] = std::string(new_address);
  all_changes.cond_var.SignalAll();
}

void GRPCManager::StartEventCheckingThread() {
  DCHECK(!event_checking_thread_);
  event_checking_thread_ = absl::make_unique<utils::concurrency::Thread>(
      [this]() { MainEventCheckingThread(); });
}

void GRPCManager::StopEventCheckingThread() {
  if (event_checking_thread_) {
    DCHECK(done_was_called_);
    event_checking_thread_->Join();
  }
}

void GRPCManager::MainEventCheckingThread() {
  auto& all_changes = GetGlobalChanges();
  while (!done_was_called_) {
    std::vector<UpdateAddress> pending_changes;
    {
      utils::concurrency::MutexLock lock(&all_changes.mutex);

      // Wait for changes.
      absl::flat_hash_map<int, KeyChanges>::iterator it_per_key;
      while (!done_was_called_) {
        it_per_key = all_changes.per_key.find(key_.value());
        if (it_per_key == all_changes.per_key.end() ||
            it_per_key->second.pending_changes.empty()) {
          all_changes.cond_var.WaitWithTimeout(&all_changes.mutex, &lock, 10);
          continue;
        }
        break;
      }

      pending_changes = std::move(it_per_key->second.pending_changes);
      it_per_key->second.pending_changes.clear();
    }

    // Execute events.
    for (const auto& change : pending_changes) {
      auto status = UpdateWorkerAddress(change.worker_idx, change.new_address);
      if (!status.ok()) {
        YDF_LOG(WARNING) << "Cannot update worker address: "
                         << status.message();
      }
    }
  }
}

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
