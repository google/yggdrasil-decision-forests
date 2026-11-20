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

#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_worker.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "grpcpp/create_channel.h"
#include "grpcpp/loas2_credentials_options.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/channel_arguments.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/grpc/include/grpcpp/security/credentials.h"
#include "third_party/grpc/include/grpcpp/security/server_credentials.h"
#include "third_party/grpc/include/grpcpp/server_builder.h"
#include "third_party/grpc/include/grpcpp/server_context.h"
#include "third_party/grpc/include/grpcpp/support/status.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc.grpc.pb.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_common.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests {
namespace distribute {
namespace grpc_worker {
namespace {

// Maximum execution time of a request.
constexpr int kDeadLineInHours = 24 * 40;

// Threads of the synchronous gRPC server of the worker polling for new
// requests (see "StartGRPCWorker"): at least "kMinPollers" (gRPC's default: 1),
// and the threads that finish a request poll for the next ones as long as
// there are fewer than "kMaxPollers" (gRPC's default: 2), instead of exiting.
// Constants: the server starts before the worker receives its configuration.
constexpr int kMinPollers = 4;
constexpr int kMaxPollers = 128;

// Converts an Absl Status into a GRPC Status.
grpc::Status AbslStatusToGrpcStatus(const absl::Status& src) {
  if (src.ok()) {
    return grpc::Status();
  } else {
    return grpc::Status(grpc::StatusCode::UNKNOWN, src.ToString());
  }
}

}  // namespace

namespace internal {

WorkerService::WorkerService(utils::concurrency::Notification* stop_server,
                             bool use_loas)
    : stop_server_(stop_server), use_loas_(use_loas), hook_(this) {}

void WorkerService::ShutDown() { FinalizeIntraWorkerCommunication(); }

absl::Status WorkerService::AsynchronousRequestToOtherWorker(
    Blob blob, int target_worker_idx, AbstractWorker* emitter_worker) {
  intra_worker_communication_->pending_queries.Push(
      std::make_pair(target_worker_idx, std::move(blob)));
  return absl::OkStatus();
}

absl::StatusOr<Blob> WorkerService::NextAsynchronousAnswerFromOtherWorker(
    AbstractWorker* emitter_worker) {
  auto answer = intra_worker_communication_->pending_answers.Pop();
  if (!answer.has_value()) {
    return absl::OutOfRangeError("No more results available");
  }
  return std::move(answer.value());
}

absl::Status WorkerHook::AsynchronousRequestToOtherWorker(
    Blob blob, int target_worker_idx, AbstractWorker* emitter_worker) {
  return parent_->AsynchronousRequestToOtherWorker(
      std::move(blob), target_worker_idx, emitter_worker);
}

absl::StatusOr<Blob> WorkerHook::NextAsynchronousAnswerFromOtherWorker(
    AbstractWorker* emitter_worker) {
  return parent_->NextAsynchronousAnswerFromOtherWorker(emitter_worker);
}

absl::StatusOr<Blob> WorkerHook::BlockingRequestToOtherWorker(
    Blob blob, int target_worker_idx, AbstractWorker* emitter_worker) {
  // The worker only runs requests between its setup and "Done": the
  // inter-worker connections exist.
  return parent_->BlockingInterWorkerRequest(std::move(blob),
                                             target_worker_idx);
}

absl::StatusOr<Blob> WorkerService::RunQueryOnce(const proto::Query& request) {
  if (!request.has_query_id()) {
    return worker_->RunRequest(request.blob());
  }
  const std::pair<uint64_t, uint64_t> key = {request.manager_uid(),
                                             request.query_id()};
  std::shared_ptr<RunningQuery> running;
  bool original = false;
  {
    absl::MutexLock lock(running_queries_mutex_);
    auto& entry = running_queries_[key];
    if (entry == nullptr) {
      entry = std::make_shared<RunningQuery>();
      original = true;
    }
    running = entry;
  }
  if (!original) {
    LOG(WARNING) << "Worker #" << request.worker_idx() << ": request "
                 << request.query_id()
                 << " received again while running; waiting for it";
    running->done.WaitForNotification();
    return running->result;
  }
  running->result = worker_->RunRequest(request.blob());
  {
    absl::MutexLock lock(running_queries_mutex_);
    running_queries_.erase(key);
  }
  running->done.Notify();
  // The copies hold "running" until they have read the result.
  return running->result;
}

grpc::Status WorkerService::Run(grpc::ServerContext* context,
                                const proto::Query* request,
                                proto::Answer* reply) {
  {
    utils::concurrency::MutexLock l(mutex_);
    RETURN_IF_ERROR(AbslStatusToGrpcStatus(EnsureReadyWorker(
        request->manager_uid(), *request, request->worker_idx(), &l)));
    num_active_requests_++;
  }

  auto result_or = RunQueryOnce(*request);

  {
    utils::concurrency::MutexLock l(mutex_);
    num_active_requests_--;
    if (stopping_worker_) {
      LOG(INFO) << "Still " << num_active_requests_ << " active requests";
      if (num_active_requests_ == 0) {
        request_done_cv_.Signal();
      }
    }
  }

  if (!result_or.ok()) {
    reply->set_error(absl::StrCat("Worker #", request->worker_idx(), ": ",
                                  result_or.status().ToString()));
  } else {
    *reply->mutable_blob() = std::move(result_or).value();
  }
  return grpc::Status::OK;
}

grpc::Status WorkerService::WorkerRun(grpc::ServerContext* context,
                                      const proto::WorkerQuery* request,
                                      proto::WorkerAnswer* reply) {
  const auto not_ready = [&]() {
    LOG(WARNING) << "Worker received an inter worker request before being "
                    "initialized by the manager";
    reply->set_error(
        "Worker received an inter worker request before being initialized by "
        "the manager");
    return grpc::Status::OK;
  };
  // Without waiting for a set up in progress (e.g. of a restarted worker).
  if (!worker_ready_) {
    return not_ready();
  }
  {
    utils::concurrency::MutexLock l(mutex_);
    if (!worker_ready_ || request->manager_uid() != manager_uid_) {
      return not_ready();
    }
    num_active_requests_++;
  }

  auto result_or = worker_->RunRequest(request->blob());

  {
    utils::concurrency::MutexLock l(mutex_);
    num_active_requests_--;
    if (stopping_worker_ && num_active_requests_ == 0) {
      request_done_cv_.Signal();
    }
  }

  if (!result_or.ok()) {
    reply->set_error(result_or.status().ToString());
  } else {
    *reply->mutable_blob() = std::move(result_or).value();
  }
  return grpc::Status::OK;
}

grpc::Status WorkerService::UpdateWorkerAddress(
    grpc::ServerContext* context,
    const proto::UpdateWorkerAddressQuery* request, proto::Empty* reply) {
  if (!intra_worker_communication_) {
    // The correct worker address will be used when initializing the intrea
    // worker communication.
    return grpc::Status::OK;
  }
  auto& worker = *intra_worker_communication_->workers[request->worker_idx()];
  utils::concurrency::MutexLock l(worker.mutex_address);
  worker.expected_address = request->new_address();
  return grpc::Status::OK;
}

grpc::Status WorkerService::Shutdown(grpc::ServerContext* context,
                                     const proto::ShutdownQuery* request,
                                     proto::Empty* reply) {
  LOG(INFO) << "Shutdown worker";
  utils::concurrency::MutexLock l(mutex_);
  if (worker_) {
    RETURN_IF_ERROR(AbslStatusToGrpcStatus(BlockingDoneOnWorker(&l)));
    stopping_worker_ = false;
  }
  if (request->kill_worker_manager() && !stop_server_->HasBeenNotified()) {
    stop_server_->Notify();
  }
  return grpc::Status::OK;
}

grpc::Status WorkerService::Ping(grpc::ServerContext* context,
                                 const proto::Empty* request,
                                 proto::Empty* reply) {
  LOG(INFO) << "Reply to ping";
  return grpc::Status::OK;
}

absl::Status WorkerService::BlockingDoneOnWorker(
    utils::concurrency::MutexLock* lock) EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  stopping_worker_ = true;
  worker_ready_ = false;
  RETURN_IF_ERROR(worker_->Done());
  LOG(INFO) << "Waiting for the " << num_active_requests_
            << " active request(s) to complete";
  while (num_active_requests_ > 0) {
    request_done_cv_.Wait(&mutex_, lock);
  }
  FinalizeIntraWorkerCommunication();
  worker_.reset();
  return absl::OkStatus();
}

absl::Status WorkerService::EnsureReadyWorker(
    uint64_t manager_uid, const proto::Query& request, const int worker_idx,
    utils::concurrency::MutexLock* lock) EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  if (worker_) {
    if (manager_uid_ != manager_uid) {
      // The manager has changed e.g. the managed was killed and rescheduled.
      LOG(INFO) << "The manager has changed.";
      if (stopping_worker_) {
        // Another call is changing the worker.
        while (stopping_worker_) {
          stopping_worker_done_cv_.Wait(&mutex_, lock);
        }
      } else {
        RETURN_IF_ERROR(BlockingDoneOnWorker(lock));
        stopping_worker_ = false;
        stopping_worker_done_cv_.SignalAll();
      }
    } else {
      if (stopping_worker_) {
        // A new worker is being created.
        return absl::InternalError("A newer managed id was observed");
      }
      // Already initialized worker.
      return absl::OkStatus();
    }
  }

  if (!request.has_worker_config()) {
    LOG(INFO) << "Reject worker initialization as worker config is missing.";
    return absl::UnavailableError("worker config required");
  }

  LOG(INFO) << "Initialize worker.";

  manager_uid_ = manager_uid;
  inter_worker_timeout_ =
      absl::Seconds(request.worker_config().request_timeout_seconds());
  keepalive_time_ms_ = request.worker_config().keepalive_time_ms();
  keepalive_timeout_ms_ = request.worker_config().keepalive_timeout_ms();

  if (request.worker_config().manager_uid() != manager_uid_) {
    return absl::InvalidArgumentError(
        "Two different managers are fighting for the same worker");
  }

  ASSIGN_OR_RETURN(auto worker, AbstractWorkerRegisterer::Create(
                                    request.worker_config().worker_name()));
  RETURN_IF_ERROR(InternalInitializeWorker(
      worker_idx, request.worker_config().worker_addresses_size(), worker.get(),
      &hook_));
  RETURN_IF_ERROR(worker->Setup(request.worker_config().welcome_blob()));
  // The worker is only visible (e.g. to the requests of the other workers)
  // once set up.
  worker_ = std::move(worker);

  InitializerInterWorkerCommunication(request.worker_config());
  worker_ready_ = true;

  return absl::OkStatus();
}

absl::StatusOr<Blob> WorkerService::BlockingInterWorkerRequest(
    Blob blob, const int target_worker) {
  ASSIGN_OR_RETURN(auto stub, EnsureIntraWorkerStubIsReady(target_worker));

  proto::WorkerQuery query;
  *query.mutable_blob() = std::move(blob);
  query.set_manager_uid(manager_uid_);

  // Deadline of the request, re-sendings included (see
  // "WorkerConfig.request_timeout_seconds").
  std::optional<absl::Time> deadline;
  if (inter_worker_timeout_ > absl::ZeroDuration()) {
    deadline = absl::Now() + inter_worker_timeout_;
  }

  int num_re_emitting = 0;
  proto::WorkerAnswer answer;
  while (true) {
    grpc::ClientContext context;
    ConfigureClientContext(&context);
    if (deadline.has_value()) {
      context.set_deadline(absl::ToChronoTime(*deadline));
    }

    const auto status = stub->WorkerRun(&context, query, &answer);

    if (!status.ok()) {
      const auto deadline_exceeded = [&]() {
        LOG(WARNING) << "Intra worker GRPC call to worker #" << target_worker
                     << " without answer after " << inter_worker_timeout_
                     << ": " << status.error_message();
        return absl::DeadlineExceededError(
            absl::StrCat("No answer from worker #", target_worker, " after ",
                         absl::FormatDuration(inter_worker_timeout_), ": ",
                         status.error_message()));
      };
      if (deadline.has_value() &&
          (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED ||
           absl::Now() >= *deadline)) {
        return deadline_exceeded();
      }
      LOG(WARNING) << "Intra worker GRPC call failed with error: "
                   << status.error_message();
      // List of non-documented GRPC errors that can indicate a temporary
      // impossibility to reach the server.
      if (IsTransientError(status)) {
        absl::Duration sleep_duration = absl::Seconds(5);
        if (deadline.has_value()) {
          const absl::Duration remaining = *deadline - absl::Now();
          if (remaining <= absl::ZeroDuration()) {
            return deadline_exceeded();
          }
          sleep_duration = std::min(sleep_duration, remaining);
        }
        absl::SleepFor(sleep_duration);
        if (deadline.has_value() && absl::Now() >= *deadline) {
          return deadline_exceeded();
        }
        // The worker died during the execution (e.g. rescheduling).
        // Let's try again, on a new connection.
        num_re_emitting++;
        LOG(WARNING) << "Re-emitting request (num_re_emitting:"
                     << num_re_emitting << ")";

        ASSIGN_OR_RETURN(stub, EnsureIntraWorkerStubIsReady(
                                   target_worker, /*failed_stub=*/stub));

        continue;
      } else {
        // Something is not right.
        return absl::UnknownError(status.error_message());
      }
    } else {
      if (answer.has_error()) {
        LOG(WARNING)
            << "Worker called with intra worker GRPC call returned an error: "
            << answer.error();
        return absl::UnknownError(answer.error());
      } else {
        return std::move(*answer.mutable_blob());
      }
    }
  }
}

void WorkerService::ProcessInterWorkerCommunication() {
  DCHECK(intra_worker_communication_);
  while (true) {
    auto pending_blob_or = intra_worker_communication_->pending_queries.Pop();
    if (!pending_blob_or.has_value()) {
      break;
    }

    const auto target_worker = pending_blob_or.value().first;
    auto answer = BlockingInterWorkerRequest(
        std::move(pending_blob_or).value().second, target_worker);

    intra_worker_communication_->pending_answers.Push(std::move(answer));
  }
}

void WorkerService::InitializerInterWorkerCommunication(
    const proto::WorkerConfig& worker_config) {
  DCHECK(!intra_worker_communication_);
  intra_worker_communication_ = std::make_unique<InterWorkerCommunication>();
  // One thread per request to the other workers in flight.
  const int num_threads = worker_config.worker_to_worker_parallelism() > 0
                              ? worker_config.worker_to_worker_parallelism()
                              : worker_config.parallel_execution_per_worker();
  intra_worker_communication_->threads.Start(
      num_threads, [&]() { ProcessInterWorkerCommunication(); });

  intra_worker_communication_->workers.reserve(
      worker_config.worker_addresses_size());
  for (int worker_idx = 0; worker_idx < worker_config.worker_addresses_size();
       worker_idx++) {
    auto worker = std::make_unique<InterWorkerCommunication::Worker>();
    utils::concurrency::MutexLock l(worker->mutex_address);
    worker->expected_address = worker_config.worker_addresses(worker_idx);
    intra_worker_communication_->workers.push_back(std::move(worker));
  }
}

absl::StatusOr<proto::Server::Stub*>
WorkerService::EnsureIntraWorkerStubIsReady(
    const int worker_idx, const proto::Server::Stub* failed_stub) {
  CHECK(intra_worker_communication_);
  CHECK_LT(worker_idx, intra_worker_communication_->workers.size());
  auto& worker = *intra_worker_communication_->workers[worker_idx];

  utils::concurrency::MutexLock l(worker.mutex_address);

  // A failed stub is replaced, unless another thread already did it (the
  // discarded stubs are never released, so their addresses are not reused).
  const bool replace_failed_stub =
      failed_stub != nullptr && worker.stub.get() == failed_stub;
  if (worker.stub && worker.expected_address == worker.connected_address &&
      !replace_failed_stub) {
    return worker.stub.get();
  }

  if (worker.stub) {
    if (replace_failed_stub) {
      LOG(WARNING) << "Open a new connection to worker #" << worker_idx
                   << " at " << worker.expected_address;
    } else {
      LOG(WARNING) << "Update address of worker #" << worker_idx << " from "
                   << worker.connected_address << " to "
                   << worker.expected_address;
    }
    worker.discarded_stubs_.push_back(std::move(worker.stub));
    worker.stub.reset();
  } else {
    LOG(WARNING) << "Create stub to worker #" << worker_idx;
  }

  std::shared_ptr<grpc::ChannelCredentials> credential;
  if (use_loas_) {
    return absl::InvalidArgumentError("LOAS is not available in OSS build");
  } else {
    credential = grpc::InsecureChannelCredentials();
  }

  worker.connected_address = worker.expected_address;
  auto channel = grpc::CreateCustomChannel(
      worker.connected_address, credential,
      WorkerChannelArguments(keepalive_time_ms_, keepalive_timeout_ms_));
  worker.stub = proto::Server::NewStub(channel);
  return worker.stub.get();
}

void WorkerService::FinalizeIntraWorkerCommunication() {
  if (intra_worker_communication_) {
    intra_worker_communication_->pending_answers.Close();
    intra_worker_communication_->pending_queries.Close();
    intra_worker_communication_->threads.JoinAndClear();
  }
  intra_worker_communication_.reset();
}

};  // namespace internal

absl::StatusOr<std::unique_ptr<GRPCWorkerServer>> StartGRPCWorker(
    int port, bool use_loas) {
  auto server = std::make_unique<GRPCWorkerServer>();

  server->service =
      std::make_unique<internal::WorkerService>(&server->stop_server, use_loas);

  std::shared_ptr<grpc::ServerCredentials> credential;
  if (use_loas) {
    return absl::InvalidArgumentError("Loas not available");
  } else {
    credential = grpc::InsecureServerCredentials();
  }

  grpc::ServerBuilder builder;
  std::string server_address = absl::StrCat("[::]:", port);
  LOG(INFO) << "Start worker server at address " << server_address
            << " (gRPC handler threads: " << kMinPollers << " to "
            << kMaxPollers << " kept polling)";
  builder.AddListeningPort(server_address, credential, &server->port);
  AcceptKeepalivePings(&builder);
  // Each request in flight runs on its own thread of the synchronous server.
  // With the default options (1 to 2 threads polling for new requests), most
  // threads are created for a request and destroyed after it: with many
  // workers, the bursts of worker-to-worker requests (e.g. the aggregations)
  // create and destroy hundreds of threads per second. Up to "kMaxPollers"
  // threads go back to polling for the next requests instead. This is not a
  // limit of the number of threads (a request can wait for the requests of
  // other workers: a limit could deadlock).
  builder.SetSyncServerOption(
      grpc::ServerBuilder::SyncServerOption::MIN_POLLERS, kMinPollers);
  builder.SetSyncServerOption(
      grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, kMaxPollers);

  builder.RegisterService(server->service.get());

  server->grpc_server = builder.BuildAndStart();
  if (!server->grpc_server) {
    return absl::UnknownError("Could not start the worker GRPC server");
  }

  return std::move(server);
}

void WaitForGRPCWorkerToShutdown(GRPCWorkerServer* server) {
  server->server_thread = std::make_unique<utils::concurrency::Thread>(
      [&]() { server->grpc_server->Wait(); });
  server->stop_server.WaitForNotification();
  absl::SleepFor(absl::Seconds(1));
  server->service->ShutDown();
  server->grpc_server->Shutdown();
  server->server_thread->Join();
}

absl::Status WorkerMain(const int port, bool use_loas) {
  ASSIGN_OR_RETURN(auto server, StartGRPCWorker(port, use_loas));
  WaitForGRPCWorkerToShutdown(server.get());
  return absl::OkStatus();
}

}  // namespace grpc_worker
}  // namespace distribute
}  // namespace yggdrasil_decision_forests
