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

#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/channel_arguments.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
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

grpc::Status WorkerService::Run(grpc::ServerContext* context,
                                const proto::Query* request,
                                proto::Answer* reply) {
  {
    utils::concurrency::MutexLock l(&mutex_);
    RETURN_IF_ERROR(AbslStatusToGrpcStatus(EnsureReadyWorker(
        request->manager_uid(), *request, request->worker_idx(), &l)));
    num_active_requests_++;
  }

  auto result_or = worker_->RunRequest(request->blob());

  {
    utils::concurrency::MutexLock l(&mutex_);
    num_active_requests_--;
    if (stopping_worker_) {
      YDF_LOG(INFO) << "Still " << num_active_requests_ << " active requests";
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
  if (!worker_) {
    YDF_LOG(WARNING) << "Worker received an inter worker request before being "
                        "initialized by the manager";
    reply->set_error(
        "Worker received an inter worker request before being initialized by "
        "the manager");
    return grpc::Status::OK;
  }

  auto result_or = worker_->RunRequest(request->blob());
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
  utils::concurrency::MutexLock l(&worker.mutex_address);
  worker.expected_address = request->new_address();
  return grpc::Status::OK;
}

grpc::Status WorkerService::Shutdown(grpc::ServerContext* context,
                                     const proto::ShutdownQuery* request,
                                     proto::Empty* reply) {
  YDF_LOG(INFO) << "Shutdown worker";
  utils::concurrency::MutexLock l(&mutex_);
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
  YDF_LOG(INFO) << "Reply to ping";
  return grpc::Status::OK;
}

absl::Status WorkerService::BlockingDoneOnWorker(
    utils::concurrency::MutexLock* lock) EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
  stopping_worker_ = true;
  RETURN_IF_ERROR(worker_->Done());
  YDF_LOG(INFO) << "Waiting for the " << num_active_requests_
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
      YDF_LOG(INFO) << "The manager has changed.";
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
    YDF_LOG(INFO)
        << "Reject worker initialization as worker config is missing.";
    return absl::UnavailableError("worker config required");
  }

  YDF_LOG(INFO) << "Initialize worker.";

  manager_uid_ = manager_uid;

  if (request.worker_config().manager_uid() != manager_uid_) {
    return absl::InvalidArgumentError(
        "Two different managers are fighting for the same worker");
  }

  ASSIGN_OR_RETURN(worker_, AbstractWorkerRegisterer::Create(
                                request.worker_config().worker_name()));
  RETURN_IF_ERROR(InternalInitializeWorker(
      worker_idx, request.worker_config().worker_addresses_size(),
      worker_.get(), &hook_));
  RETURN_IF_ERROR(worker_->Setup(request.worker_config().welcome_blob()));

  InitializerInterWorkerCommunication(request.worker_config());

  return absl::OkStatus();
}

absl::StatusOr<Blob> WorkerService::BlockingInterWorkerRequest(
    Blob blob, const int target_worker) {
  ASSIGN_OR_RETURN(auto stub, EnsureIntraWorkerStubIsReady(target_worker));

  proto::WorkerQuery query;
  *query.mutable_blob() = std::move(blob);
  query.set_manager_uid(manager_uid_);

  int num_re_emitting = 0;
  proto::WorkerAnswer answer;
  while (true) {
    grpc::ClientContext context;
    ConfigureClientContext(&context);

    const auto status = stub->WorkerRun(&context, query, &answer);

    if (!status.ok()) {
      YDF_LOG(WARNING) << "Intra worker GRPC call failed with error: "
                       << status.error_message();
      // List of non-documented GRPC errors that can indicate a temporary
      // impossibility to reach the server.
      if (IsTransientError(status)) {
        absl::SleepFor(absl::Seconds(5));
        // The worker died during the execution (e.g. rescheduling).
        // Let's try again.
        num_re_emitting++;
        YDF_LOG(WARNING) << "Re-emitting request (num_re_emitting:"
                         << num_re_emitting << ")";

        ASSIGN_OR_RETURN(stub, EnsureIntraWorkerStubIsReady(target_worker));

        continue;
      } else {
        // Something is not right.
        return absl::UnknownError(status.error_message());
      }
    } else {
      if (answer.has_error()) {
        YDF_LOG(WARNING)
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
  intra_worker_communication_ = absl::make_unique<InterWorkerCommunication>();
  intra_worker_communication_->threads.Start(
      worker_config.parallel_execution_per_worker(),
      [&]() { ProcessInterWorkerCommunication(); });

  intra_worker_communication_->workers.reserve(
      worker_config.worker_addresses_size());
  for (int worker_idx = 0; worker_idx < worker_config.worker_addresses_size();
       worker_idx++) {
    auto worker = absl::make_unique<InterWorkerCommunication::Worker>();
    utils::concurrency::MutexLock l(&worker->mutex_address);
    worker->expected_address = worker_config.worker_addresses(worker_idx);
    intra_worker_communication_->workers.push_back(std::move(worker));
  }
}

absl::StatusOr<proto::Server::Stub*>
WorkerService::EnsureIntraWorkerStubIsReady(const int worker_idx) {
  CHECK(intra_worker_communication_);
  CHECK_LT(worker_idx, intra_worker_communication_->workers.size());
  auto& worker = *intra_worker_communication_->workers[worker_idx];

  utils::concurrency::MutexLock l(&worker.mutex_address);

  if (worker.stub && worker.expected_address == worker.connected_address) {
    return worker.stub.get();
  }

  if (worker.stub) {
    YDF_LOG(WARNING) << "Update address of worker #" << worker_idx << " from "
                     << worker.connected_address << " to "
                     << worker.expected_address;
    worker.discarded_stubs_.push_back(std::move(worker.stub));
    worker.stub.reset();
  } else {
    YDF_LOG(WARNING) << "Create stub to worker #" << worker_idx;
  }

  std::shared_ptr<grpc::ChannelCredentials> credential;
  if (use_loas_) {
    return absl::InvalidArgumentError("LOAS is not available in OSS build");
  } else {
    credential = grpc::InsecureChannelCredentials();
  }

  grpc::ChannelArguments channel_arguments;
  channel_arguments.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
  channel_arguments.SetMaxSendMessageSize(std::numeric_limits<int>::max());
  worker.connected_address = worker.expected_address;
  auto channel = grpc::CreateCustomChannel(worker.connected_address, credential,
                                           channel_arguments);
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
  auto server = absl::make_unique<GRPCWorkerServer>();

  server->service = absl::make_unique<internal::WorkerService>(
      &server->stop_server, use_loas);

  std::shared_ptr<grpc::ServerCredentials> credential;
  if (use_loas) {
    return absl::InvalidArgumentError("Loas not available");
  } else {
    credential = grpc::InsecureServerCredentials();
  }

  grpc::ServerBuilder builder;
  std::string server_address = absl::StrCat("[::]:", port);
  YDF_LOG(INFO) << "Start worker server at address " << server_address;
  builder.AddListeningPort(server_address, credential, &server->port);

  builder.RegisterService(server->service.get());

  server->grpc_server = builder.BuildAndStart();
  if (!server->grpc_server) {
    return absl::UnknownError("Could not start the worker GRPC server");
  }

  return std::move(server);
}

void WaitForGRPCWorkerToShutdown(GRPCWorkerServer* server) {
  server->server_thread = absl::make_unique<utils::concurrency::Thread>(
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
