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

#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/channel_arguments.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc.grpc.pb.h"
#include "yggdrasil_decision_forests/utils/distribute/utils.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests {
namespace distribute {
namespace {

// Maximum execution time of a request.
constexpr int kDeadLineInHours = 24 * 40;

grpc::Status AbslStatusToGrpcStatus(const absl::Status& src) {
  if (src.ok()) {
    return grpc::Status();
  } else {
    return grpc::Status(grpc::StatusCode::UNKNOWN, src.ToString());
  }
}

class WorkerService final : public proto::Server::Service {
 public:
  WorkerService(utils::concurrency::Notification* stop_server, bool use_loas)
      : stop_server_(stop_server), use_loas_(use_loas), hook_(this) {}

  void ShutDown() { FinalizeIntraWorkerCommunication(); }

 protected:
  // Implementation of worker->worker request.
  absl::Status AsynchronousRequestToOtherWorker(
      Blob blob, int target_worker_idx, AbstractWorker* emitter_worker) {
    intra_worker_communication_->pending_queries.Push(
        std::make_pair(target_worker_idx, std::move(blob)));
    return absl::OkStatus();
  }

  // Implementation of the worker->worker async reply.
  utils::StatusOr<Blob> NextAsynchronousAnswerFromOtherWorker(
      AbstractWorker* emitter_worker) {
    auto answer = intra_worker_communication_->pending_answers.Pop();
    if (!answer.has_value()) {
      return absl::OutOfRangeError("No more results available");
    }
    return std::move(answer.value());
  }

 private:
  class WorkerHook : public AbstractWorkerHook {
   public:
    WorkerHook(WorkerService* parent) : parent_(parent) {}

    absl::Status AsynchronousRequestToOtherWorker(
        Blob blob, int target_worker_idx,
        AbstractWorker* emitter_worker) override {
      return parent_->AsynchronousRequestToOtherWorker(
          std::move(blob), target_worker_idx, emitter_worker);
    }

    utils::StatusOr<Blob> NextAsynchronousAnswerFromOtherWorker(
        AbstractWorker* emitter_worker) override {
      return parent_->NextAsynchronousAnswerFromOtherWorker(emitter_worker);
    }

   private:
    WorkerService* parent_;
  };

  // Execution of a query emitted by the manager.
  grpc::Status Run(grpc::ServerContext* context, const proto::Query* request,
                   proto::Answer* reply) override {
    {
      utils::concurrency::MutexLock l(&mutex_);
      RETURN_IF_ERROR(AbslStatusToGrpcStatus(
          EnsureReadyWorker(request->manager_uid(), request->config_path(),
                            request->worker_idx(), &l)));
      num_active_requests_++;
    }

    auto result_or = worker_->RunRequest(request->blob());

    {
      utils::concurrency::MutexLock l(&mutex_);
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

  // Execution of a query emitted by another worker.
  grpc::Status WorkerRun(grpc::ServerContext* context,
                         const proto::WorkerQuery* request,
                         proto::WorkerAnswer* reply) override {
    if (!worker_) {
      LOG(WARNING) << "Worker received an inter worker request before being "
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

  grpc::Status Shutdown(grpc::ServerContext* context,
                        const proto::ShutdownQuery* request,
                        proto::Empty* reply) override {
    LOG(INFO) << "Shutdown worker";
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

  grpc::Status Ping(grpc::ServerContext* context, const proto::Empty* request,
                    proto::Empty* reply) override {
    LOG(INFO) << "Reply to ping";
    return grpc::Status::OK;
  }

  // Calls "Done" on the worker, wait for all the pending operation to be done
  // or cancel, and destroy the "worker_" object.
  absl::Status BlockingDoneOnWorker(utils::concurrency::MutexLock* lock)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    stopping_worker_ = true;
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

  // After a call to this method, the worker is ready to processed requests.
  // This method should be called before any request.
  absl::Status EnsureReadyWorker(uint64_t manager_uid,
                                 absl::string_view config_path,
                                 const int worker_idx,
                                 utils::concurrency::MutexLock* lock)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
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
    LOG(INFO) << "Initialize worker.";

    manager_uid_ = manager_uid;

    proto::WorkerConfig worker_config;
    RETURN_IF_ERROR(
        file::GetBinaryProto(config_path, &worker_config, file::Defaults()));
    if (worker_config.manager_uid() != manager_uid_) {
      return absl::InvalidArgumentError(
          "Two different managers are fighting for the same worker");
    }

    worker_addresses_ = {worker_config.worker_addresses().begin(),
                         worker_config.worker_addresses().end()};

    ASSIGN_OR_RETURN(
        worker_, AbstractWorkerRegisterer::Create(worker_config.worker_name()));
    RETURN_IF_ERROR(InternalInitializeWorker(
        worker_idx, worker_config.worker_addresses_size(), worker_.get(),
        &hook_));
    RETURN_IF_ERROR(worker_->Setup(worker_config.welcome_blob()));

    InitializerInterWorkerCommunication(
        worker_config.worker_addresses_size(),
        worker_config.parallel_execution_per_worker());

    return absl::OkStatus();
  }

  // Blocking inter worker request.
  utils::StatusOr<Blob> BlockingInterWorkerRequest(Blob blob,
                                                   const int target_worker) {
    RETURN_IF_ERROR(EnsureIntraWorkerStubIsReady(target_worker));

    proto::WorkerQuery query;
    *query.mutable_blob() = std::move(blob);
    query.set_manager_uid(manager_uid_);

    int num_re_emitting = 0;
    proto::WorkerAnswer answer;
    while (true) {
      grpc::ClientContext context;
      context.set_wait_for_ready(true);
      context.set_deadline(std::chrono::system_clock::now() +
                           std::chrono::hours(kDeadLineInHours));
      const auto status =
          intra_worker_communication_->other_workers[target_worker]
              .stub->WorkerRun(&context, query, &answer);

      if (!status.ok()) {
        LOG(WARNING) << "Intra worker GRPC call failed with error: "
                     << status.error_message();
        // List of non-documented GRPC errors that can indicate a temporary
        // impossibility to reach the server.
        if (status.error_message() == "Socket closed" ||
            status.error_message() == "Transport closed" ||
            status.error_message() == "Connection reset by peer" ||
            status.error_message() == "Broken pipe" ||
            status.error_message() == "keepalive watchdog timeout") {
          // The worker died during the execution (e.g. rescheduling).
          // Let's try again.
          num_re_emitting++;
          LOG(WARNING) << "Re-emitting request (num_re_emitting:"
                       << num_re_emitting << ")";
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

  // Loop for a thread processing inter worker requests.
  void ProcessInterWorkerCommunication() {
    CHECK(intra_worker_communication_);
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

  // Initialize the connection and thread for the inter worker communication.
  // This method should be called before any inter worker communication.
  void InitializerInterWorkerCommunication(const int num_workers,
                                           const int num_threads) {
    CHECK(!intra_worker_communication_);
    intra_worker_communication_ = absl::make_unique<InterWorkerCommunication>();
    intra_worker_communication_->threads.Start(
        num_threads, [&]() { ProcessInterWorkerCommunication(); });
    intra_worker_communication_->other_workers.resize(num_workers);
  }

  // Ensures that the communication with another worker is ready.
  absl::Status EnsureIntraWorkerStubIsReady(const int worker_idx) {
    CHECK(intra_worker_communication_);
    CHECK_LT(worker_idx, intra_worker_communication_->other_workers.size());
    auto& worker = intra_worker_communication_->other_workers[worker_idx];

    if (worker.stub) {
      return absl::OkStatus();
    }

    std::shared_ptr<grpc::ChannelCredentials> credential;
    if (use_loas_) {
      return absl::InvalidArgumentError("Loas not available");
    } else {
      credential = grpc::InsecureChannelCredentials();
    }

    grpc::ChannelArguments channel_arguments;
    channel_arguments.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
    channel_arguments.SetMaxSendMessageSize(std::numeric_limits<int>::max());

    worker.channel = grpc::CreateCustomChannel(worker_addresses_[worker_idx],
                                               credential, channel_arguments);
    worker.stub = proto::Server::NewStub(worker.channel);
    return absl::OkStatus();
  }

  // Finalize the current worker communication.
  // No more inter worker communication should be done after this call, except
  // for "InitializerInterWorkerCommunication" to re-initialize it.
  void FinalizeIntraWorkerCommunication() {
    if (intra_worker_communication_) {
      intra_worker_communication_->pending_answers.Close();
      intra_worker_communication_->pending_queries.Close();
      intra_worker_communication_->threads.JoinAndClear();
    }
    intra_worker_communication_.reset();
  }

  // Non owning pointer to the notification that stops the server.
  utils::concurrency::Notification* stop_server_ = nullptr;

  // Active worker implementation.
  std::unique_ptr<AbstractWorker> worker_;

  // UID of the manager. Only valid if worker_ is set.
  uint64_t manager_uid_;

  // Socket address of all the workers.
  std::vector<std::string> worker_addresses_;

  // Fields related to the inter worker communication.
  struct InterWorkerCommunication {
    // List of target worker index and data emitted by this worker.
    utils::concurrency::Channel<std::pair<int, Blob>> pending_queries;

    // Answers to this worker queries.
    utils::concurrency::Channel<utils::StatusOr<Blob>> pending_answers;

    // Thread emitting and receiving intra-workers requests/answers.
    ThreadVector threads;

    struct OtherWorkers {
      std::shared_ptr<grpc::Channel> channel;
      std::unique_ptr<proto::Server::Stub> stub;
    };

    // Communication channel to other workers for intra worker communication.
    std::vector<OtherWorkers> other_workers;
  };

  std::unique_ptr<InterWorkerCommunication> intra_worker_communication_;

  // utils::concurrency::Mutex protecting the initialization of the worker.
  utils::concurrency::Mutex mutex_ GUARDED_BY(mutex_);

  // True when the worker is being stopped (i.e. waiting for all the requests to
  // be completed) because the user called "Done" on the manager, or because the
  // manager have changed.
  bool stopping_worker_ GUARDED_BY(mutex_) = false;

  // Signal when the worker are done beeing stopped i.e. stopping_worker_ goes
  // from true to false.
  utils::concurrency::CondVar stopping_worker_done_cv_;

  // Signal are the end of a request execution when not other request are
  // running (i.e. num_active_requests_=0).
  utils::concurrency::CondVar request_done_cv_;

  // Number of requests currently beeing processed.
  int num_active_requests_ GUARDED_BY(mutex_) = 0;

  // Does the worker uses LOAS.
  bool use_loas_;

  // Callback to inter-worker communication.
  WorkerHook hook_;
};

}  // namespace

absl::Status GRPCWorkerMainWorkerMain(const int port, bool use_loas) {
  utils::concurrency::Notification stop_server;
  WorkerService service(&stop_server, use_loas);

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
  service.ShutDown();
  server->Shutdown();
  server_thread.Join();
  return absl::OkStatus();
}

}  // namespace distribute
}  // namespace yggdrasil_decision_forests
