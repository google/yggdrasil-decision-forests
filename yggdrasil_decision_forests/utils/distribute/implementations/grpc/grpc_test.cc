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

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/distribute/core.h"
#include "yggdrasil_decision_forests/utils/distribute/distribute.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc.pb.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_manager.h"
#include "yggdrasil_decision_forests/utils/distribute/implementations/grpc/grpc_worker.h"
#include "yggdrasil_decision_forests/utils/distribute/test_utils.h"
#include "yggdrasil_decision_forests/utils/distribute/toy_worker.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace distribute {
namespace grpc_worker {
namespace {

// Create a GRPC manager and its workers.
ManagerCreatorAndWorkers CreateGrpcManagerCreator(
    int parallel_execution_per_worker = 1, int num_workers = 5,
    int request_timeout_seconds = 0, int worker_to_worker_parallelism = 0) {
  ManagerCreatorAndWorkers manager_and_workers;
  // Manager configuration.
  proto::Config config;
  config.set_implementation_key("GRPC");
  config.set_verbosity(1);
  config.set_working_directory(
      file::JoinPath(test::TmpDirectory(), "work_dir"));
  config.MutableExtension(proto::grpc)
      ->set_request_timeout_seconds(request_timeout_seconds);
  config.MutableExtension(proto::grpc)
      ->set_worker_to_worker_parallelism(worker_to_worker_parallelism);
  auto* addresses =
      config.MutableExtension(proto::grpc)->mutable_socket_addresses();
  for (int worker_idx = 0; worker_idx < num_workers; worker_idx++) {
    // Create address.
    auto* address = addresses->add_addresses();
    address->set_ip("localhost");
    std::string error;
    const int port = test::PickUnusedPortOrDie();
    CHECK_GT(port, 0);
    address->set_port(port);

    // Create worker thread.
    manager_and_workers.worker_threads.push_back(
        std::make_unique<utils::concurrency::Thread>(
            [port]() { CHECK_OK(WorkerMain(port)); }));
    absl::SleepFor(absl::Seconds(0.2));
  }

  // Start manager.
  manager_and_workers.manager_creator = [config,
                                         parallel_execution_per_worker]()
      -> std::unique_ptr<AbstractManager> {
    auto manager_or = CreateManager(config, /*worker_name=*/kToyWorkerKey,
                                    /*welcome_blob=*/"hello",
                                    parallel_execution_per_worker);
    if (!manager_or.ok()) {
      LOG(ERROR) << "Failed to create manager: " << manager_or.status();
      return nullptr;
    }
    return std::move(manager_or.value());
  };
  return manager_and_workers;
}

ManagerAndWorkers CreateGrpcManager(int parallel_execution_per_worker = 1) {
  auto creator = CreateGrpcManagerCreator(parallel_execution_per_worker);
  ManagerAndWorkers m_and_w;
  m_and_w.worker_threads = std::move(creator.worker_threads);
  m_and_w.manager = creator.manager_creator();
  return m_and_w;
}

TEST(GRPC, WorkerError) {
  auto all = CreateGrpcManager();
  TestWorkerError(all.manager.get());
  all.Join();
}

TEST(GRPC, AsyncError) {
  auto all = CreateGrpcManager();
  TestAsyncError(all.manager.get());
  all.Join();
}

TEST(GRPC, AsyncErrorInPool) {
  auto all = CreateGrpcManager();
  TestAsyncErrorInPool(all.manager.get());
  all.Join();
}

TEST(GRPC, BlockingRequest) {
  auto all = CreateGrpcManager();
  TestBlockingRequest(all.manager.get());
  all.Join();
}

TEST(GRPC, LongBlockingRequest) {
  auto all = CreateGrpcManager();
  TestLongBlockingRequest(all.manager.get());
  all.Join();
}

// A request slower than the timeout is sent again, and runs only once: the
// copy waits for the running request.
TEST(GRPC, RequestTimeoutRunsOnce) {
  auto creator = CreateGrpcManagerCreator(/*parallel_execution_per_worker=*/1,
                                          /*num_workers=*/2,
                                          /*request_timeout_seconds=*/2);
  ManagerAndWorkers all;
  all.worker_threads = std::move(creator.worker_threads);
  all.manager = creator.manager_creator();
  ASSERT_OK_AND_ASSIGN(const std::string before,
                       all.manager->BlockingRequest("num_sleeps", 0));
  const absl::Time begin = absl::Now();
  ASSERT_OK_AND_ASSIGN(const std::string sleep_result,
                       all.manager->BlockingRequest("sleep", 0));
  EXPECT_EQ(sleep_result, "");
  EXPECT_LT(absl::Now() - begin, absl::Seconds(9));
  int num_before;
  ASSERT_TRUE(absl::SimpleAtoi(before, &num_before));
  ASSERT_OK_AND_ASSIGN(const std::string after,
                       all.manager->BlockingRequest("num_sleeps", 0));
  EXPECT_EQ(after, absl::StrCat(num_before + 1));
  EXPECT_OK(all.manager->Done(true));
  all.Join();
}

// A request to another worker slower than its deadline (half of the
// manager's timeout) fails with DEADLINE_EXCEEDED instead of waiting, before
// the manager re-sends its request: a stuck worker cannot block the others,
// and the request runs once.
TEST(GRPC, InterWorkerRequestTimeout) {
  auto creator = CreateGrpcManagerCreator(/*parallel_execution_per_worker=*/2,
                                          /*num_workers=*/2,
                                          /*request_timeout_seconds=*/4);
  ManagerAndWorkers all;
  all.worker_threads = std::move(creator.worker_threads);
  all.manager = creator.manager_creator();
  // The workers are initialized by their first request.
  for (int w = 0; w < 2; w++) {
    ASSERT_OK_AND_ASSIGN(const std::string result,
                         all.manager->BlockingRequest("worker_idx", w));
    EXPECT_EQ(result, absl::StrCat(w));
  }
  const absl::Time begin = absl::Now();
  ASSERT_OK_AND_ASSIGN(
      const std::string result,
      all.manager->BlockingRequest("other_worker_sleep", 0));
  EXPECT_EQ(result, "DEADLINE_EXCEEDED");
  EXPECT_LT(absl::Now() - begin, absl::Seconds(5));
  EXPECT_OK(all.manager->Done(true));
  all.Join();
}

// An inter-worker request that fails with a transient error (e.g. target worker
// unreachable) caps its retry sleep to the remaining deadline rather than
// sleeping past it.
TEST(GRPC, InterWorkerTransientErrorRespectsDeadline) {
  auto creator = CreateGrpcManagerCreator(/*parallel_execution_per_worker=*/2,
                                          /*num_workers=*/2,
                                          /*request_timeout_seconds=*/4);
  ManagerAndWorkers all;
  all.worker_threads = std::move(creator.worker_threads);
  all.manager = creator.manager_creator();
  for (int w = 0; w < 2; w++) {
    ASSERT_OK_AND_ASSIGN(const std::string result,
                         all.manager->BlockingRequest("worker_idx", w));
    EXPECT_EQ(result, absl::StrCat(w));
  }
  auto* grpc_manager = dynamic_cast<GRPCManager*>(all.manager.get());
  ASSERT_NE(grpc_manager, nullptr);
  ASSERT_OK(grpc_manager->DebugShutdownWorker(1));
  all.worker_threads[1]->Join();
  all.worker_threads.pop_back();

  const absl::Time begin = absl::Now();
  ASSERT_OK_AND_ASSIGN(
      const std::string result,
      all.manager->BlockingRequest("next_worker_idx_or_error", 0));
  EXPECT_THAT(result, testing::HasSubstr("DEADLINE_EXCEEDED"));
  // inter_worker_timeout_ is 2s; without capping the 5s retry sleep, this would
  // block for > 5s (and exceed request_timeout_seconds = 4s).
  EXPECT_LT(absl::Now() - begin, absl::Seconds(4));
  EXPECT_OK(all.manager->Done(true));
  all.Join();
}

// A request of another worker to a worker being set up (e.g. restarted, and
// loading its dataset) fails at once, without running on the worker before the
// end of its set up.
TEST(GRPC, InterWorkerRequestDuringSetup) {
  auto creator = CreateGrpcManagerCreator(/*parallel_execution_per_worker=*/2,
                                          /*num_workers=*/2);
  ManagerAndWorkers all;
  all.worker_threads = std::move(creator.worker_threads);
  all.manager = creator.manager_creator();
  ASSERT_OK_AND_ASSIGN(const std::string result,
                       all.manager->BlockingRequest("worker_idx", 0));
  EXPECT_EQ(result, "0");

  // The set up of worker #1 starts with its first request.
  ToyWorker::SetSetupDelay(absl::Seconds(4));
  utils::concurrency::Thread setup([&]() {
    ASSERT_OK_AND_ASSIGN(const std::string result,
                         all.manager->BlockingRequest("worker_idx", 1));
    EXPECT_EQ(result, "1");
  });
  absl::SleepFor(absl::Seconds(1));
  const absl::Time begin = absl::Now();
  ASSERT_OK_AND_ASSIGN(
      const std::string next_worker_idx_or_error,
      all.manager->BlockingRequest("next_worker_idx_or_error", 0));
  EXPECT_THAT(next_worker_idx_or_error,
              testing::HasSubstr("before being initialized by the manager"));
  EXPECT_LT(absl::Now() - begin, absl::Seconds(2));
  setup.Join();
  ToyWorker::SetSetupDelay(absl::ZeroDuration());

  // Once set up.
  ASSERT_OK_AND_ASSIGN(
      const std::string valid_next_worker_idx_or_error,
      all.manager->BlockingRequest("next_worker_idx_or_error", 0));
  EXPECT_EQ(valid_next_worker_idx_or_error, "1");
  EXPECT_OK(all.manager->Done(true));
  all.Join();
}

// The requests of a worker to the other workers run at the same time, up to
// "worker_to_worker_parallelism" (default: "parallel_execution_per_worker").
TEST(GRPC, WorkerToWorkerParallelism) {
  for (const int parallelism : {0, 16}) {
    SCOPED_TRACE(parallelism);
    auto creator = CreateGrpcManagerCreator(
        /*parallel_execution_per_worker=*/2, /*num_workers=*/3,
        /*request_timeout_seconds=*/0, parallelism);
    ManagerAndWorkers all;
    all.worker_threads = std::move(creator.worker_threads);
    all.manager = creator.manager_creator();
    // The workers are initialized by their first request.
    for (int w = 0; w < 3; w++) {
      ASSERT_OK_AND_ASSIGN(const std::string result,
                           all.manager->BlockingRequest("worker_idx", w));
      EXPECT_EQ(result, absl::StrCat(w));
    }
    const absl::Time begin = absl::Now();
    ASSERT_OK_AND_ASSIGN(
        const std::string result,
        all.manager->BlockingRequest("concurrent_requests:16", 0));
    EXPECT_EQ(result, parallelism == 0 ? "2" : "16");
    if (parallelism > 0) {
      // The 16 requests of 1 second at once, instead of 8 waves of 2.
      EXPECT_LT(absl::Now() - begin, absl::Seconds(6));
    }
    EXPECT_OK(all.manager->Done(true));
    all.Join();
  }
}

TEST(GRPC, AsynchronousRequest) {
  auto all = CreateGrpcManager();
  TestAsynchronousRequest(all.manager.get());
  all.Join();
}

TEST(GRPC, BlockingRequestWithSpecificWorker) {
  auto all = CreateGrpcManager();
  TestBlockingRequestWithSpecificWorker(all.manager.get());
  all.Join();
}

TEST(GRPC, AsynchronousRequestWithSpecificWorker) {
  auto all = CreateGrpcManager();
  TestAsynchronousRequestWithSpecificWorker(all.manager.get());
  all.Join();
}

TEST(GRPC, AsynchronousIntraWorkerCommunication) {
  auto all = CreateGrpcManager();
  TestAsynchronousIntraWorkerCommunication(all.manager.get());
  all.Join();
}

TEST(GRPC, BlockingIntraWorkerCommunication) {
  auto all = CreateGrpcManager();
  TestBlockingIntraWorkerCommunication(all.manager.get());
  all.Join();
}

TEST(GRPC, AsynchronousParallelWorkerExecution) {
  auto all = CreateGrpcManager(5);
  TestAsynchronousParallelWorkerExecution(all.manager.get());
  all.Join();
}

TEST(GRPC, TestChangeManagerNice) {
  auto all = CreateGrpcManagerCreator(5, 1);
  TestChangeManager(&all, /*nice=*/true);
  all.Join();
}

  TEST(GRPC, DISABLED_TestChangeManagerNotNice) {
  auto all = CreateGrpcManagerCreator(5, 1);
  TestChangeManager(&all, /*nice=*/false);
  all.Join();
}

TEST(GRPC, TestMessup) {
  auto all = CreateGrpcManager();
  TestMessup(all.manager.get(), [&]() {
    // This method creates a new worker #0, and tell the manager to replace the
    // old worker #0 by the new worker #0. After this change, any call to the
    // old worker #0 will trigger a LOG(FATAL).

    // Mark worker #0 as forbidden. New requests to this worker will trigger a
    // LOG(FATAL).
    CHECK_OK(all.manager->BlockingRequest("forbidden", 0).status());

    auto* grpc_manager = dynamic_cast<GRPCManager*>(all.manager.get());
    CHECK(grpc_manager != nullptr);
    CHECK_OK(grpc_manager->DebugShutdownWorker(0));

    // New port for worker #0.
    const int port = test::PickUnusedPortOrDie();
    CHECK_GT(port, 0);

    // Isolate the forbidden worker thread.
    all.discarded_worker_threads.push_back(
        std::move(all.worker_threads.front()));

    // Create worker thread.
    all.worker_threads.front() = std::make_unique<utils::concurrency::Thread>(
        [port]() { CHECK_OK(WorkerMain(port)); });
    absl::SleepFor(absl::Seconds(0.2));

    CHECK_OK(grpc_manager->UpdateWorkerAddress(
        /*worker_idx=*/0, absl::StrCat("localhost:", port)));
  });
  all.Join();
}

TEST(GRPC, DiscardedStubsReleasedOnWorkerAddressUpdate) {
  auto all = CreateGrpcManager();
  auto* grpc_manager = dynamic_cast<GRPCManager*>(all.manager.get());
  ASSERT_NE(grpc_manager, nullptr);

  // Send a blocking request to worker 0.
  EXPECT_OK(all.manager->BlockingRequest("worker_idx", 0).status());

  // Capture a weak pointer to the current stub for worker 0.
  auto stub_weak = grpc_manager->WorkerStubWeakPtrForTesting(0);
  EXPECT_FALSE(stub_weak.expired());

  // Shutdown worker 0.
  EXPECT_OK(grpc_manager->DebugShutdownWorker(0));

  // Create new worker thread on a new port.
  const int new_port = test::PickUnusedPortOrDie();
  all.discarded_worker_threads.push_back(std::move(all.worker_threads.front()));
  all.worker_threads.front() = std::make_unique<utils::concurrency::Thread>(
      [new_port]() { CHECK_OK(WorkerMain(new_port)); });
  absl::SleepFor(absl::Seconds(0.2));

  // Update worker 0 address.
  EXPECT_OK(grpc_manager->UpdateWorkerAddress(
      0, absl::StrCat("localhost:", new_port)));

  // Send another request to worker 0 using the new connection.
  EXPECT_OK(all.manager->BlockingRequest("worker_idx", 0).status());

  // Verify that the old stub has been destroyed and released.
  EXPECT_TRUE(stub_weak.expired());

  EXPECT_OK(all.manager->Done(true));
  all.Join();
}

// A TCP proxy to a worker. "KillBackend" simulates the death of the worker
// process without closing its connections (e.g. a Borg task killed on a
// machine which does not reset the connections): the open connections stay
// open, but no more bytes go through them (half-open connections). The new
// connections go to the new worker restarted at the same address.
class HalfOpenProxy {
 public:
  explicit HalfOpenProxy(const int backend_port) : backend_port_(backend_port) {
    listen_fd_ = socket(AF_INET6, SOCK_STREAM, 0);
    CHECK_GE(listen_fd_, 0);
    const int one = 1;
    CHECK_EQ(
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)), 0);
    sockaddr_in6 address = {};
    address.sin6_family = AF_INET6;
    address.sin6_addr = in6addr_loopback;
    address.sin6_port = 0;
    CHECK_EQ(bind(listen_fd_, reinterpret_cast<sockaddr*>(&address),
                  sizeof(address)),
             0);
    CHECK_EQ(listen(listen_fd_, 16), 0);
    socklen_t length = sizeof(address);
    CHECK_EQ(
        getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&address), &length),
        0);
    port_ = ntohs(address.sin6_port);
    accept_thread_ =
        std::make_unique<utils::concurrency::Thread>([this]() { Accept(); });
  }

  ~HalfOpenProxy() {
    stop_ = true;
    accept_thread_->Join();
    absl::MutexLock lock(mutex_);
    for (auto& connection : connections_) {
      connection->thread->Join();
      close(connection->client_fd);
      close(connection->backend_fd);
    }
    close(listen_fd_);
  }

  int port() const { return port_; }

  // The current connections become half-open, and the next ones go to
  // "new_backend_port".
  void KillBackend(const int new_backend_port) {
    absl::MutexLock lock(mutex_);
    for (auto& connection : connections_) {
      connection->dead = true;
    }
    backend_port_ = new_backend_port;
  }

 private:
  struct Connection {
    int client_fd;
    int backend_fd;
    std::atomic<bool> dead{false};
    std::unique_ptr<utils::concurrency::Thread> thread;
  };

  void Accept() {
    while (!stop_) {
      pollfd poll_fd = {listen_fd_, POLLIN, 0};
      if (poll(&poll_fd, 1, /*timeout=*/100) <= 0) continue;
      const int client_fd = accept(listen_fd_, nullptr, nullptr);
      if (client_fd < 0) continue;
      absl::MutexLock lock(mutex_);
      const int backend_fd = socket(AF_INET6, SOCK_STREAM, 0);
      CHECK_GE(backend_fd, 0);
      sockaddr_in6 address = {};
      address.sin6_family = AF_INET6;
      address.sin6_addr = in6addr_loopback;
      address.sin6_port = htons(backend_port_);
      if (connect(backend_fd, reinterpret_cast<sockaddr*>(&address),
                  sizeof(address)) != 0) {
        close(client_fd);
        close(backend_fd);
        continue;
      }
      auto connection = std::make_unique<Connection>();
      connection->client_fd = client_fd;
      connection->backend_fd = backend_fd;
      Connection* raw = connection.get();
      connection->thread = std::make_unique<utils::concurrency::Thread>(
          [this, raw]() { Forward(raw); });
      connections_.push_back(std::move(connection));
    }
  }

  // Forwards the bytes in both directions. Once dead, drops them, and never
  // closes the connection with the client.
  void Forward(Connection* connection) {
    pollfd poll_fds[2] = {{connection->client_fd, POLLIN, 0},
                          {connection->backend_fd, POLLIN, 0}};
    char buffer[16384];
    while (!stop_) {
      if (poll(poll_fds, 2, /*timeout=*/100) <= 0) continue;
      for (int side = 0; side < 2; side++) {
        if (!(poll_fds[side].revents & (POLLIN | POLLHUP | POLLERR))) continue;
        const ssize_t size = read(poll_fds[side].fd, buffer, sizeof(buffer));
        if (size <= 0) {
          if (!connection->dead) {
            // Closed by an end: close the other end.
            shutdown(connection->client_fd, SHUT_RDWR);
            shutdown(connection->backend_fd, SHUT_RDWR);
            return;
          }
          poll_fds[side].fd = -1;  // Ignored by "poll".
          continue;
        }
        if (connection->dead) continue;
        const int destination = poll_fds[1 - side].fd;
        for (ssize_t sent = 0; sent < size;) {
          const ssize_t num = write(destination, buffer + sent, size - sent);
          if (num <= 0) break;
          sent += num;
        }
      }
    }
  }

  int listen_fd_;
  int port_;
  std::atomic<bool> stop_{false};
  std::unique_ptr<utils::concurrency::Thread> accept_thread_;
  absl::Mutex mutex_;
  int backend_port_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::unique_ptr<Connection>> connections_ ABSL_GUARDED_BY(mutex_);
};

// A worker dies while it runs a request, without closing the connection (the
// connection of the manager becomes half-open), and is restarted at the same
// address. The keep-alive pings detect the dead connection, and the manager
// sends the request again on a new connection, to the new worker (which
// receives the worker configuration): the request completes well before the
// request timeout.
TEST(GRPC, RequestSentAgainToRestartedWorkerOnHalfOpenConnection) {
  ASSERT_OK_AND_ASSIGN(auto worker_1, StartGRPCWorker(/*port=*/0));
  auto proxy = std::make_unique<HalfOpenProxy>(worker_1->port);

  proto::Config config;
  config.set_implementation_key("GRPC");
  config.set_verbosity(1);
  config.set_working_directory(
      file::JoinPath(test::TmpDirectory(), "work_dir"));
  auto& grpc_config = *config.MutableExtension(proto::grpc);
  grpc_config.set_request_timeout_seconds(60);
  grpc_config.set_keepalive_time_ms(500);
  grpc_config.set_keepalive_timeout_ms(1000);
  grpc_config.mutable_grpc_addresses()->add_addresses(
      absl::StrCat("ipv6:[::1]:", proxy->port()));
  ASSERT_OK_AND_ASSIGN(auto manager,
                       CreateManager(config,
                                     /*worker_name=*/kToyWorkerKey,
                                     /*welcome_blob=*/"hello",
                                     /*parallel_execution_per_worker=*/2)
                      );
  EXPECT_OK(manager->BlockingRequest("set:worker_1", 0).status());
  ASSERT_OK_AND_ASSIGN(const std::string result,
                       manager->BlockingRequest("get", 0));
  EXPECT_EQ(result, "worker_1");

  ASSERT_OK_AND_ASSIGN(auto worker_2, StartGRPCWorker(/*port=*/0));
  utils::concurrency::Thread worker_2_thread(
      [&]() { WaitForGRPCWorkerToShutdown(worker_2.get()); });

  // Worker 1 dies while running the 5 seconds "sleep" request.
  utils::concurrency::Thread killer([&]() {
    absl::SleepFor(absl::Seconds(1));
    LOG(INFO) << "Kill worker 1";
    proxy->KillBackend(worker_2->port);
  });
  const absl::Time begin = absl::Now();
  EXPECT_OK(manager->BlockingRequest("sleep", 0).status());
  const absl::Duration duration = absl::Now() - begin;
  killer.Join();
  LOG(INFO) << "Request completed in " << duration;
  // Kill: 1s. Detection: < 2s. Pause after a transient error: 5s. New
  // request: 5s.
  EXPECT_LT(duration, absl::Seconds(30));
  // The request ran on the new worker.
  ASSERT_OK_AND_ASSIGN(const std::string result_after_restart,
                       manager->BlockingRequest("get", 0));
  EXPECT_EQ(result_after_restart, "");

  EXPECT_OK(manager->Done(true));
  worker_2_thread.Join();
  proxy.reset();
  worker_1->stop_server.Notify();
  WaitForGRPCWorkerToShutdown(worker_1.get());
}

}  // namespace
}  // namespace grpc_worker
}  // namespace distribute
}  // namespace yggdrasil_decision_forests
