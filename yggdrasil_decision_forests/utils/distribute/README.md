# Distribute

**Distribute** is an open-source C++ library that allows for the implementation
of distributed algorithms in systems such as Borg and Cloud. Distribute provides
a small number of low-level primitives, granting developers maximum flexibility.

Distribute is used for all the distributed computation of YDF, including
hyper-parameter tuning, benchmarks, and distributed training.

## Features

-   Logic control in case of worker preemption or failure.
-   Extensive testing on pipelines with hundreds of workers over multiple days.
-   Cross-worker communication.
-   Generic workers can be used in multiple different pipelines at the same
    time.
-   Multiple available backends (Borg workers, TensorFlow Distribute,
    Multi-threads). Multi-threads backend is particularly useful during
    development.

## Computation model

**Initialization**

-   Multiple machines execute a manager and N worker processes. Each worker is
    assigned to an integer id in [0, N). Each worker knows its id.
-   The pipeline is initialized by the manager with a "welcome" data blob (e.g.
    a proto). The welcome blob cannot be modified.
-   The workers are initialized with this welcome blob of data. After a worker
    is preempted and restarted, it is re-initialized with this same welcome blob
    of data.

**Computation**

-   Computation is triggered by "queries".
-   A query is created by the manager or a worker and contains a blob of data
    called the query data (typically a proto).
-   The query is executed by a worker who replies with a blob of data (called
    the answer data) and an absl::Status.
-   Queries can be issued synchronously (blocking) or asynchronously
    (non-blocking).
-   Queries can be sent globally (i.e., any available worker can execute them)
    or to specific workers (identified by the worker id).
-   A worker can emit queries while executing a query. This is useful for
    cascade execution.
-   The number of queries processed by each worker in parallel is configurable
    in the manager and can be adjusted during pipeline execution.
-   If a query computation fails, the absl::Status is returned to the caller.

**Failure scenario**

-   After being preempted and restarted, a worker is re-initialized with a
    welcome blob of data. The welcome blob of data can be used, for example, to
    specify a CNS path to a checkpoint location.
-   If the manager is restarted (e.g., preempted, failure), all the workers are
    restarted.
-   Each query emitter (manager or workers) is responsible for the queries their
    emit.
-   If a worker is interrupted while executing a global query (i.e. a query that
    any worker can execute), the next available worker will execute this query
    automatically.
-   If a worker is interrupted while executing a worker targeted query (i.e. a
    query that can only be executed by a given worker), the query emitter waits
    for the worker to be back online and re-send the query automatically.
-   If a query emitter (manager or worker) is interrupted while a worker is
    executing one of its query, the query answer is discarded.

**Shutdown**

-   When the user code on the manager stops a pipeline, the shutdown method is
    called on all the workers.
-   When the manager stops a pipeline, the worker processes can be interrupted
    or kept running.
-   If the worker processes are not interrupted, a new manager can be created to
    start a new pipeline.

## Minimal example

**The worker : `worker.cc`**

```c++
class ToyWorker final : public AbstractWorker {

public:

  virtual ~ToyWorker()  = default;

  // Initialize the worker with the welcome blob.
  // Note: "Blob" is an alias for "std::string".
  absl::Status Setup(Blob welcome_blob) override {
    LOG(INFO) << "I am worker #" << WorkerIdx();
    return absl::OkStatus();
  }

  // Stop the worker.
  absl::Status Done() override {
    LOG(INFO) << "Bye";
    return absl::OkStatus();
  }

  // Execute a request.
  absl::StatusOr<Blob> RunRequest(Blob blob) override {
    if(blob == "ping") return "pong";
    return absl::InvalidArgumentError("Unknown task");
  }
};

constexpr char kToyWorkerKey[] = "ToyWorker";
REGISTER_Distribution_Worker(ToyWorker, kToyWorkerKey);
```

**The manager : `manager.cc`**

```c++
// Initialize
proto::Config config;
config.set_implementation_key("MULTI_THREAD"); // For debugging
config.MutableExtension(proto::multi_thread)->set_num_workers(5);
auto manager = CreateManager(config, /*worker_name=*/kToyWorkerKey, /*welcome_blob=*/"hello");

// Process

// Blocking request to any worker.
auto result =  manager->BlockingRequest("ping").value();

// Blocking request to a specific worker.
auto result =  manager->BlockingRequest("ping", /*worker_idx=*/ 2).value();

// Async request to any worker.
for(int i=0; i<100; i++){
  manager->AsynchronousRequest("ping");
}
for(int i=0; i<100; i++){
  auto result = manager->NextAsynchronousAnswer().value();
}

// Async request to a specific worker.
for(int i=0; i<100; i++){
  manager->AsynchronousRequest("ping", /*worker_idx=*/ i % manager->NumWorkers());
}
for(int i=0; i<100; i++){
  auto result = manager->NextAsynchronousAnswer().value();
}

// Note: Workers can also execute "AsynchronousRequest".

// Shutdown. This calls "Done" on all the workers and wait until it finishes.
manager->Done();
```

## Examples

### Beginner

-   [unit tests](https://source.corp.google.com/piper///depot/google3/third_party/yggdrasil_decision_forests/utils/distribute/distribute_test.cc):
    Distribute unit tests. Shows all features.

-   [distribute cli](https://source.corp.google.com/piper///depot/google3/third_party/yggdrasil_decision_forests/utils/distribute_cli/):
    Distribute the execution of CLI commands.

### Intermediate

-   [hyperparameter_sweep](https://source.corp.google.com/piper///depot/google3/third_party/yggdrasil_decision_forests/examples/hyperparameter_sweep/README.md):
    Trains and save many models with various input features.

-   [benchmark v2](https://source.corp.google.com/piper///depot/google3/learning/lib/ami/simple_ml/benchmark_v2/README.md):
    An ML benchmark trainings and evaluating millions of models.

-   [hyperparameters optimizer](https://source.corp.google.com/piper///depot/google3/third_party/yggdrasil_decision_forests/learner/hyperparameters_optimizer/BUILD):
    YDF hyper-parameter tuner.

### Advanced

-   [Distributed GBT](https://source.corp.google.com/piper///depot/google3/third_party/yggdrasil_decision_forests/learner/distributed_gradient_boosted_trees/BUILD):
    Distributed GBT training.
