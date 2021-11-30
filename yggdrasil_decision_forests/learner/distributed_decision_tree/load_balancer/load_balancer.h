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

// The LoadBalancer class helps balancing the workload among workers for
// distributed decision tree learning when workers have non-uniform and
// dynamically changing working speed -- commonly the case on shared
// cloud servers.
//
// It achieves that by dynamically controlling the mapping between features
// and workers.
//
// At each iteration, the user calls "AddWorkDurationMeasurement()" to provide
// the amount of time each worker took to execute the feature dependent tasks
// i.e. finding the best splits. Following multiple time measures were some
// workers are significantly slower than the others, the load balancer will
// propose a set of orders to transfer the ownership of features from the slow
// workers to the fast workers. Those orders are pending while the workers are
// loading from disk the data of the newly received features. Once all the
// workers have been prepared for the change (e.g. loading in the background the
// new features), the order can be applied
// ("ApplyPendingOrder()").
//
// Internally, the work balancer algorithm works as follows:
//
// 1. The computation speed of each worker is estimated over a rolling period of
//    time (the last `estimation_window_length` observations).
// 2. Workers that run slower than "median_margin_ratio x the medium worker
//    time" (e.g. median_margin_ratio=2) are considered "slow".
// 3. The computation time per feature of all the workers are estimated. The
//    balancer is proposing a balancing plan where the features of slow workers
//    are re-assigned to faster workers (while making sure not to create new
//    slow workers using the speed estimates).
// The plan is created with the following constraints:
//   - No worker can get assigned with more than "(1 + max_unbalance_ratio) x
//   number of initially assigned features".
//   - A plan cannot change more than
//   "max_balancing_changes_per_dynamic_balancing" feature allocations at the
//   same time.
// 4. Once a plan is created, workers start loading the new features in the
//    background.
// 5. Once all the workers are done loading the new features (which can take a
//    while), the new plan is put into effect (i.e. the manager will make
//    queries with the new feature assignation). The unused features on the slow
//    workers are also unloaded from memory.
//
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_LOAD_BALANCER_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_LOAD_BALANCER_H_

#include "absl/time/time.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache.pb.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/load_balancer/load_balancer.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {

class LoadBalancer {
 public:
  // Time measurement of a work done by a worker.
  struct Measure {
    double time = 0;       // Work wall time in seconds.
    int num_features = 0;  // Number of features processed in the work.
    bool operator<(const Measure& other) { return time < other.time; }
  };

  // A unit order defines a change in the attribution of the features to the
  // workers e.g. transferring feature #1 ("feature") from worker #5
  // ("source_worker") to worker #7 ("destination_worker").
  struct FeatureAssignedWorkerChange {
    int source_worker;
    int destination_worker;
    int feature;
  };

  // An order is composed of multiple UnitOrders. A given feature cannot be in
  // multiple unit orders of an order. A feature cannot be transferred from a
  // worker to itself.
  typedef std::vector<FeatureAssignedWorkerChange> Order;

  // A change as seen by a worker.
  struct ChangePerWorker {
    std::vector<int> load_features;    // The received features.
    std::vector<int> unload_features;  // The removed features.
  };

  // Initialize the load balancer.
  static utils::StatusOr<LoadBalancer> Create(
      const std::vector<int>& features, int num_workers,
      const dataset_cache::proto::CacheMetadata& cache_metadata,
      const proto::LoadBalancerOptions& options);

  // Gets the index of the workers currently owning a given feature.
  utils::StatusOr<int> FeatureOwner(int feature) const;

  // Adds a new work time measurement. Possibly, triggers the dynamic
  // balancing of the features and propose some pending changes (return true).
  utils::StatusOr<bool> AddWorkDurationMeasurement(
      const std::vector<Measure>& measure_per_workers);

  // Adds a new feature loading time measurement. At least one measure of
  // feature loading time should be provided before any call to
  // "AddWorkDurationMeasurement".
  void AddFeatureLoadingDurationMeasurement(double time);

  // Checks is a change is pending for execution.
  bool HasPendingOrder() const;

  // Applies the current pending changes. If a change in pending, no new
  // order/optimization will be proposed.
  absl::Status ApplyPendingOrder();

  // Pending change for a specific worker.
  const ChangePerWorker& PendingOrderPerWorker(int worker) const;

  // Human readable information about the balancer.
  std::string Info(bool detailed = true) const;

  // Mapping feature -> worker.
  std::vector<int> WorkersPerFeatures() const;

  // Mapping worker -> {list of features}. If possible use
  // "FeaturesPerWorker(worker)" instead.
  std::vector<std::vector<int>> FeaturesPerWorkers() const;

  // Features owned by the worker. More efficient than
  // FeaturesPerWorkers()[worker].
  const std::vector<int>& FeaturesPerWorker(int worker) const;

  // Checks if the dynamic balancing is active i.e. the worker->feature
  // assignment can change.
  bool is_dynamic_balancing_active() const {
    return options_.dynamic_balancing_frequency_iteration() > 1 ||
           options_.dynamic_balancing_frequency_seconds() > 1;
  }

  // Creates a "plan" to share efficiently the evaluation split values of a set
  // of features.
  //
  // A "plan" is a partially ordered sequence of instructions of the form
  // "worker i request the evaluation of split j to worker k".
  //
  // Plans are currently made with two rounds such that each worker only talks
  // to sqrt(num_workers) at each round.
  //
  // For example, if a single worker needs to share a split value to 99 other
  // workers. It will first share it to 9 other workers. Then, in a second
  // round, the 10 workers than now have the evaluation split (the original
  // worker and the 9 new ones) will share it to the remaining 90 workers (9
  // output communication for each).
  utils::StatusOr<proto::SplitSharingPlan> MakeSplitSharingPlan(
      const std::vector<int>& feature_idxs);

 private:
  // Data about a worker.
  struct Worker {
    // Features owned by this worker.
    std::vector<int> features;

    // Time-series of measures. Each measure is a
    std::vector<Measure> measures;

    // Pending orders for this specific worker.
    ChangePerWorker pending_orders;
  };

  // Data about a dataset feature.
  struct Feature {
    // Estimation of the "cost" of a feature. The absolute values does not
    // matter, only the order and the relative magnitude in between scores.
    double cost_score = -1.;

    // Currently, owning worker.
    int worker = -1;

    // True the feature is used for training and should be assigned to a
    // worker.
    bool active = false;
  };

  LoadBalancer() {}

  // Initialize the load balancer.
  absl::Status Initialize(
      const std::vector<int>& features, int num_workers,
      const proto::LoadBalancerOptions& options,
      const dataset_cache::proto::CacheMetadata& cache_metadata);

  // Estimation of the working time of a worker (total and per features) of a
  // worker.
  struct WorkTimeEstimate {
    double time_per_features;
    int num_features;
    int worker_idx;

    double time() const { return time_per_features * num_features; }

    bool operator<(const WorkTimeEstimate& other) const {
      return time() < other.time();
    }
  };
  std::vector<WorkTimeEstimate> CreateWorkTimeEstimatePerWorker() const;

  // Tries to balance the worker. After "TryCreateBalancingOrders" is called,
  // and if a new change was proposed, "HasPendingOrder()" will be true.
  absl::Status TryCreateBalancingOrders();

  // Creates a random balancing of the workers. The balancing does not take
  // any previous feature assignation or worker speed. This method is only
  // used for unit testing.
  absl::Status CreateRandomBalancingOrders();

  // Estimate the time is takes for a worker to load a new feature.
  utils::StatusOr<double> EstimateFeatureLoadingTime() const;

  // Relative cost of each feature.
  utils::StatusOr<double> CostPerFeatureType(
      int feature, const dataset_cache::proto::CacheMetadata& cache_metadata);

  int GetWorstCandidateWallTime(
      const std::vector<LoadBalancer::WorkTimeEstimate>& sorted_wall_times)
      const;

  int GetBestCandidateWallTime(
      const std::vector<LoadBalancer::WorkTimeEstimate>& sorted_wall_times)
      const;

  // Balancer options.
  proto::LoadBalancerOptions options_;

  // Indices of the features that need assignment.
  std::vector<int> active_features_;

  // Maximum number of features to assign to a worker.
  int max_num_features_per_workers_;

  // Per worker data.
  std::vector<Worker> workers_;

  // Per feature data.
  std::vector<Feature> features_;

  // Number of time measurements received so far.
  int num_measures_;

  // Sum of the leading time measurement received so far.
  double sum_feature_loading_time_;

  // Number of loading time measurement received so far.
  int num_feature_loading_time_;

  // Random generator.
  utils::RandomEngine random_;

  // Number of feature reassignment proposed so far.
  int num_unit_orders_ = 0;

  // Number of re-balancing optimizations done so far.
  int num_rebalances_ = 0;

  // Number of orders emitted so far i.e. number of re-balancing that lead to
  // a change in feature assignation.
  int num_orders = 0;

  // Pending orders.
  Order pending_orders_;

  // Value "num_measures_" the last time re-balancing was tried.
  int num_measures_last_balancing_ = 0;

  // Time  at that last  re-balancing was tried.
  absl::Time time_last_balancing_;
};

}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DISTRIBUTED_DECISION_TREE_LOAD_BALANCER_H_
