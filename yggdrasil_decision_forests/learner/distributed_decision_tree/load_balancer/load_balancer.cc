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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/load_balancer/load_balancer.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/substitute.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {

utils::StatusOr<LoadBalancer> LoadBalancer::Create(
    const std::vector<int>& features, int num_workers,
    const dataset_cache::proto::CacheMetadata& cache_metadata,
    const proto::LoadBalancerOptions& options) {
  LoadBalancer balancer;
  RETURN_IF_ERROR(
      balancer.Initialize(features, num_workers, options, cache_metadata));
  return balancer;
}

absl::Status LoadBalancer::Initialize(
    const std::vector<int>& features, int num_workers,
    const proto::LoadBalancerOptions& options,
    const dataset_cache::proto::CacheMetadata& cache_metadata) {
  if (features.empty()) {
    return absl::InvalidArgumentError("Empty features");
  }
  if (num_workers <= 0) {
    return absl::InvalidArgumentError("No workers");
  }

  // Reset the attributes.
  workers_.assign(num_workers, {});
  active_features_ = features;
  options_ = options;
  num_measures_ = 0;
  sum_feature_loading_time_ = 0;
  num_feature_loading_time_ = 0;

  const int max_feature_idx_ =
      *std::max_element(features.begin(), features.end());
  features_.assign(max_feature_idx_ + 1, {});

  // Score the features.
  std::vector<std::pair<double, int>> feature_and_scores;
  feature_and_scores.reserve(active_features_.size());
  for (const auto feature : active_features_) {
    ASSIGN_OR_RETURN(const auto score,
                     CostPerFeatureType(feature, cache_metadata));
    feature_and_scores.push_back({score, feature});
  }
  std::sort(feature_and_scores.begin(), feature_and_scores.end(),
            std::greater<>());

  // Initial feature assignment.
  int cur = 0;
  for (const auto feature_and_score : feature_and_scores) {
    const auto worker_idx = (cur++) % num_workers;
    const auto feature_idx = feature_and_score.second;
    workers_[worker_idx].features.push_back(feature_idx);
    features_[feature_idx].worker = worker_idx;
    features_[feature_idx].active = true;
    features_[feature_idx].cost_score = feature_and_score.first;
  }

  const auto max_num_features = workers_.front().features.size();
  max_num_features_per_workers_ =
      static_cast<int>(std::ceil(static_cast<double>(max_num_features) *
                                 (1.0 + options_.max_unbalance_ratio())));

  LOG(INFO) << "Create load balancer:\n" << Info();
  return absl::OkStatus();
}

utils::StatusOr<int> LoadBalancer::FeatureOwner(int feature) const {
  if (!features_[feature].active) {
    return absl::InternalError("Non active feature");
  }
  if (features_[feature].worker < 0) {
    return absl::InternalError("Non assigned feature");
  }
  return features_[feature].worker;
}

std::string LoadBalancer::Info(bool detailed) const {
  std::string info;

  if (detailed) {
    absl::SubstituteAndAppend(&info, "Features($0):\n",
                              active_features_.size());
    for (const auto& feature_idx : active_features_) {
      const auto& feature = features_[feature_idx];
      absl::SubstituteAndAppend(&info, "\t#$0 worker:$1 score:$2\n",
                                feature_idx, feature.worker,
                                feature.cost_score);
    }
    absl::SubstituteAndAppend(&info, "\n");

    absl::SubstituteAndAppend(&info, "Workers($0):\n", workers_.size());
    for (int worker_idx = 0; worker_idx < workers_.size(); worker_idx++) {
      const auto& worker = workers_[worker_idx];
      absl::SubstituteAndAppend(&info,
                                "\t#$0 measure:$1 features($2):", worker_idx,
                                worker.measures.size(), worker.features.size());
      for (const auto feature : worker.features) {
        absl::SubstituteAndAppend(&info, " $0", feature);
      }
      absl::SubstituteAndAppend(&info, "\n");
    }
    absl::SubstituteAndAppend(&info, "Maximum features per workers: $0\n",
                              max_num_features_per_workers_);
  } else {
    absl::SubstituteAndAppend(&info, "workers:$0", workers_.size());
    absl::SubstituteAndAppend(&info, " features:$0", active_features_.size());
    absl::SubstituteAndAppend(&info, " measures:$0", num_measures_);
    absl::SubstituteAndAppend(&info, " pending-order:$0", HasPendingOrder());
    absl::SubstituteAndAppend(&info, " max-features-per-workers:$0",
                              max_num_features_per_workers_);
    absl::SubstituteAndAppend(&info, " rebalance-with-change:$0/$1", num_orders,
                              num_rebalances_);
    absl::SubstituteAndAppend(&info, " total_num_changes:$0", num_unit_orders_);

    if (num_feature_loading_time_ > 0) {
      absl::SubstituteAndAppend(
          &info, " feature-loading-time:$0",
          sum_feature_loading_time_ / num_feature_loading_time_);
      absl::SubstituteAndAppend(
          &info, " feature-loading-time:$0",
          sum_feature_loading_time_ / num_feature_loading_time_);
    }
  }

  return info;
}

utils::StatusOr<bool> LoadBalancer::AddWorkDurationMeasurement(
    const std::vector<Measure>& measure_per_workers) {
  if (measure_per_workers.size() != workers_.size()) {
    return absl::InternalError("Wrong number of workers");
  }

  for (int worker_idx = 0; worker_idx < workers_.size(); worker_idx++) {
    auto& worker = workers_[worker_idx];
    const auto& measure = measure_per_workers[worker_idx];

    if (measure.num_features > worker.features.size()) {
      return absl::InternalError(absl::Substitute(
          "Measurement with $0 features received for worker #$1 with $2 "
          "features.",
          measure.num_features, worker_idx, worker.features.size()));
    }

    if (measure.num_features == 0) {
      continue;
    }

    // TODO(gbm): Use circular buffer.
    worker.measures.insert(worker.measures.begin(), measure);
    if (worker.measures.size() > options_.estimation_window_length()) {
      worker.measures.resize(options_.estimation_window_length());
    }
  }

  num_measures_++;
  if (!HasPendingOrder()) {
    const auto now = absl::Now();

    const bool run_balancing =
        // Iteration criterion.
        (options_.dynamic_balancing_frequency_iteration() > 0 &&
         (num_measures_ - num_measures_last_balancing_ >=
          options_.dynamic_balancing_frequency_iteration())) ||
        // Time criterion.
        (options_.dynamic_balancing_frequency_seconds() > 0 &&
         (now - time_last_balancing_ >=
          absl::Seconds(options_.dynamic_balancing_frequency_seconds())));

    if (run_balancing) {
      num_measures_last_balancing_ = num_measures_;
      time_last_balancing_ = now;
      if (options_.internal().random_dynamic_balancing()) {
        RETURN_IF_ERROR(CreateRandomBalancingOrders());
        return true;
      } else {
        const int save_num_orders = num_orders;
        RETURN_IF_ERROR(TryCreateBalancingOrders());
        return num_orders != save_num_orders;
      }
    }
  }
  return false;
}

void LoadBalancer::AddFeatureLoadingDurationMeasurement(double time) {
  sum_feature_loading_time_ += time;
  num_feature_loading_time_++;
}

bool LoadBalancer::HasPendingOrder() const { return !pending_orders_.empty(); }

const LoadBalancer::ChangePerWorker& LoadBalancer::PendingOrderPerWorker(
    int worker) const {
  return workers_[worker].pending_orders;
}

absl::Status LoadBalancer::ApplyPendingOrder() {
  if (!pending_orders_.empty()) {
    LOG(INFO) << "Apply pending orders";
  }

  for (const auto& order : pending_orders_) {
    auto& src_features = workers_[order.source_worker].features;
    auto& dst_features = workers_[order.destination_worker].features;

    if (features_[order.feature].worker != order.source_worker) {
      return absl::InternalError(absl::Substitute(
          "Invalid order. Source worker $0 does not own feature $1. Instead, "
          "it is worker $2 that owns it.",
          order.source_worker, order.feature, features_[order.feature].worker));
    }

    src_features.erase(
        std::remove(src_features.begin(), src_features.end(), order.feature),
        src_features.end());
    dst_features.push_back(order.feature);

    if (!options_.internal().random_dynamic_balancing()) {
      if (src_features.empty()) {
        return absl::InternalError(
            absl::StrCat("Invalid order. No more features for worker #",
                         order.source_worker));
      }

      if (dst_features.size() > max_num_features_per_workers_) {
        return absl::InternalError(
            absl::StrCat("Invalid order. Too many features for worker #",
                         order.destination_worker));
      }
    }

    features_[order.feature].worker = order.destination_worker;
  }

  pending_orders_.clear();
  for (auto& worker : workers_) {
    worker.pending_orders.load_features.clear();
    worker.pending_orders.unload_features.clear();
  }

  return absl::OkStatus();
}

std::vector<LoadBalancer::WorkTimeEstimate>
LoadBalancer::CreateWorkTimeEstimatePerWorker() const {
  std::vector<WorkTimeEstimate> estimates;

  for (int worker_idx = 0; worker_idx < workers_.size(); worker_idx++) {
    auto& worker = workers_[worker_idx];
    if (worker.measures.size() < options_.estimation_window_length()) {
      continue;
    }
    double sum_wall_time = 0;
    double weight_wall_time = 0;
    for (const auto& measure : worker.measures) {
      sum_wall_time += measure.time;
      weight_wall_time += measure.num_features;
    }

    if (weight_wall_time > 0) {
      const auto estimated_wall_time = sum_wall_time / weight_wall_time;
      estimates.push_back(
          {/*time_per_features=*/estimated_wall_time,
           /*num_features=*/static_cast<int>(worker.features.size()),
           /*worker_idx=*/worker_idx});
    }
  }

  return estimates;
}

utils::StatusOr<double> LoadBalancer::EstimateFeatureLoadingTime() const {
  if (num_feature_loading_time_ == 0) {
    return absl::InternalError(
        "At least one measurement of feature loading time required.");
  }
  return sum_feature_loading_time_ / num_feature_loading_time_;
}

absl::Status LoadBalancer::CreateRandomBalancingOrders() {
  LOG(INFO) << "Random re-balancing of the features";

  std::vector<int> active_feature_to_worker;
  active_feature_to_worker.reserve(active_features_.size());
  for (int active_feature_idx = 0; active_feature_idx < active_features_.size();
       active_feature_idx++) {
    active_feature_to_worker.push_back(active_feature_idx % workers_.size());
  }
  std::shuffle(active_feature_to_worker.begin(), active_feature_to_worker.end(),
               random_);

  // Randomly re-assign each feature to a worker.
  int num_unit_orders = 0;
  for (int active_feature_idx = 0; active_feature_idx < active_features_.size();
       active_feature_idx++) {
    const auto feature = active_features_[active_feature_idx];
    const auto dst_worker = active_feature_to_worker[active_feature_idx];
    const auto src_worker = features_[feature].worker;
    if (dst_worker == src_worker) {
      continue;
    }

    pending_orders_.push_back({
        /*source_worker=*/
        src_worker,
        /*destination_worker=*/dst_worker,
        /*feature=*/feature,
    });
    workers_[src_worker].pending_orders.unload_features.push_back(feature);
    workers_[dst_worker].pending_orders.load_features.push_back(feature);
    num_unit_orders++;
  }
  max_num_features_per_workers_ = active_features_.size();
  LOG(INFO) << num_unit_orders << " unit orders generated";
  return absl::OkStatus();
}

absl::Status LoadBalancer::TryCreateBalancingOrders() {
  ASSIGN_OR_RETURN(const auto feature_loading_time,
                   EstimateFeatureLoadingTime());
  LOG(INFO) << "Try to balance workers' load [feature-loading-time:"
            << feature_loading_time
            << " max-features-per-workers:" << max_num_features_per_workers_
            << "]";
  int num_unit_orders = 0;

  auto time_per_worker = CreateWorkTimeEstimatePerWorker();

  const auto print_debug_info = [&time_per_worker]() {
    // Print debug infos.
    LOG(INFO) << "Best and worst sorted wall time per workers:";
    const int num_display_items = 5;
    for (int idx = 0; idx < num_display_items && idx < time_per_worker.size();
         idx++) {
      const auto& item = time_per_worker[idx];
      LOG(INFO) << "\tidx:" << idx
                << " time:" << item.time_per_features * item.num_features
                << " time-p-f:" << item.time_per_features
                << " worker:" << item.worker_idx
                << " num_features:" << item.num_features;
    }
    if (time_per_worker.size() > num_display_items) {
      LOG(INFO) << "\t...";
      for (int idx = std::max<int>(time_per_worker.size() - num_display_items,
                                   num_display_items);
           idx < time_per_worker.size(); idx++) {
        const auto& item = time_per_worker[idx];
        LOG(INFO) << "\tidx:" << idx
                  << " time:" << item.time_per_features * item.num_features
                  << " time-p-f:" << item.time_per_features
                  << " worker:" << item.worker_idx
                  << " num_features:" << item.num_features;
      }
    }
  };

  // List of features already used in one of the order.
  absl::flat_hash_set<int> used_features;

  for (int round_idx = 0;
       round_idx < options_.max_balancing_changes_per_dynamic_balancing();
       round_idx++) {
    // From the fastest (good) to the slowest (bad) workers.
    std::sort(time_per_worker.begin(), time_per_worker.end());

    // Print debug infos.
    if (round_idx == 0) {
      print_debug_info();
    }

    if (time_per_worker.size() < 2) {
      break;
    }

    double median_time;
    if ((time_per_worker.size() % 2) == 0) {
      // even
      median_time = (time_per_worker[time_per_worker.size() / 2].time() +
                     time_per_worker[time_per_worker.size() / 2 - 1].time()) /
                    2;
    } else {
      // odd
      median_time = time_per_worker[time_per_worker.size() / 2].time();
    }
    const auto min_src_time = median_time * options_.median_margin_ratio();
    LOG(INFO) << "Bad workers: median-worker-time: " << median_time
              << " minimum-time-for-bad-worker: " << min_src_time;

    // Get the source and destination workers.
    const auto src_time_idx = GetWorstCandidateWallTime(time_per_worker);
    const auto dst_time_idx = GetBestCandidateWallTime(time_per_worker);
    if (dst_time_idx == -1 || src_time_idx == -1 ||
        time_per_worker[src_time_idx].time() < min_src_time) {
      break;
    }

    if (src_time_idx <= dst_time_idx) {
      break;
    }

    const auto src_worker = time_per_worker[src_time_idx].worker_idx;
    const auto dst_worker = time_per_worker[dst_time_idx].worker_idx;

    // TODO(gbm): After the first one, all the other transfer costs are free.
    const double gain = time_per_worker[src_time_idx].time_per_features -
                        time_per_worker[dst_time_idx].time_per_features;
    if (gain <= 0) {
      break;
    }

    // Select an available feature i.e. a feature not already used.
    const auto& candidate_features = workers_[src_worker].features;
    int feature = -1;
    for (auto rit = candidate_features.rbegin();
         rit != candidate_features.rend(); rit++) {
      if (used_features.find(*rit) == used_features.end()) {
        feature = *rit;
        break;
      }
    }
    if (feature == -1) {
      return absl::InternalError("No available feature");
    }
    used_features.insert(feature);
    time_per_worker[src_time_idx].num_features--;
    time_per_worker[dst_time_idx].num_features++;

    pending_orders_.push_back({
        /*source_worker=*/src_worker,
        /*destination_worker=*/dst_worker,
        /*feature=*/feature,
    });
    workers_[src_worker].pending_orders.unload_features.push_back(feature);
    workers_[dst_worker].pending_orders.load_features.push_back(feature);

    LOG(INFO) << "Reassigning feature " << feature << " from worker #"
              << src_worker << " to worker #" << dst_worker
              << " for an estimated gain of " << gain;
    num_unit_orders++;
  }

  if (num_unit_orders > 0) {
    LOG(INFO) << "Create order with " << num_unit_orders
              << " feature transfers";
    num_orders++;
  }
  num_unit_orders_ += num_unit_orders;
  num_rebalances_++;
  return absl::OkStatus();
}

std::vector<int> LoadBalancer::WorkersPerFeatures() const {
  std::vector<int> result;
  result.reserve(features_.size());
  for (const auto& feature : features_) {
    result.push_back(feature.worker);
  }
  return result;
}

std::vector<std::vector<int>> LoadBalancer::FeaturesPerWorkers() const {
  std::vector<std::vector<int>> result;
  result.reserve(workers_.size());
  for (const auto& worker : workers_) {
    result.push_back({worker.features.begin(), worker.features.end()});
  }
  return result;
}

const std::vector<int>& LoadBalancer::FeaturesPerWorker(int worker) const {
  return workers_[worker].features;
}

utils::StatusOr<double> LoadBalancer::CostPerFeatureType(
    int feature, const dataset_cache::proto::CacheMetadata& cache_metadata) {
  // TODO(gbm): Tune these costs.
  const double very_large = 1000000;
  const auto& col_metadata = cache_metadata.columns(feature);
  switch (col_metadata.type_case()) {
    case dataset_cache::proto::CacheMetadata_Column::kNumerical:
      if (col_metadata.numerical().discretized()) {
        return 1.0 + static_cast<double>(
                         col_metadata.numerical().num_discretized_values()) /
                         very_large;
      } else {
        return 5.0;
      }

    case dataset_cache::proto::CacheMetadata_Column::kCategorical:
      return 1.0 +
             static_cast<double>(col_metadata.categorical().num_values()) /
                 very_large;

    case dataset_cache::proto::CacheMetadata_Column::kBoolean:
      return 1.0;

    default:
      return absl::InternalError("Feature type not supported in balancer");
  }
}

int LoadBalancer::GetWorstCandidateWallTime(
    const std::vector<LoadBalancer::WorkTimeEstimate>& sorted_wall_times)
    const {
  for (int i = sorted_wall_times.size() - 1; i > 0; i--) {
    if (sorted_wall_times[i].num_features > 1) {
      return i;
    }
  }
  return -1;
}

int LoadBalancer::GetBestCandidateWallTime(
    const std::vector<LoadBalancer::WorkTimeEstimate>& sorted_wall_times)
    const {
  for (int i = 0; i < sorted_wall_times.size() - 1; i++) {
    if (sorted_wall_times[i].num_features < max_num_features_per_workers_) {
      return i;
    }
  }
  return -1;
}

utils::StatusOr<proto::SplitSharingPlan> LoadBalancer::MakeSplitSharingPlan(
    const std::vector<int>& feature_idxs) {
  proto::SplitSharingPlan plan;
  auto& round_1 = *plan.add_rounds();
  auto& round_2 = *plan.add_rounds();

  // Group the features by ownership.
  std::vector<std::vector<int>> workers_to_features(workers_.size());
  for (auto feature_idx : feature_idxs) {
    ASSIGN_OR_RETURN(auto src_worker, FeatureOwner(feature_idx));
    workers_to_features[src_worker].push_back(feature_idx);
  }

  const int num_outputs_per_round =
      static_cast<int>(std::ceil(std::sqrt(workers_.size())));

  int next_round_1_dst_worker = 0;
  for (int src_worker_idx = 0; src_worker_idx < workers_to_features.size();
       src_worker_idx++) {
    const auto& src_features = workers_to_features[src_worker_idx];
    if (src_features.empty()) {
      continue;
    }
    // Index of the workers currently having the evaluation data.
    std::vector<int> workers_with_feature;

    // The owner of the feature has always the evaluation data.
    workers_with_feature.push_back(src_worker_idx);

    // First round
    for (int local_dst_worker = 0; local_dst_worker < num_outputs_per_round;
         local_dst_worker++) {
      if (next_round_1_dst_worker == src_worker_idx) {
        next_round_1_dst_worker =
            (next_round_1_dst_worker + 1) % workers_.size();
      }
      const int dst_worker = next_round_1_dst_worker;
      workers_with_feature.push_back(dst_worker);

      next_round_1_dst_worker = (next_round_1_dst_worker + 1) % workers_.size();

      auto& request = (*round_1.mutable_requests())[dst_worker];
      auto& item = *request.add_items();
      item.set_src_worker(src_worker_idx);
      *item.mutable_features() = {src_features.begin(), src_features.end()};
    }

    std::sort(workers_with_feature.begin(), workers_with_feature.end());

    // Second round.
    int next_local_src_worker_idx = 0;
    for (int dst_worker = 0; dst_worker < workers_.size(); dst_worker++) {
      // Check if this "dst_worker" worker already has the feature data.
      const auto lb = std::lower_bound(workers_with_feature.begin(),
                                       workers_with_feature.end(), dst_worker);
      if (lb != workers_with_feature.end() && *lb == dst_worker) {
        continue;
      }

      auto& request = (*round_2.mutable_requests())[dst_worker];
      auto& item = *request.add_items();
      item.set_src_worker(workers_with_feature[next_local_src_worker_idx]);
      next_local_src_worker_idx =
          (next_local_src_worker_idx + 1) % workers_with_feature.size();
      *item.mutable_features() = {src_features.begin(), src_features.end()};
    }
  }

  // Set the "last_request_of_plan" flag.
  for (int worker = 0; worker < workers_.size(); worker++) {
    // We make sure that all the workers are running the final logic at the same
    // time.
    (*round_2.mutable_requests())[worker].set_last_request_of_plan(true);
  }

  return plan;
}

}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
