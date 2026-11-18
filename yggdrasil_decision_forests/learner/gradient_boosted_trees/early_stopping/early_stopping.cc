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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/early_stopping/early_stopping.h"

#include <cmath>
#include <optional>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/early_stopping/early_stopping_snapshot.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace learner {
namespace gradient_boosted_trees {

absl::Status EarlyStopping::Update(
    const float validation_loss,
    const std::vector<float>& validation_secondary_metrics, const int num_trees,
    const int current_iter_idx) {
  if (trees_per_iterations_ == -1) {
    return absl::InternalError(
        "The number of trees per iterations should be set before the update");
  }
  if (current_iter_idx >= initial_iteration_) {
    if (best_metric_index_.has_value()) {
      STATUS_CHECK_GE(*best_metric_index_, 0);
      STATUS_CHECK_LT(*best_metric_index_,
                      static_cast<int>(validation_secondary_metrics.size()));
    }
    if (IsBetter(validation_loss, validation_secondary_metrics)) {
      best_loss_ = validation_loss;
      best_metrics_ = validation_secondary_metrics;
      best_num_trees_ = num_trees;
    }
  }
  last_loss_ = validation_loss;
  last_metrics_ = validation_secondary_metrics;
  last_num_trees_ = num_trees;
  return absl::OkStatus();
}

bool EarlyStopping::IsBetter(
    const float validation_loss,
    const std::vector<float>& validation_secondary_metrics) const {
  if (best_metric_index_.has_value() &&
      std::isnan(validation_secondary_metrics[*best_metric_index_])) {
    return false;
  }
  if (best_num_trees_ == -1) {
    return true;
  }
  if (best_metric_index_.has_value()) {
    return best_metric_higher_is_better_
               ? validation_secondary_metrics[*best_metric_index_] >
                     best_metrics_[*best_metric_index_]
               : validation_secondary_metrics[*best_metric_index_] <
                     best_metrics_[*best_metric_index_];
  }
  return validation_loss < best_loss_;
}

bool EarlyStopping::ShouldStop(const int current_iter_idx) {
  if (current_iter_idx < initial_iteration_ || best_num_trees_ == -1) {
    return false;
  }
  if (last_num_trees_ - best_num_trees_ >= num_trees_look_ahead_) {
    if (best_metric_index_.has_value()) {
      LOG(INFO) << "Early stop of the training because the tracked validation "
                   "metric does not "
                << (best_metric_higher_is_better_ ? "increase" : "decrease")
                << " anymore. Best validation metric: "
                << best_metrics_[*best_metric_index_];
    } else {
      LOG(INFO) << "Early stop of the training because the validation "
                   "loss does not decrease anymore. Best valid-loss: "
                << best_loss_;
    }
    return true;
  }
  return false;
}

proto::EarlyStoppingSnapshot EarlyStopping::Save() const {
  proto::EarlyStoppingSnapshot p;
  p.set_best_loss(best_loss_);
  p.set_last_loss(last_loss_);
  p.set_best_num_trees(best_num_trees_);
  p.set_last_num_trees(last_num_trees_);
  p.set_num_trees_look_ahead(num_trees_look_ahead_);
  p.set_trees_per_iterations(trees_per_iterations_);
  p.set_initial_iteration(initial_iteration_);

  *p.mutable_best_metrics() = {best_metrics_.begin(), best_metrics_.end()};
  *p.mutable_last_metrics() = {last_metrics_.begin(), last_metrics_.end()};
  if (best_metric_index_.has_value()) {
    p.set_best_metric_index(*best_metric_index_);
    p.set_best_metric_higher_is_better(best_metric_higher_is_better_);
  }
  return p;
}

absl::Status EarlyStopping::Load(const proto::EarlyStoppingSnapshot& p) {
  best_loss_ = p.best_loss();
  last_loss_ = p.last_loss();
  best_num_trees_ = p.best_num_trees();
  last_num_trees_ = p.last_num_trees();
  num_trees_look_ahead_ = p.num_trees_look_ahead();
  trees_per_iterations_ = p.trees_per_iterations();
  initial_iteration_ = p.initial_iteration();

  best_metrics_ = {p.best_metrics().begin(), p.best_metrics().end()};
  last_metrics_ = {p.last_metrics().begin(), p.last_metrics().end()};
  if (p.has_best_metric_index()) {
    STATUS_CHECK_GE(p.best_metric_index(), 0);
    if (best_num_trees_ != -1) {
      STATUS_CHECK_LT(p.best_metric_index(),
                      static_cast<int>(best_metrics_.size()));
    }
    best_metric_index_ = p.best_metric_index();
    best_metric_higher_is_better_ = p.best_metric_higher_is_better();
  } else {
    best_metric_index_.reset();
    best_metric_higher_is_better_ = false;
  }
  return absl::OkStatus();
}

}  // namespace gradient_boosted_trees
}  // namespace learner
}  // namespace yggdrasil_decision_forests
