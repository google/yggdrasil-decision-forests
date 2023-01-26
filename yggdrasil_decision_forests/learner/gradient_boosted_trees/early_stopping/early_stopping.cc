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

#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/early_stopping/early_stopping_snapshot.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"

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
  if (current_iter_idx >= initial_iteration_ &&
      (best_num_trees_ == -1 || validation_loss < best_loss_)) {
    best_loss_ = validation_loss;
    best_metrics_ = validation_secondary_metrics;
    best_num_trees_ = num_trees;
  }
  last_loss_ = validation_loss;
  last_metrics_ = validation_secondary_metrics;
  last_num_trees_ = num_trees;
  return absl::OkStatus();
}

bool EarlyStopping::ShouldStop(const int current_iter_idx) {
  if (current_iter_idx < initial_iteration_) {
    return false;
  }
  if (last_num_trees_ - best_num_trees_ >= num_trees_look_ahead_) {
    YDF_LOG(INFO) << "Early stop of the training because the validation "
                     "loss does not decrease anymore. Best valid-loss: "
                  << best_loss_;
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
  return absl::OkStatus();
}

}  // namespace gradient_boosted_trees
}  // namespace learner
}  // namespace yggdrasil_decision_forests
