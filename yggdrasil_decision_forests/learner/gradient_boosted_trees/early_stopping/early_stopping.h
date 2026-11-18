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

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_EARLY_STOPPING_EARLY_STOPPING_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_EARLY_STOPPING_EARLY_STOPPING_H_

#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/early_stopping/early_stopping_snapshot.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace learner {
namespace gradient_boosted_trees {

class EarlyStopping {
 public:
  EarlyStopping(const int early_stopping_num_trees_look_ahead,
                const int initial_iteration)
      : num_trees_look_ahead_(early_stopping_num_trees_look_ahead),
        initial_iteration_(initial_iteration) {
    DCHECK_GE(initial_iteration_, 0);
  }

  // Updates the internal state of the early stopping controller.
  //
  // "set_trees_per_iterations" should be called before the first update.
  absl::Status Update(const float validation_loss,
                      const std::vector<float>& validation_secondary_metrics,
                      const int num_trees, const int current_iter_idx);

  // Should the training stop?
  bool ShouldStop(const int current_iter_idx);

  // Best model.
  int best_num_trees() const { return best_num_trees_; }
  float best_loss() const { return best_loss_; }
  const std::vector<float>& best_metrics() const { return best_metrics_; }

  // Last model.
  float last_loss() const { return last_loss_; }
  const std::vector<float>& last_metrics() const { return last_metrics_; }
  // Number of trees of the model during the last update. Zero if the model was
  // never evaluated on the validation dataset.
  int last_num_trees() const { return last_num_trees_; }

  // Number of trees trained at each iteration. "set_trees_per_iterations"
  // should be called before the first update.
  int trees_per_iterations() const { return trees_per_iterations_; }
  void set_trees_per_iterations(const int trees_per_iterations) {
    trees_per_iterations_ = trees_per_iterations;
  }

  // Metric index and optimization direction used for early stopping when
  // "early_stopping_custom_metric" is set.
  std::optional<int> best_metric_index() const { return best_metric_index_; }
  bool best_metric_higher_is_better() const {
    return best_metric_higher_is_better_;
  }
  void set_best_metric(const int best_metric_index,
                       const bool best_metric_higher_is_better) {
    best_metric_index_ = best_metric_index;
    best_metric_higher_is_better_ = best_metric_higher_is_better;
  }

  // Exports the internal representation of the class to a proto.
  proto::EarlyStoppingSnapshot Save() const;

  // Restores the internal representation of the class from a proto.
  absl::Status Load(const proto::EarlyStoppingSnapshot& p);

 private:
  bool IsBetter(float validation_loss,
                const std::vector<float>& validation_secondary_metrics) const;


  // Validation loss at the best step of the model. Only valid if
  // "best_num_trees_ >= 0".
  float best_loss_ = 0.f;
  float last_loss_ = 0.f;

  std::vector<float> last_metrics_;
  std::vector<float> best_metrics_;

  // Number of trees in the model at the best validation step.
  int best_num_trees_ = -1;
  int last_num_trees_ = 0;

  int num_trees_look_ahead_;

  std::optional<int> best_metric_index_;
  bool best_metric_higher_is_better_ = false;

  // Iteration that starts the computation of the validation loss.
  //
  // See GradientBoostedTreesTrainingConfig::early_stopping_initial_iteration
  // for more details.
  int initial_iteration_;

  int trees_per_iterations_ = -1;

};

}  // namespace gradient_boosted_trees
}  // namespace learner
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_EARLY_STOPPING_EARLY_STOPPING_H_
