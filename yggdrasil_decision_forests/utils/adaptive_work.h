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

#ifndef YGGDRASIL_DECISION_FORESTS_TOOL_ADAPTIVE_WORK_H_
#define YGGDRASIL_DECISION_FORESTS_TOOL_ADAPTIVE_WORK_H_

#include "absl/base/thread_annotations.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests {
namespace utils {

// Given a total budget, a number of equally complex tasks that can be
// approximated by an approximation factor, and a stream of how much budget each
// task consumed, returns the optimal approximation factor for all the tasks.
//
// Note: This is different from computing the approximation factor of the
// remaining tasks such that all the tasks can be completed in the budget.
//
// This class is thread safe.
class AdaptativeWork {
 public:
  // Args:
  //   num_tasks: Total number of tasks.
  //   total_budget: Total available budget.
  //   warming_up_budget: Minimum amount of consumed budget before producing an
  //   estimate.
  AdaptativeWork(int num_tasks, double total_budget, double warming_up_budget,
                 double min_factor);

  // Reports a newly completed task.
  void ReportTaskDone(double approximation_factor, double consumed_budget)
      LOCKS_EXCLUDED(mu_);

  double OptimalApproximationFactor() const LOCKS_EXCLUDED(mu_);

 private:
  // Note: See constructor documentation.
  const int num_tasks_;
  const double total_budget_;
  const double warming_up_budget_;
  const double min_factor_;

  // Total consumed budget so far.
  double consumed_budget_ GUARDED_BY(mu_) = 0.0;
  // Number of tasks ran so far.
  int num_ran_tasks_ GUARDED_BY(mu_) = 0;
  // Sum of consumed_budget / approximation_factor for the completed tasks.
  double sum_consumed_budget_div_approximation_factor_ GUARDED_BY(mu_) =
      0.0;
  mutable utils::concurrency::Mutex mu_;
};

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_TOOL_ADAPTIVE_WORK_H_
