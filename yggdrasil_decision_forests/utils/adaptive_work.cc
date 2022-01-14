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

#include "yggdrasil_decision_forests/utils/adaptive_work.h"

#include <algorithm>
#include <limits>

#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {}  // namespace

AdaptativeWork::AdaptativeWork(const int num_tasks, const double total_budget,
                               const double warming_up_budget,
                               const double min_factor)
    : num_tasks_(num_tasks),
      total_budget_(total_budget),
      warming_up_budget_(warming_up_budget),
      min_factor_(min_factor) {
  CHECK_GT(total_budget, 0.0);
  CHECK_GE(warming_up_budget, 0.0);
}

void AdaptativeWork::ReportTaskDone(const double approximation_factor,
                                    const double consumed_budget) {
  utils::concurrency::MutexLock lock(&mu_);
  CHECK_GT(approximation_factor, 0.0);
  CHECK_LE(approximation_factor, 1.0);
  consumed_budget_ += consumed_budget;
  num_ran_tasks_++;
  sum_consumed_budget_div_approximation_factor_ +=
      consumed_budget / approximation_factor;
}

double AdaptativeWork::OptimalApproximationFactor() const {
  utils::concurrency::MutexLock lock(&mu_);
  if (consumed_budget_ < warming_up_budget_ || num_ran_tasks_ == 0) {
    return 1.;
  }
  const double forecasted_total_budget_without_approx =
      sum_consumed_budget_div_approximation_factor_ * num_tasks_ /
      num_ran_tasks_;
  const double approximation_factor =
      std::max(std::numeric_limits<double>::epsilon(),
               total_budget_ / forecasted_total_budget_without_approx);
  return utils::clamp(approximation_factor, min_factor_, 1.);
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests
