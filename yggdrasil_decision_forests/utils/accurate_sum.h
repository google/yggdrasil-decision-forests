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

// Implementation of the Kahan summation algorithm for accurate sums.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_ACCURATE_SUM_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_ACCURATE_SUM_H_

namespace yggdrasil_decision_forests {
namespace utils {

class AccurateSum {
 public:
  AccurateSum() {}

  AccurateSum(const double sum, const double error_sum)
      : sum_(sum), error_sum_(error_sum) {}

  void Add(const double value) {
    error_sum_ += value;
    const auto new_sum = sum_ + error_sum_;
    error_sum_ += sum_ - new_sum;
    sum_ = new_sum;
  }

  double Sum() const { return sum_; }

  double ErrorSum() const { return error_sum_; }

 private:
  double sum_ = 0.;
  double error_sum_ = 0.;
};

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_ACCURATE_SUM_H_
