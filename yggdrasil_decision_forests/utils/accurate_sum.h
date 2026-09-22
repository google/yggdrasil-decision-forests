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

// Compensated summation algorithms for accurate sums.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_ACCURATE_SUM_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_ACCURATE_SUM_H_

#include <cmath>

namespace yggdrasil_decision_forests {
namespace utils {

// Kahan compensated summation.
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

// Neumaier compensated summation.
//
// Described in: Neumaier, A. (1974). "Rundungsfehleranalyse einiger Verfahren
// zur Summation endlicher Summen". Zeitschrift für Angewandte Mathematik und
// Mechanik, 54(1), 39-51. https://doi.org/10.1002/zamm.19740540106
//
// A bit slower but more accurate variant of KahanSum
class NeumaierSum {
 public:
  NeumaierSum() {}

  NeumaierSum(const double sum, const double compensation)
      : sum_(sum), compensation_(compensation) {}

  void Add(const double value) {
    const double new_sum = sum_ + value;
    if (std::abs(sum_) >= std::abs(value)) {
      compensation_ += (sum_ - new_sum) + value;
    } else {
      compensation_ += (value - new_sum) + sum_;
    }
    sum_ = new_sum;
  }

  double Value() const { return sum_ + compensation_; }

  double Compensation() const { return compensation_; }

  void Reset() {
    sum_ = 0.;
    compensation_ = 0.;
  }

 private:
  double sum_ = 0.;
  double compensation_ = 0.;
};

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_ACCURATE_SUM_H_
