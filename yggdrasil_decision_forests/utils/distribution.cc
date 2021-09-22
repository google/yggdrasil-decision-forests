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

#include "yggdrasil_decision_forests/utils/distribution.h"

#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace utils {

void NormalDistributionDouble::Save(
    proto::NormalDistributionDouble* proto) const {
  proto->set_sum(sum_);
  proto->set_sum_squares(sum_squares_);
  proto->set_count(count_);
}

void NormalDistributionDouble::Load(
    const proto::NormalDistributionDouble& proto) {
  sum_ = proto.sum();
  sum_squares_ = proto.sum_squares();
  count_ = proto.count();
}

void InitializeConfusionMatrixProto(
    int32_t nr, int32_t nc, proto::IntegersConfusionMatrixDouble* confusion) {
  DCHECK_GE(nr, 0);
  DCHECK_GE(nc, 0);
  confusion->set_nrow(nr);
  confusion->set_ncol(nc);
  confusion->set_sum(0);
  confusion->mutable_counts()->Resize(nr * nc, 0.);
}

void AddToConfusionMatrixProto(
    int32_t r, int32_t c, double value,
    proto::IntegersConfusionMatrixDouble* confusion) {
  DCHECK_GE(r, 0);
  DCHECK_GE(c, 0);
  DCHECK_LT(r, confusion->nrow());
  DCHECK_LT(c, confusion->ncol());
  const int index =
      IntegersConfusionMatrixDouble::Index(r, c, confusion->nrow());
  confusion->set_counts(index, confusion->counts(index) + value);
  confusion->set_sum(confusion->sum() + value);
}

void AddToConfusionMatrixProto(const proto::IntegersConfusionMatrixDouble& src,
                               proto::IntegersConfusionMatrixDouble* dst) {
  DCHECK_EQ(src.nrow(), dst->nrow());
  DCHECK_EQ(src.ncol(), dst->ncol());
  DCHECK_EQ(src.counts_size(), dst->counts_size());
  for (int i = 0; i < src.counts_size(); i++) {
    dst->set_counts(i, src.counts(i) + dst->counts(i));
  }
  dst->set_sum(dst->sum() + src.sum());
}

double ConfusionMatrixProtoTrace(
    const proto::IntegersConfusionMatrixDouble& confusion) {
  CHECK_EQ(confusion.nrow(), confusion.ncol());
  double sum = 0;
  for (int row = 0; row < confusion.ncol(); row++) {
    const int index =
        IntegersConfusionMatrixDouble::Index(row, row, confusion.nrow());
    sum += confusion.counts(index);
  }
  return sum;
}

double ConfusionMatrixProtoSumColumns(
    const proto::IntegersConfusionMatrixDouble& confusion, int32_t row) {
  double sum = 0;
  for (int col = 0; col < confusion.ncol(); col++) {
    sum += confusion.counts(
        IntegersConfusionMatrixDouble::Index(row, col, confusion.nrow()));
  }
  return sum;
}

int TopClass(const proto::IntegerDistributionFloat& dist) {
  int top_index = 0;
  float top_value = 0.;
  for (int i = 0; i < dist.counts_size(); i++) {
    if (dist.counts(i) > top_value) {
      top_value = dist.counts(i);
      top_index = i;
    }
  }
  return top_index;
}

int TopClass(const proto::IntegerDistributionDouble& dist) {
  int top_index = 0;
  float top_value = 0.;
  for (int i = 0; i < dist.counts_size(); i++) {
    if (dist.counts(i) > top_value) {
      top_value = dist.counts(i);
      top_index = i;
    }
  }
  return top_index;
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests
