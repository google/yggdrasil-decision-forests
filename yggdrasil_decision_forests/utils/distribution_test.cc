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

#include <math.h>

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_replace.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace utils {
namespace {

using test::EqualsProto;

TEST(Distribution, NormalDistribution) {
  NormalDistributionDouble dist;
  EXPECT_EQ(dist.Mean(), 0);
  EXPECT_EQ(dist.Std(), 0);
  EXPECT_EQ(dist.NumObservations(), 0);
  dist.Add(5.f);
  EXPECT_EQ(dist.Mean(), 5);
  EXPECT_EQ(dist.Std(), 0);
  EXPECT_EQ(dist.NumObservations(), 1);
  dist.Add(10.);
  EXPECT_EQ(dist.Mean(), 7.5);
  EXPECT_EQ(dist.Std(),
            sqrt(((5 - 7.5) * (5 - 7.5) + (10 - 7.5) * (10 - 7.5)) / 2));
  EXPECT_EQ(dist.NumObservations(), 2);

  proto::NormalDistributionDouble proto;
  dist.Save(&proto);

  NormalDistributionDouble dist2;
  dist2.Load(proto);

  EXPECT_EQ(dist.Mean(), dist2.Mean());
  EXPECT_EQ(dist.Std(), dist2.Std());
  EXPECT_EQ(dist.NumObservations(), dist2.NumObservations());
}

TEST(Distribution, IntegerDistributionInt64) {
  IntegerDistributionInt64 dist;
  dist.SetNumClasses(4);
  dist.Add(1, 2);
  dist.Add(2, 3);

  EXPECT_EQ(dist.count(1), 2);
  EXPECT_EQ(dist.count(2), 3);
  EXPECT_EQ(dist.NumClasses(), 4);

  // entropy(c(2,3)) = 0.6730117 in R.
  EXPECT_NEAR(dist.Entropy(), 0.6730117, 0.0001);

  proto::IntegerDistributionInt64 proto;
  dist.Save(&proto);

  IntegerDistributionInt64 dist2;
  dist2.Load(proto);

  EXPECT_EQ(dist.NumClasses(), dist2.NumClasses());
  EXPECT_EQ(dist.count(1), dist2.count(1));
  EXPECT_EQ(dist.count(2), dist2.count(2));
}

TEST(Distribution, IntegerDistributionDouble) {
  IntegerDistributionDouble dist;
  dist.SetNumClasses(4);
  dist.Add(1, 2.5);
  dist.Add(2, 3.5);

  EXPECT_EQ(dist.count(1), 2.5);
  EXPECT_EQ(dist.count(2), 3.5);
  EXPECT_EQ(dist.NumClasses(), 4);

  // entropy(c(2.5,3.5)) = 0.6791933 in R.
  EXPECT_NEAR(dist.Entropy(), 0.6791933, 0.0001);

  proto::IntegerDistributionDouble proto;
  dist.Save(&proto);

  IntegerDistributionDouble dist2;
  dist2.Load(proto);

  EXPECT_EQ(dist.NumClasses(), dist2.NumClasses());
  EXPECT_EQ(dist.count(1), dist2.count(1));
  EXPECT_EQ(dist.count(2), dist2.count(2));
}

TEST(Distribution, IntegerDistributionInt64Normalization) {
  IntegerDistributionInt64 dist;
  dist.SetNumClasses(4);
  dist.Add(1, 2);
  dist.Add(2, 3);

  dist.Normalize();

  EXPECT_EQ(dist.count(1), 2 / 5);
  EXPECT_EQ(dist.count(2), 3 / 5);
}

TEST(Distribution, IntegerDistributionDoubleNormalization) {
  IntegerDistributionDouble dist;
  dist.SetNumClasses(4);
  dist.Add(1, 2.5);
  dist.Add(2, 3.5);

  dist.Normalize();

  EXPECT_EQ(dist.count(1), 2.5 / 6.);
  EXPECT_EQ(dist.count(2), 3.5 / 6.);
}

TEST(Distribution, IntegerDistributionDoubleNormalizeAndClampPositive) {
  IntegerDistributionDouble dist;
  dist.SetNumClasses(4);
  dist.Add(1, 2.5);
  dist.Add(2, 3.5);

  dist.NormalizeAndClampPositive();

  EXPECT_EQ(dist.count(1), 2.5 / 6.);
  EXPECT_EQ(dist.count(2), 3.5 / 6.);
}

TEST(Distribution, IntegerDistributionDoubleNormalizedAddition) {
  IntegerDistributionDouble counter1;
  counter1.SetNumClasses(4);
  counter1.Add(1, 2.5);
  counter1.Add(2, 3.5);

  IntegerDistributionDouble counter2;
  counter2.SetNumClasses(4);
  counter2.Add(2, 5.5);
  counter2.Add(3, 7.5);

  IntegerDistributionDouble dist;
  dist.SetNumClasses(4);
  dist.Add(counter1);
  dist.Add(counter2);
  EXPECT_EQ(dist.count(1), 2.5);
  EXPECT_EQ(dist.count(2), 9.);
  EXPECT_EQ(dist.count(3), 7.5);

  IntegerDistributionDouble dist2;
  dist2.SetNumClasses(4);
  dist2.AddNormalized(counter1);
  dist2.AddNormalized(counter2);
  EXPECT_EQ(dist2.count(1), 2.5 / 6);
  EXPECT_EQ(dist2.count(2), 3.5 / 6 + 5.5 / 13);
  EXPECT_EQ(dist2.count(3), 7.5 / 13);
}

TEST(Distribution, IntegerDistributionIntMerge) {
  IntegerDistributionInt64 counter1;
  counter1.SetNumClasses(4);
  counter1.Add(1);
  counter1.Add(2);

  IntegerDistributionInt64 counter2;
  counter2.SetNumClasses(4);
  counter2.Add(2);
  counter2.Add(3);

  counter1.Add(counter2);
  EXPECT_EQ(counter1.count(0), 0);
  EXPECT_EQ(counter1.count(1), 1);
  EXPECT_EQ(counter1.count(2), 2);
  EXPECT_EQ(counter1.count(3), 1);

  counter1.Sub(counter2);
  EXPECT_EQ(counter1.count(0), 0);
  EXPECT_EQ(counter1.count(1), 1);
  EXPECT_EQ(counter1.count(2), 1);
  EXPECT_EQ(counter1.count(3), 0);
}

TEST(Distribution, AddNormalizedToIntegerDistributionProto) {
  const proto::IntegerDistributionDouble src =
      PARSE_TEST_PROTO(R"pb(
        counts: 1.5 counts: 3.75 counts: 2.25 sum: 7.5
      )pb");
  proto::IntegerDistributionDouble dst =
      PARSE_TEST_PROTO(R"pb(
        counts: 0.0 counts: 0.5 counts: 0.5 sum: 1.0
      )pb");
  AddNormalizedToIntegerDistributionProto(src, 0.2, &dst);
  EXPECT_NEAR(dst.counts(0), 0.04, 0.00001);
  EXPECT_NEAR(dst.counts(1), 0.6, 0.00001);
  EXPECT_NEAR(dst.counts(2), 0.56, 0.00001);
  EXPECT_NEAR(dst.sum(), 1.2, 0.00001);
}

TEST(Distribution, SubNormalizedProto) {
  IntegerDistributionDouble counter;
  counter.SetNumClasses(2);
  counter.Add(0, 1);
  proto::IntegerDistributionDouble src =
      PARSE_TEST_PROTO(R"pb(
        counts: 0 counts: 2 sum: 2
      )pb");
  counter.SubNormalizedProto(src);
  EXPECT_NEAR(counter.count(0), 1, 0.00001);
  EXPECT_NEAR(counter.count(1), -1, 0.00001);
}

TEST(Distribution, IntegerDistributionInt64NormalizedAddition) {
  IntegerDistributionInt64 counter1;
  counter1.SetNumClasses(4);
  counter1.Add(1, 2);
  counter1.Add(2, 3);

  IntegerDistributionInt64 counter2;
  counter2.SetNumClasses(4);
  counter2.Add(2, 5);
  counter2.Add(3, 7);

  IntegerDistributionInt64 dist;
  dist.SetNumClasses(4);
  dist.Add(counter1);
  dist.Add(counter2);
  EXPECT_EQ(dist.count(1), 2);
  EXPECT_EQ(dist.count(2), 8);
  EXPECT_EQ(dist.count(3), 7);

  IntegerDistributionInt64 dist2;
  dist2.SetNumClasses(4);
  dist2.AddNormalized(counter1);
  dist2.AddNormalized(counter2);
  EXPECT_EQ(dist2.count(1), 2 / 5);
  EXPECT_EQ(dist2.count(2), 3 / 5 + 5 / (5 + 7));
  EXPECT_EQ(dist2.count(3), 7 / (5 + 7));
}

TEST(Distribution, BinaryToIntegerConfusionMatrixInt64) {
  BinaryToIntegerConfusionMatrixInt64 conf;
  conf.SetNumClassesIntDim(4);

  conf.Add(false, 1);
  conf.Add(true, 1);
  conf.Add(true, 2);

  // entropy(c(2,1)) in R.
  EXPECT_NEAR(conf.InitEntropy(), 0.636514, 0.0001);

  // entropy(c(1)) / 3 + entropy(c(1,1)) * 2 / 3 in R.
  EXPECT_NEAR(conf.FinalEntropy(), 0.462098, 0.0001);
}

TEST(Distribution, BinaryToIntegerConfusionMatrixDouble) {
  BinaryToIntegerConfusionMatrixDouble conf;
  conf.SetNumClassesIntDim(4);

  conf.Add(false, 1.);
  conf.Add(true, 1.);
  conf.Add(true, 2.);

  // entropy(c(2,1)) in R.
  EXPECT_NEAR(conf.InitEntropy(), 0.636514, 0.0001);

  // entropy(c(1)) / 3 + entropy(c(1,1)) * 2 / 3 in R.
  EXPECT_NEAR(conf.FinalEntropy(), 0.462098, 0.0001);
}

TEST(Distribution, IntegersConfusionMatrix_AppendTextReport) {
  IntegersConfusionMatrixDouble confusion;
  confusion.SetSize(4, 4);
  dataset::proto::Column column;
  column.mutable_categorical()->set_is_already_integerized(false);
  column.mutable_categorical()->set_number_of_unique_values(4);
  auto& items = *column.mutable_categorical()->mutable_items();
  items["a"].set_index(0);
  items["bb"].set_index(1);
  items["ccc"].set_index(2);
  items["dddd"].set_index(3);
  for (int col = 0; col < confusion.ncol(); col++) {
    for (int row = 0; row < confusion.nrow(); row++) {
      confusion.Add(row, col, col * 4 + std::pow(8, row));
    }
  }
  std::string representation;
  confusion.AppendTextReport(column, &representation);
  EXPECT_EQ(representation, R"(truth\prediction
        a   bb  ccc  dddd
   a    1    5    9    13
  bb    8   12   16    20
 ccc   64   68   72    76
dddd  512  516  520   524
Total: 2436
)");
}

TEST(Distribution, IntegersConfusionMatrix_AppendTextReportAlreadyIntegerized) {
  IntegersConfusionMatrixDouble confusion;
  confusion.SetSize(4, 4);
  dataset::proto::Column column;
  column.mutable_categorical()->set_is_already_integerized(true);
  column.mutable_categorical()->set_number_of_unique_values(4);
  for (int col = 0; col < confusion.ncol(); col++) {
    for (int row = 0; row < confusion.nrow(); row++) {
      confusion.Add(row, col, col * 4 + std::pow(8, row));
    }
  }
  std::string representation;
  confusion.AppendTextReport(column, &representation);
  EXPECT_EQ(representation, R"(truth\prediction
     0    1    2    3
0    1    5    9   13
1    8   12   16   20
2   64   68   72   76
3  512  516  520  524
Total: 2436
)");
}

TEST(Distribution, ConfusionMatrixProtoSumColumns) {
  proto::IntegersConfusionMatrixDouble confusion;
  InitializeConfusionMatrixProto(3, 3, &confusion);
  AddToConfusionMatrixProto(0, 0, 1, &confusion);
  AddToConfusionMatrixProto(0, 1, 1, &confusion);
  AddToConfusionMatrixProto(1, 1, 1, &confusion);
  AddToConfusionMatrixProto(1, 0, 1, &confusion);
  AddToConfusionMatrixProto(2, 0, 1, &confusion);
  EXPECT_EQ(ConfusionMatrixProtoSumColumns(confusion, 0), 2);
  EXPECT_EQ(ConfusionMatrixProtoSumColumns(confusion, 1), 2);
  EXPECT_EQ(ConfusionMatrixProtoSumColumns(confusion, 2), 1);
}

TEST(Distribution, IntegersConfusionMatrix_AppendHtmlReport) {
  IntegersConfusionMatrixDouble confusion;
  confusion.SetSize(2, 2);
  dataset::proto::Column column;
  column.mutable_categorical()->set_is_already_integerized(false);
  column.mutable_categorical()->set_number_of_unique_values(2);
  auto& items = *column.mutable_categorical()->mutable_items();
  items["a"].set_index(0);
  items["bb"].set_index(1);
  for (int col = 0; col < confusion.ncol(); col++) {
    for (int row = 0; row < confusion.nrow(); row++) {
      confusion.Add(row, col, col * 4 + std::pow(8, row));
    }
  }
  std::string representation;
  confusion.AppendHtmlReport(column, &representation);
  EXPECT_EQ(representation,
            absl::StrReplaceAll(R"(<table class="confusion_matrix">
<tr>
<th align="left">Truth\Prediction</th>
<th align="right">a</th>
<th align="right">bb</th>
</tr>
<tr>
<th align="right">a</th>
<td align="right" style="background-color:hsl(100, 50%, 98%);">1</td>
<td align="right" style="background-color:hsl(100, 50%, 90%);">5</td>
</tr>
<tr>
<th align="right">bb</th>
<td align="right" style="background-color:hsl(100, 50%, 84%);">8</td>
<td align="right" style="background-color:hsl(100, 50%, 76%);">12</td>
</tr>
</table>
<p>Total: 26</p>
)",
                                {{"\n", ""}}));
}

TEST(Distribution, BinaryToNormalDistributionDouble) {
  BinaryToNormalDistributionDouble conf;

  conf.Add(false, 1);
  conf.Add(false, 2);
  conf.Add(true, 1);
  conf.Add(true, 3);

  EXPECT_NEAR(conf.NumObservations(), 4, 0.0001);

  // var(c(1,2)) / 2 +  var(c(1,3)) / 2 = 0.625
  // with var the variance.
  EXPECT_NEAR(conf.FinalVariance(), 0.625, 0.0001);
}

TEST(BinaryDistributionEntropy, Base) {
  const double epsilon = 0.0001;

  EXPECT_NEAR(BinaryDistributionEntropyF(0.0), 0., epsilon);
  EXPECT_NEAR(BinaryDistributionEntropyF(0.5), -2.0 * 0.5 * log(0.5), epsilon);
  EXPECT_NEAR(BinaryDistributionEntropyF(1.0), 0., epsilon);

  EXPECT_NEAR(BinaryDistributionEntropyF(0.001), 0.00790, epsilon);
  EXPECT_NEAR(BinaryDistributionEntropyF(0.999), 0.00790, epsilon);

  EXPECT_NEAR(BinaryDistributionEntropyF(-0.001), 0, epsilon);
  EXPECT_NEAR(BinaryDistributionEntropyF(1.001), 0, epsilon);

  EXPECT_NEAR(BinaryDistributionEntropyF(-10), 0, epsilon);
  EXPECT_NEAR(BinaryDistributionEntropyF(10), 0, epsilon);

  EXPECT_NEAR(BinaryDistributionEntropyF(0. / 0.), 0, epsilon);
  EXPECT_NEAR(BinaryDistributionEntropyF(-0. / 0.), 0, epsilon);
  EXPECT_NEAR(BinaryDistributionEntropyF(1. / 0.), 0, epsilon);
  EXPECT_NEAR(BinaryDistributionEntropyF(-1. / 0.), 0, epsilon);
}

TEST(Distribution, IntegersConfusionMatrix_AddToConfusionMatrixProto) {
  const proto::IntegersConfusionMatrixDouble src = PARSE_TEST_PROTO(
      R"pb(
        nrow: 2 ncol: 2 sum: 10 counts: 1 counts: 2 counts: 3 counts: 4
      )pb");
  proto::IntegersConfusionMatrixDouble dst = PARSE_TEST_PROTO(
      R"pb(
        nrow: 2 ncol: 2 sum: 100 counts: 10 counts: 20 counts: 30 counts: 40
      )pb");
  AddToConfusionMatrixProto(src, &dst);
  proto::IntegersConfusionMatrixDouble expected_dst = PARSE_TEST_PROTO(
      R"pb(
        nrow: 2 ncol: 2 sum: 110 counts: 11 counts: 22 counts: 33 counts: 44
      )pb");
  EXPECT_THAT(dst, EqualsProto(expected_dst));
}

TEST(Distribution, IntegerDistributionFloatTopClass) {
  proto::IntegerDistributionFloat proto;
  proto.add_counts(1);
  proto.add_counts(3);
  proto.add_counts(2);
  proto.set_sum(1 + 2 + 3);
  EXPECT_EQ(TopClass(proto), 1);
}

TEST(Distribution, IntegerDistributionDoubleTopClass) {
  proto::IntegerDistributionDouble proto;
  proto.add_counts(1);
  proto.add_counts(3);
  proto.add_counts(2);
  proto.set_sum(1 + 2 + 3);
  EXPECT_EQ(TopClass(proto), 1);
}

}  // namespace
}  // namespace utils
}  // namespace yggdrasil_decision_forests
