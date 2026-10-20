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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/quantile_sketch.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/training.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/testing_macros.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

// Exact weight of the values < v and <= v.
void ExactRanks(const std::vector<float>& values,
                const std::vector<float>& weights, const float v, double* lt,
                double* le) {
  *lt = 0;
  *le = 0;
  for (size_t i = 0; i < values.size(); i++) {
    const double w = weights.empty() ? 1. : weights[i];
    if (values[i] < v) *lt += w;
    if (values[i] <= v) *le += w;
  }
}

// Checks that the bounds of the summary are valid for the given values.
void CheckValidBounds(const QuantileSummary& summary,
                      const std::vector<float>& values,
                      const std::vector<float>& weights) {
  double total = 0;
  for (size_t i = 0; i < values.size(); i++) {
    total += weights.empty() ? 1. : weights[i];
  }
  EXPECT_NEAR(summary.total_weight(), total, 1e-6 * total);
  for (const auto& entry : summary.entries()) {
    double lt, le;
    ExactRanks(values, weights, entry.value, &lt, &le);
    const double tol = 1e-6 * (1 + total);
    EXPECT_LE(entry.min_rank, lt + tol);
    EXPECT_GE(entry.max_rank, le - tol);
    EXPECT_LE(entry.weight, le - lt + tol);
  }
  EXPECT_TRUE(std::is_sorted(
      summary.entries().begin(), summary.entries().end(),
      [](const auto& a, const auto& b) { return a.value < b.value; }));
}

// Checks that the rank error of the quantile queries is certified.
void CheckQueries(const QuantileSummary& summary,
                  const std::vector<float>& values,
                  const std::vector<float>& weights) {
  for (int i = 0; i <= 20; i++) {
    const double q = i / 20.;
    const double target = q * summary.total_weight();
    const auto& entry = summary.Query(q);
    const int entry_idx = &entry - summary.entries().data();
    const double certified = summary.WorstCaseRankError(entry_idx, target);
    double lt, le;
    ExactRanks(values, weights, entry.value, &lt, &le);
    const double true_error = std::max({0., lt - target, target - le});
    EXPECT_LE(true_error, certified + 1e-6 * (1 + summary.total_weight()));
  }
}

TEST(QuantileSummary, ExactFromValues) {
  ASSERT_OK_AND_ASSIGN(const auto summary,
                       QuantileSummary::FromValues({3, 1, 2, 2, NAN}, {}));
  ASSERT_EQ(summary.entries().size(), 3);
  EXPECT_EQ(summary.total_weight(), 4);
  EXPECT_EQ(summary.entries()[1].value, 2);
  EXPECT_EQ(summary.entries()[1].min_rank, 1);
  EXPECT_EQ(summary.entries()[1].max_rank, 3);
  EXPECT_EQ(summary.entries()[1].weight, 2);
  EXPECT_EQ(summary.Query(0.5).value, 2);
}

TEST(QuantileSummary, Empty) {
  ASSERT_OK_AND_ASSIGN(const auto summary, QuantileSummary::FromValues({}, {}));
  EXPECT_TRUE(summary.empty());
  EXPECT_THAT(ExtractBinBoundaries(summary, 10), IsEmpty());
  const auto merged = QuantileSummary::Merge(summary, summary);
  EXPECT_TRUE(merged.empty());
}

TEST(QuantileSummary, ConstantFeature) {
  std::vector<float> values(1000, 5.f);
  ASSERT_OK_AND_ASSIGN(auto summary, QuantileSummary::FromValues(values, {}));
  // A single (heavy) value is isolated in its own bin.
  EXPECT_THAT(ExtractBinBoundaries(summary, 10),
              ElementsAre(5.f, std::nextafter(5.f, 1000.f)));
}

TEST(QuantileSummary, ProtoRoundTrip) {
  ASSERT_OK_AND_ASSIGN(const auto summary,
                       QuantileSummary::FromValues({1, 2, 3}, {1, 2, 3}));
  proto::QuantileSummary message;
  summary.ToProto(&message);
  const QuantileSummary restored(message);
  ASSERT_EQ(restored.entries().size(), 3);
  EXPECT_EQ(restored.total_weight(), 6);
  EXPECT_EQ(restored.entries()[2].min_rank, 3);
}

class QuantileSummaryRandom : public ::testing::TestWithParam<bool> {};

// Shards a random dataset, summarizes each shard, prunes and merges the
// summaries, and checks the validity of the bounds and queries.
TEST_P(QuantileSummaryRandom, MergeAndPrune) {
  const bool weighted = GetParam();
  utils::RandomEngine random(1234);
  std::normal_distribution<float> normal;
  std::uniform_real_distribution<float> unif(0.1f, 2.f);
  for (int trial = 0; trial < 5; trial++) {
    const int num_values = 3000 + trial * 1000;
    std::vector<float> values(num_values);
    std::vector<float> weights;
    for (auto& value : values) {
      value = normal(random);
      // 30% of zeros (point mass) and some duplicates.
      if (unif(random) < 0.67f) value = 0.f;
      if (unif(random) < 0.3f) value = std::round(value * 4.f) / 4.f;
    }
    if (weighted) {
      weights.resize(num_values);
      for (auto& w : weights) w = unif(random);
    }

    const int num_shards = 7;
    const int max_entries = 64;
    QuantileSummary merged;
    for (int shard = 0; shard < num_shards; shard++) {
      const int begin = shard * num_values / num_shards;
      const int end = (shard + 1) * num_values / num_shards;
      std::vector<float> shard_values(values.begin() + begin,
                                      values.begin() + end);
      std::vector<float> shard_weights;
      if (weighted) {
        shard_weights.assign(weights.begin() + begin, weights.begin() + end);
      }
      ASSERT_OK_AND_ASSIGN(auto summary, QuantileSummary::FromValues(
                                             shard_values, shard_weights));
      CheckValidBounds(summary, shard_values, shard_weights);
      summary.Prune(max_entries);
      EXPECT_LE(summary.entries().size(), max_entries);
      CheckValidBounds(summary, shard_values, shard_weights);

      // Proto round trip (as done between the workers and the manager).
      proto::QuantileSummary message;
      summary.ToProto(&message);
      merged = QuantileSummary::Merge(merged, QuantileSummary(message));
      merged.Prune(max_entries);
    }
    CheckValidBounds(merged, values, weights);
    CheckQueries(merged, values, weights);

    // The median certified error is small.
    const double target = 0.5 * merged.total_weight();
    const auto& entry = merged.Query(0.5);
    EXPECT_LE(
        merged.WorstCaseRankError(&entry - merged.entries().data(), target),
        0.15 * merged.total_weight());

    // The point mass at zero is isolated.
    const auto boundaries = ExtractBinBoundaries(merged, 16);
    EXPECT_TRUE(std::is_sorted(boundaries.begin(), boundaries.end()));
    EXPECT_NE(std::find(boundaries.begin(), boundaries.end(), 0.f),
              boundaries.end());
    EXPECT_NE(std::find(boundaries.begin(), boundaries.end(),
                        std::nextafter(0.f, 1.f)),
              boundaries.end());
  }
}

INSTANTIATE_TEST_SUITE_P(Weighted, QuantileSummaryRandom,
                         ::testing::Values(false, true));

TEST(BuildStreamingSummary, SamplingAndChunks) {
  utils::RandomEngine random(1);
  std::uniform_real_distribution<float> unif;
  std::vector<float> values(50000);
  for (auto& value : values) value = unif(random);

  StreamingSummaryOptions options;
  options.max_entries = 128;
  options.max_examples = 10000;
  options.chunk_size = 1000;
  ASSERT_OK_AND_ASSIGN(const auto summary,
                       BuildStreamingSummary(values, {}, options, &random));
  // Only the sampled examples are summarized.
  EXPECT_NEAR(summary.total_weight(), 10000, 1e-6);
  EXPECT_LE(summary.entries().size(), 128);
  // Uniform data: the median is close to 0.5.
  EXPECT_NEAR(summary.Query(0.5).value, 0.5, 0.05);

  // Without sampling, the bounds are certified for the full dataset.
  options.max_examples = -1;
  ASSERT_OK_AND_ASSIGN(const auto full_summary,
                       BuildStreamingSummary(values, {}, options, &random));
  EXPECT_NEAR(full_summary.total_weight(), 50000, 1e-6);
  CheckQueries(full_summary, values, {});
}

TEST(BuildStreamingSummary, Deadline) {
  std::vector<float> values(100000);
  for (int i = 0; i < values.size(); i++) values[i] = i;
  StreamingSummaryOptions options;
  options.chunk_size = 1000;
  options.deadline = absl::Now() - absl::Seconds(1);
  utils::RandomEngine random(1);
  ASSERT_OK_AND_ASSIGN(const auto summary,
                       BuildStreamingSummary(values, {}, options, &random));
  // Only one chunk is processed.
  EXPECT_EQ(summary.total_weight(), 1000);
}

}  // namespace
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
