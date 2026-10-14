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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/histogram_bnb.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/utils.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/quantile_sketch.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/splitter.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/training.pb.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace {

using ::testing::ElementsAre;

// A synthetic dataset with numerical features and gradient/hessian labels.
struct Data {
  std::vector<std::vector<float>> features;  // features[f][example]
  std::vector<float> gradients;
  std::vector<float> hessians;
  std::vector<float> weights;  // Empty if not weighted.
  size_t num_examples() const { return gradients.size(); }

  LabelData Labels(const RatioType ratio_type) const {
    return {gradients, hessians, weights, ratio_type};
  }
};

enum class FeatureKind { kGaussian, kDuplicates, kHeavyTail, kPointMass };

Data GenerateData(const int num_examples, const bool weighted,
                  const bool zero_hessians, utils::RandomEngine* random) {
  std::normal_distribution<float> normal;
  std::uniform_real_distribution<float> unif;
  std::cauchy_distribution<float> cauchy;
  Data data;
  const std::vector<FeatureKind> kinds = {
      FeatureKind::kGaussian, FeatureKind::kDuplicates, FeatureKind::kHeavyTail,
      FeatureKind::kPointMass, FeatureKind::kGaussian};
  data.features.resize(kinds.size());
  for (int f = 0; f < kinds.size(); f++) {
    auto& values = data.features[f];
    values.resize(num_examples);
    for (auto& value : values) {
      switch (kinds[f]) {
        case FeatureKind::kGaussian:
          value = normal(*random);
          break;
        case FeatureKind::kDuplicates:
          value = std::round(normal(*random) * 3.f);
          break;
        case FeatureKind::kHeavyTail:
          value = cauchy(*random);
          break;
        case FeatureKind::kPointMass:
          value = unif(*random) < 0.7f ? 0.f : normal(*random);
          break;
      }
    }
  }
  data.gradients.resize(num_examples);
  data.hessians.resize(num_examples);
  for (int i = 0; i < num_examples; i++) {
    // Signal on features 0 and 3, with noise.
    const float signal = (data.features[0][i] > 0.3f ? 1.f : -0.5f) +
                         0.5f * (data.features[3][i] != 0.f);
    data.gradients[i] = signal + 0.8f * normal(*random);
    data.hessians[i] = 0.05f + unif(*random);
    if (zero_hessians && unif(*random) < 0.05f) {
      data.hessians[i] = 0.f;
    }
  }
  if (weighted) {
    data.weights.resize(num_examples);
    for (auto& w : data.weights) w = 0.2f + 2.f * unif(*random);
  }
  return data;
}

struct BruteForceResult {
  bool found = false;
  double score = 0;
  int feature = -1;
  float threshold = 0;
};

// Exhaustive search of the best numerical split.
BruteForceResult BruteForceSplit(
    const Data& data, const std::vector<UnsignedExampleIdx>& examples,
    const SplitCriterion& criterion, const int64_t min_examples) {
  const auto labels = data.Labels(criterion.ratio_type());
  BinStats parent;
  for (const auto i : examples) labels.AddExample(i, 0.f, &parent);
  BruteForceResult result;
  const double min_score = criterion.MinimumScore(parent);
  for (int f = 0; f < data.features.size(); f++) {
    const auto& values = data.features[f];
    std::vector<UnsignedExampleIdx> sorted = examples;
    std::stable_sort(sorted.begin(), sorted.end(),
                     [&](auto a, auto b) { return values[a] < values[b]; });
    BinStats prefix;
    for (size_t j = 0; j + 1 < sorted.size(); j++) {
      labels.AddExample(sorted[j], values[sorted[j]], &prefix);
      if (values[sorted[j]] == values[sorted[j + 1]]) continue;
      if (prefix.count < min_examples ||
          static_cast<int64_t>(sorted.size()) - prefix.count < min_examples) {
        continue;
      }
      BinStats pos = parent;
      pos.SubSums(prefix);
      if (!criterion.IsValidSplit(prefix, pos)) continue;
      const double score = criterion.Score(parent, prefix, pos);
      if (!(score > min_score)) continue;
      if (!result.found || score > result.score) {
        result.found = true;
        result.score = score;
        result.feature = f;
        result.threshold = decision_tree::MidThreshold(values[sorted[j]],
                                                       values[sorted[j + 1]]);
      }
    }
  }
  return result;
}

// Exact score of the split "feature >= threshold".
double ScoreOfSplit(const Data& data,
                    const std::vector<UnsignedExampleIdx>& examples,
                    const SplitCriterion& criterion, const int feature,
                    const float threshold) {
  const auto labels = data.Labels(criterion.ratio_type());
  BinStats parent, neg;
  for (const auto i : examples) {
    labels.AddExample(i, 0.f, &parent);
    if (data.features[feature][i] < threshold) {
      labels.AddExample(i, 0.f, &neg);
    }
  }
  BinStats pos = parent;
  pos.SubSums(neg);
  return criterion.Score(parent, neg, pos);
}

struct BnbOptions {
  int num_workers = 3;
  int num_bins = 16;
  int sketch_size = 64;
  int num_sub_bins = 8;
  int64_t exact_scan_threshold = 32;
  int max_refinement_rounds = -1;
  int64_t min_examples = 1;
};

struct BnbResult {
  bool found = false;
  Split split;
  int rounds = 0;
  int64_t num_refined_examples = 0;
};

// Simulates the distributed histogram branch-and-bound for a single node
// containing "examples". Each worker owns a contiguous range of examples.
BnbResult RunBnb(const Data& data,
                 const std::vector<UnsignedExampleIdx>& examples,
                 const SplitCriterion& criterion, const BnbOptions& options,
                 const bool use_skip_lists = true) {
  const auto labels = data.Labels(criterion.ratio_type());
  const int num_features = data.features.size();
  const size_t n = data.num_examples();

  // Example -> node (only examples in "examples" are in node 0).
  ExampleToNodeMap example_to_node(n, kClosedNode);
  for (const auto i : examples) example_to_node[i] = 0;

  // Examples of each worker.
  std::vector<std::vector<UnsignedExampleIdx>> worker_examples(
      options.num_workers);
  for (const auto i : examples) {
    worker_examples[i * options.num_workers / n].push_back(i);
  }

  // Parent statistics.
  BinStats parent;
  for (int w = 0; w < options.num_workers; w++) {
    BinStats partial;
    for (const auto i : worker_examples[w]) {
      labels.AddExample(i, 0.f, &partial);
    }
    parent.Add(partial);
  }

  NodeSplitSearch search(&criterion, parent, options.min_examples);
  for (int f = 0; f < num_features; f++) {
    // Quantile summaries.
    QuantileSummary merged;
    for (int w = 0; w < options.num_workers; w++) {
      std::vector<float> values;
      for (const auto i : worker_examples[w]) {
        values.push_back(data.features[f][i]);
      }
      auto summary = QuantileSummary::FromValues(values, {}).value();
      summary.Prune(options.sketch_size);
      merged = QuantileSummary::Merge(merged, summary);
      merged.Prune(options.sketch_size);
    }
    const auto boundaries = ExtractBinBoundaries(merged, options.num_bins);

    // Histograms.
    std::vector<BinStats> reduced(boundaries.size() + 1);
    for (int w = 0; w < options.num_workers; w++) {
      // Simulate the worker by masking the other examples.
      ExampleToNodeMap worker_e2n(n, kClosedNode);
      for (const auto i : worker_examples[w]) worker_e2n[i] = 0;
      std::vector<std::vector<BinStats>> histograms(
          1, std::vector<BinStats>(boundaries.size() + 1));
      CHECK_OK(FillNumericalHistograms(data.features[f], boundaries, worker_e2n,
                                       {0}, labels, &histograms));
      for (size_t b = 0; b < reduced.size(); b++) {
        reduced[b].Add(histograms[0][b]);
      }
    }
    search.AddNumericalFeature(f, /*na_replacement=*/0.f, reduced);
  }

  // Skip lists: worker -> feature -> surviving examples.
  std::vector<std::vector<std::vector<UnsignedExampleIdx>>> skip(
      options.num_workers,
      std::vector<std::vector<UnsignedExampleIdx>>(num_features));
  for (int w = 0; w < options.num_workers; w++) {
    for (int f = 0; f < num_features; f++) skip[w][f] = worker_examples[w];
  }

  BnbResult result;
  while (true) {
    const auto live_bins = search.UpdateAndListLiveBins();
    if (live_bins.empty()) break;
    if (options.max_refinement_rounds >= 0 &&
        result.rounds >= options.max_refinement_rounds) {
      break;
    }
    result.rounds++;
    // Refinements per feature.
    std::vector<std::vector<int>> live_per_feature(num_features);
    std::vector<BinRefinement> refinements(live_bins.size());
    for (int l = 0; l < live_bins.size(); l++) {
      const auto& live = live_bins[l];
      auto& refinement = refinements[l];
      refinement.min_value = live.stats.min_value;
      refinement.max_value = live.stats.max_value;
      refinement.unique_values =
          live.stats.count <= options.exact_scan_threshold;
      if (!refinement.unique_values) {
        refinement.boundaries = EqualWidthSubBinBoundaries(
            live.stats.min_value, live.stats.max_value, options.num_sub_bins);
      }
      live_per_feature[live.feature].push_back(l);
    }
    std::vector<std::vector<std::vector<BinStats>>> per_worker_results(
        live_bins.size(),
        std::vector<std::vector<BinStats>>(options.num_workers));
    for (int f = 0; f < num_features; f++) {
      if (live_per_feature[f].empty()) continue;
      std::vector<BinRefinement> feature_refinements;
      for (const int l : live_per_feature[f]) {
        feature_refinements.push_back(refinements[l]);
      }
      for (int w = 0; w < options.num_workers; w++) {
        std::vector<std::vector<BinStats>> results;
        std::vector<UnsignedExampleIdx> survivors;
        const auto& candidates =
            use_skip_lists ? skip[w][f] : worker_examples[w];
        result.num_refined_examples += candidates.size();
        CHECK_OK(RefineBins(data.features[f], candidates, feature_refinements,
                            labels, &results, &survivors));
        skip[w][f] = survivors;
        for (int k = 0; k < live_per_feature[f].size(); k++) {
          per_worker_results[live_per_feature[f][k]][w] = results[k];
        }
      }
    }
    for (int l = 0; l < live_bins.size(); l++) {
      std::vector<BinStats> merged;
      if (refinements[l].unique_values) {
        merged = MergeUniqueValueBins(per_worker_results[l]);
      } else {
        merged.resize(refinements[l].boundaries.size() + 1);
        for (int w = 0; w < options.num_workers; w++) {
          for (size_t b = 0; b < merged.size(); b++) {
            merged[b].Add(per_worker_results[l][w][b]);
          }
        }
      }
      CHECK_OK(search.ReplaceBin(live_bins[l].feature_slot,
                                 live_bins[l].bin_idx, merged));
    }
    CHECK_OK(search.ApplyReplacements());
  }
  result.found = search.has_split();
  if (result.found) {
    result.split = search.BestSplit().value();
  }
  return result;
}

std::vector<UnsignedExampleIdx> AllExamples(const Data& data) {
  std::vector<UnsignedExampleIdx> examples(data.num_examples());
  std::iota(examples.begin(), examples.end(), 0);
  return examples;
}

std::unique_ptr<SplitCriterion> MakeCriterion(const int criterion_idx) {
  switch (criterion_idx) {
    case 0:
      return std::make_unique<VarianceReductionCriterion>();
    case 1:
      return std::make_unique<HessianGainCriterion>(
          HessianGainCriterion::Options{});
    case 2: {
      HessianGainCriterion::Options options;
      options.l1 = 0.5;
      options.l2_numerical = 2.0;
      options.l2_categorical = 2.0;
      options.min_sum_hessian_in_leaf = 3.0;
      return std::make_unique<HessianGainCriterion>(options);
    }
    default: {
      HessianGainCriterion::Options options;
      options.l2_numerical = 1.0;
      options.subtract_parent = true;
      return std::make_unique<HessianGainCriterion>(options);
    }
  }
}

TEST(BinStats, AddAndProto) {
  BinStats a;
  a.AddExample(/*value=*/1.f, /*gradient=*/2.f, /*hessian=*/0.5f,
               /*weight=*/2.f, RatioType::kGradientOverHessian);
  a.AddExample(3.f, -1.f, 1.f, 1.f, RatioType::kGradientOverHessian);
  EXPECT_EQ(a.count, 2);
  EXPECT_DOUBLE_EQ(a.sum_weights, 3);
  EXPECT_DOUBLE_EQ(a.sum_gradients, 3);
  EXPECT_DOUBLE_EQ(a.sum_squared_gradients, 9);
  EXPECT_DOUBLE_EQ(a.sum_hessians, 2);
  EXPECT_EQ(a.min_value, 1.f);
  EXPECT_EQ(a.max_value, 3.f);
  EXPECT_EQ(a.min_ratio, -1.f);
  EXPECT_EQ(a.max_ratio, 4.f);

  proto::BinStatistics message;
  AppendBinToProto(a, &message);
  AppendBinToProto(BinStats{}, &message);
  ASSERT_OK_AND_ASSIGN(const auto bins, BinsFromProto(message));
  ASSERT_EQ(bins.size(), 2);
  EXPECT_EQ(bins[0].count, 2);
  EXPECT_EQ(bins[0].max_ratio, 4.f);
  EXPECT_TRUE(bins[1].empty());

  decision_tree::proto::LabelStatistics label_stats;
  BinStatsToLabelStatistics(a, &label_stats);
  ASSERT_OK_AND_ASSIGN(const auto restored,
                       LabelStatisticsToBinStats(label_stats));
  EXPECT_EQ(restored.count, 2);
  EXPECT_DOUBLE_EQ(restored.sum_hessians, 2);

  // Non-positive hessian: undefined ratio.
  BinStats b;
  b.AddExample(1.f, 1.f, 0.f, 1.f, RatioType::kGradientOverHessian);
  EXPECT_TRUE(std::isinf(b.min_ratio));
  EXPECT_TRUE(std::isinf(b.max_ratio));
}

TEST(EqualWidthSubBinBoundaries, Base) {
  EXPECT_THAT(EqualWidthSubBinBoundaries(0.f, 4.f, 4),
              ElementsAre(std::nextafter(0.f, 1.f), 1.f, 2.f, 3.f, 4.f));
  // Two consecutive floats.
  const float a = 1.f;
  const float b = std::nextafter(a, 2.f);
  EXPECT_THAT(EqualWidthSubBinBoundaries(a, b, 16), ElementsAre(b));
  EXPECT_TRUE(EqualWidthSubBinBoundaries(1.f, 1.f, 16).empty());
}

TEST(RefineBins, UniqueValuesAndSubBins) {
  Data data;
  data.features = {{1, 5, 2, 2, 9, 3, 7}};
  data.gradients = {1, 1, 1, 1, 1, 1, 1};
  data.hessians = {1, 1, 1, 1, 1, 1, 1};
  const auto labels = data.Labels(RatioType::kGradient);
  std::vector<BinRefinement> refinements(2);
  refinements[0].min_value = 1;
  refinements[0].max_value = 3;
  refinements[0].unique_values = true;
  refinements[1].min_value = 7;
  refinements[1].max_value = 9;
  refinements[1].boundaries = {8};
  std::vector<std::vector<BinStats>> results;
  std::vector<UnsignedExampleIdx> survivors;
  ASSERT_OK(RefineBins(data.features[0], AllExamples(data), refinements, labels,
                       &results, &survivors));
  ASSERT_EQ(results.size(), 2);
  ASSERT_EQ(results[0].size(), 3);
  EXPECT_EQ(results[0][0].min_value, 1);
  EXPECT_EQ(results[0][1].min_value, 2);
  EXPECT_EQ(results[0][1].count, 2);
  EXPECT_EQ(results[0][2].min_value, 3);
  ASSERT_EQ(results[1].size(), 2);
  EXPECT_EQ(results[1][0].count, 1);
  EXPECT_EQ(results[1][1].count, 1);
  // Example 1 (value 5) is not in any refined bin.
  EXPECT_THAT(survivors, ElementsAre(0, 2, 3, 4, 5, 6));
}

class UpperBoundTest : public ::testing::TestWithParam<int> {};

// The upper bound of a bin is greater or equal to the score of any split that
// puts the prefix and an arbitrary subset of the bin in the negative child.
TEST_P(UpperBoundTest, IsValid) {
  const auto criterion = MakeCriterion(GetParam());
  utils::RandomEngine random(GetParam());
  std::uniform_real_distribution<float> unif;
  std::normal_distribution<float> normal;
  int num_checks = 0;
  for (int trial = 0; trial < 300; trial++) {
    const int n = 20 + trial % 50;
    std::vector<float> grads(n), hess(n), weights(n);
    for (int i = 0; i < n; i++) {
      grads[i] = normal(random) + (trial % 3 == 0 ? 3.f : 0.f);
      hess[i] = 0.01f + unif(random);
      weights[i] = trial % 2 ? 1.f : 0.1f + unif(random);
    }
    // Split the examples into prefix (first part), bin and suffix.
    const int prefix_end = n * unif(random) * 0.5;
    const int bin_end = prefix_end + 1 + (n - prefix_end - 1) * unif(random);
    BinStats parent, prefix, bin;
    for (int i = 0; i < n; i++) {
      parent.AddExample(0, grads[i], hess[i], weights[i],
                        criterion->ratio_type());
      if (i < prefix_end) {
        prefix.AddExample(0, grads[i], hess[i], weights[i],
                          criterion->ratio_type());
      } else if (i < bin_end) {
        bin.AddExample(0, grads[i], hess[i], weights[i],
                       criterion->ratio_type());
      }
    }
    const auto bound = criterion->UpperBound(parent, prefix, bin);
    for (int subset_trial = 0; subset_trial < 50; subset_trial++) {
      BinStats neg = prefix;
      for (int i = prefix_end; i < bin_end; i++) {
        // Random subsets, including "sorted by ratio" subsets.
        bool select;
        if (subset_trial == 0) {
          select = grads[i] / hess[i] > 0;
        } else if (subset_trial == 1) {
          select = grads[i] > 0;
        } else {
          select = unif(random) < 0.5f;
        }
        if (select) {
          neg.AddExample(0, grads[i], hess[i], weights[i],
                         criterion->ratio_type());
        }
      }
      BinStats pos = parent;
      pos.SubSums(neg);
      const double score = criterion->Score(parent, neg, pos);
      EXPECT_LE(score, bound.value + 1e-9 * (1 + bound.scale))
          << "trial:" << trial << " subset_trial:" << subset_trial;
      num_checks++;
    }
  }
  EXPECT_EQ(num_checks, 300 * 50);
}

INSTANTIATE_TEST_SUITE_P(Criteria, UpperBoundTest,
                         ::testing::Values(0, 1, 2, 3));

struct BnbTestParam {
  int criterion_idx;
  bool weighted;
  bool zero_hessians;
  int num_workers;
  int64_t min_examples;
  int num_bins;
};

class BnbVsBruteForce : public ::testing::TestWithParam<BnbTestParam> {};

// The branch-and-bound finds the same split as the exhaustive search.
TEST_P(BnbVsBruteForce, SameSplit) {
  const auto& param = GetParam();
  const auto criterion = MakeCriterion(param.criterion_idx);
  utils::RandomEngine random(1234 + param.criterion_idx);
  int num_found = 0;
  int num_same = 0;
  const int num_trials = 25;
  for (int trial = 0; trial < num_trials; trial++) {
    const int num_examples = 200 + 150 * trial;
    const auto data = GenerateData(num_examples, param.weighted,
                                   param.zero_hessians, &random);
    // Use a random subset of the examples (simulates a non-root node).
    std::vector<UnsignedExampleIdx> examples;
    std::uniform_real_distribution<float> unif;
    for (UnsignedExampleIdx i = 0; i < num_examples; i++) {
      if (trial % 2 == 0 || unif(random) < 0.6f) examples.push_back(i);
    }

    const auto expected =
        BruteForceSplit(data, examples, *criterion, param.min_examples);
    BnbOptions options;
    options.num_workers = param.num_workers;
    options.num_bins = param.num_bins;
    options.min_examples = param.min_examples;
    const auto result = RunBnb(data, examples, *criterion, options);
    ASSERT_EQ(result.found, expected.found) << "trial " << trial;
    if (!expected.found) continue;
    num_found++;

    const auto& condition = result.split.condition;
    const int feature = condition.attribute();
    const float threshold =
        condition.condition().higher_condition().threshold();
    // The split found by the branch-and-bound is optimal.
    const double tolerance = 1e-9 * (1 + std::abs(expected.score));
    // Note: "split_score" is stored with float precision.
    EXPECT_NEAR(condition.split_score(), expected.score,
                1e-6 * (1 + std::abs(expected.score)))
        << "trial " << trial;
    EXPECT_NEAR(ScoreOfSplit(data, examples, *criterion, feature, threshold),
                expected.score, tolerance)
        << "trial " << trial;
    if (feature == expected.feature && threshold == expected.threshold) {
      num_same++;
    } else {
      LOG(INFO) << "Different but equivalent split. trial:" << trial
                << " expected:" << expected.feature
                << ">=" << expected.threshold << " got:" << feature
                << ">=" << threshold;
    }

    // Label statistics and counts.
    int64_t num_pos = 0;
    for (const auto i : examples) {
      num_pos += data.features[feature][i] >= threshold;
    }
    EXPECT_EQ(condition.num_pos_training_examples_without_weight(), num_pos);
    EXPECT_EQ(condition.num_training_examples_without_weight(),
              examples.size());
    EXPECT_EQ(result.split.label_statistics[1].num_examples(), num_pos);
    EXPECT_EQ(result.split.label_statistics[0].num_examples(),
              examples.size() - num_pos);
  }
  EXPECT_GT(num_found, num_trials / 2);
  // Apart from exact ties, the split is the same.
  EXPECT_GE(num_same, num_found - 1);
}

INSTANTIATE_TEST_SUITE_P(All, BnbVsBruteForce,
                         ::testing::Values(
                             // Variance reduction.
                             BnbTestParam{0, false, false, 1, 1, 16},
                             BnbTestParam{0, true, false, 3, 1, 16},
                             BnbTestParam{0, false, false, 4, 20, 8},
                             BnbTestParam{0, true, false, 2, 5, 2},
                             // Hessian gain.
                             BnbTestParam{1, false, false, 3, 1, 16},
                             BnbTestParam{1, true, true, 3, 1, 16},
                             BnbTestParam{2, true, false, 5, 10, 32},
                             BnbTestParam{2, false, true, 2, 1, 4},
                             BnbTestParam{3, false, false, 3, 1, 16},
                             BnbTestParam{3, true, false, 1, 7, 64}));

TEST(Bnb, SkipListsMatchFullRescans) {
  utils::RandomEngine random(7);
  const auto data = GenerateData(3000, /*weighted=*/true,
                                 /*zero_hessians=*/false, &random);
  const auto examples = AllExamples(data);
  VarianceReductionCriterion criterion;
  BnbOptions options;
  options.num_bins = 8;
  const auto with_skip = RunBnb(data, examples, criterion, options, true);
  const auto without_skip = RunBnb(data, examples, criterion, options, false);
  ASSERT_TRUE(with_skip.found);
  EXPECT_THAT(with_skip.split.condition,
              test::EqualsProto(without_skip.split.condition));
  EXPECT_EQ(with_skip.rounds, without_skip.rounds);
  // Skipping eliminated examples reduces the work.
  EXPECT_LT(with_skip.num_refined_examples, without_skip.num_refined_examples);
}

TEST(Bnb, MaxRefinementRounds) {
  utils::RandomEngine random(8);
  const auto data = GenerateData(5000, /*weighted=*/false,
                                 /*zero_hessians=*/false, &random);
  const auto examples = AllExamples(data);
  HessianGainCriterion criterion({});
  const auto expected = BruteForceSplit(data, examples, criterion, 1);

  BnbOptions options;
  options.num_bins = 4;
  const auto exact = RunBnb(data, examples, criterion, options);
  EXPECT_GT(exact.rounds, 1);
  EXPECT_NEAR(exact.split.condition.split_score(), expected.score,
              1e-6 * std::abs(expected.score));

  // The score increases with the number of rounds, and is always achievable.
  double previous_score = -std::numeric_limits<double>::infinity();
  for (int rounds = 0; rounds <= exact.rounds; rounds++) {
    options.max_refinement_rounds = rounds;
    const auto result = RunBnb(data, examples, criterion, options);
    ASSERT_TRUE(result.found);
    EXPECT_LE(result.rounds, rounds);
    const auto& condition = result.split.condition;
    const double score = condition.split_score();
    EXPECT_GE(score, previous_score);
    EXPECT_LE(score, expected.score * (1 + 1e-6));
    EXPECT_NEAR(
        ScoreOfSplit(data, examples, criterion, condition.attribute(),
                     condition.condition().higher_condition().threshold()),
        score, 1e-6 * std::abs(score));
    previous_score = score;
  }
  EXPECT_NEAR(previous_score, expected.score, 1e-6 * std::abs(expected.score));
}

TEST(Bnb, MultipleNodesHistograms) {
  // Histograms of two nodes in a single pass are the same as one pass per
  // node.
  utils::RandomEngine random(9);
  const auto data = GenerateData(1000, true, false, &random);
  const auto labels = data.Labels(RatioType::kGradient);
  ExampleToNodeMap example_to_node(data.num_examples());
  for (size_t i = 0; i < data.num_examples(); i++) {
    example_to_node[i] = i % 3 == 0 ? kClosedNode : (i % 3) - 1;
  }
  const std::vector<float> boundaries = {-1.f, 0.f, 1.f};
  std::vector<std::vector<BinStats>> both(2, std::vector<BinStats>(4));
  ASSERT_OK(FillNumericalHistograms(data.features[0], boundaries,
                                    example_to_node, {1, 0}, labels, &both));
  for (int node = 0; node < 2; node++) {
    std::vector<std::vector<BinStats>> single(1, std::vector<BinStats>(4));
    std::vector<int> node_to_slot = {-1, -1};
    node_to_slot[node] = 0;
    ASSERT_OK(FillNumericalHistograms(data.features[0], boundaries,
                                      example_to_node, node_to_slot, labels,
                                      &single));
    for (int b = 0; b < 4; b++) {
      EXPECT_EQ(single[0][b].count, both[1 - node][b].count);
      EXPECT_EQ(single[0][b].sum_gradients, both[1 - node][b].sum_gradients);
    }
  }
  const auto per_node = ListExamplesPerNode(example_to_node, 2);
  EXPECT_EQ(per_node[0].size() + per_node[1].size(), 666);
}

TEST(Bnb, CategoricalAndBoolean) {
  // Category 2 has a large gradient.
  VarianceReductionCriterion criterion;
  std::vector<BinStats> per_value(4);
  for (int value = 0; value < 4; value++) {
    for (int i = 0; i < 10; i++) {
      per_value[value].AddExample(value, value == 2 ? 3.f : 0.f + 0.01f * i,
                                  1.f, 1.f, RatioType::kGradient);
    }
  }
  BinStats parent;
  for (const auto& v : per_value) parent.Add(v);
  NodeSplitSearch search(&criterion, parent, 1);
  ASSERT_OK(search.AddCategoricalFeature(/*feature=*/5, /*na_replacement=*/0,
                                         per_value));
  std::vector<BinStats> boolean_values = {per_value[0], parent};
  boolean_values[1].SubSums(per_value[0]);
  ASSERT_OK(search.AddBooleanFeature(/*feature=*/6, false, boolean_values));
  EXPECT_TRUE(search.UpdateAndListLiveBins().empty());
  ASSERT_TRUE(search.has_split());
  ASSERT_OK_AND_ASSIGN(const auto split, search.BestSplit());
  EXPECT_EQ(split.condition.attribute(), 5);
  EXPECT_EQ(split.condition.num_pos_training_examples_without_weight(), 10);
  EXPECT_EQ(split.label_statistics[1].num_examples(), 10);
  EXPECT_NEAR(
      split.label_statistics[1].regression_with_hessian().labels().sum(), 30.,
      1e-6);
}

}  // namespace
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
