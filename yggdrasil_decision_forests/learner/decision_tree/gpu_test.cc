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

#include "yggdrasil_decision_forests/learner/decision_tree/gpu.h"

#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/time/clock.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests::model::decision_tree::gpu {
namespace {
constexpr double kEps = 1e-4;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::Pointwise;

dataset::VerticalDataset::NumericalVectorSequenceColumn BuildSyntheticInputData(
    const int num_dimensions, const int max_num_sequences,
    const int num_examples) {
  utils::RandomEngine rng;
  dataset::VerticalDataset::NumericalVectorSequenceColumn column(
      num_dimensions);
  for (int example_idx = 0; example_idx < num_examples; example_idx++) {
    std::vector<float> values;
    int num_sequence = std::poisson_distribution<int>(50)(rng);
    if (std::uniform_real_distribution<float>()(rng) < 0.01) {
      num_sequence = utils::RandomUniformInt(max_num_sequences, &rng);
    }

    if (num_sequence > max_num_sequences) {
      num_sequence = max_num_sequences;
    }
    values.reserve(num_sequence * num_dimensions);
    for (int sequence_idx = 0; sequence_idx < num_sequence; sequence_idx++) {
      for (int dim_idx = 0; dim_idx < num_dimensions; dim_idx++) {
        values.push_back(std::uniform_real_distribution<float>(-1, 1)(rng));
      }
    }
    column.Add(values);
  }
  return column;
}

struct SyntheticQueryData {
  std::vector<UnsignedExampleIdx> selected_examples;
  std::vector<float> anchors;
};

SyntheticQueryData BuildSyntheticQueryData(const int num_dimensions,
                                           const int num_examples,
                                           const int num_anchors) {
  utils::RandomEngine rng;
  SyntheticQueryData res;
  res.selected_examples.resize(num_examples);
  std::iota(res.selected_examples.begin(), res.selected_examples.end(), 0);

  res.anchors.resize(num_dimensions * num_anchors);
  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    for (int dim_idx = 0; dim_idx < num_dimensions; dim_idx++) {
      res.anchors[anchor_idx * num_dimensions + dim_idx] =
          std::uniform_real_distribution<float>(-1, 1)(rng);
    }
  }
  return res;
}

TEST(AddTwoVectorsCPU, Base) {
  std::vector<float> x{1, 2, 3, 4, 5};
  std::vector<float> y{10, 20, 30, 40, 50};
  std::vector<float> z(5);
  ASSERT_OK(AddTwoVectorsCPU(x, y, absl::Span<float>{z}));
  EXPECT_THAT(z, ElementsAre(11, 22, 33, 44, 55));
}

TEST(AddTwoVectorsGPU, Base) {
#ifndef CUDA_ENABLED
  return;
#endif
  ASSERT_OK(CheckHasGPU());
  std::vector<float> x{1, 2, 3, 4, 5};
  std::vector<float> y{10, 20, 30, 40, 50};
  std::vector<float> z(5);
  ASSERT_OK(AddTwoVectorsGPU(x, y, absl::Span<float>{z}));
  EXPECT_THAT(z, ElementsAre(11, 22, 33, 44, 55));
}

SIMPLE_PARAMETERIZED_TEST(ComputeMaxDotProductOneAnchor, bool, {false, true}) {
  dataset::VerticalDataset::NumericalVectorSequenceColumn data(2);
  data.Add({});
  data.Add({1, 1});
  data.Add({1, 1, 2, 2, 3, 3});
  const float no_found = std::numeric_limits<float>::lowest();

  ASSERT_OK_AND_ASSIGN(auto computer,
                       VectorSequenceComputer::Create({&data}, GetParam()));
  std::vector<float> dst(3);
  ASSERT_OK(computer->ComputeMaxDotProduct(0, {0, 1, 2}, {1, 1}, 1,
                                           absl::Span<float>{dst}));
  ASSERT_OK(computer->Release());
  EXPECT_THAT(dst, Pointwise(FloatNear(kEps), {no_found, 2.f, 6.f}));
}

SIMPLE_PARAMETERIZED_TEST(ComputeMaxDotProductTwoAnchor, bool, {false, true}) {
  dataset::VerticalDataset::NumericalVectorSequenceColumn data(2);
  data.Add({});
  data.Add({1, 1});
  data.Add({1, 1, 2, 2, 3, 3});

  const float no_found = std::numeric_limits<float>::lowest();
  ASSERT_OK_AND_ASSIGN(auto computer,
                       VectorSequenceComputer::Create({&data}, GetParam()));
  std::vector<float> dst(6);
  ASSERT_OK(computer->ComputeMaxDotProduct(0, {0, 1, 2}, {1, 1, 2, 2}, 2,
                                           absl::Span<float>{dst}));
  ASSERT_OK(computer->Release());
  EXPECT_THAT(dst, Pointwise(FloatNear(kEps),
                             {no_found, 2.f, 6.f, no_found, 4.f, 12.f}));
}

SIMPLE_PARAMETERIZED_TEST(ComputeNegMinSquareDistanceTwoAnchor, bool,
                          {false, true}) {
  dataset::VerticalDataset::NumericalVectorSequenceColumn data(2);
  data.Add({});
  data.Add({1, 1});
  data.Add({1, 1, 2, 2, 3, 3});
  const float no_found = -std::numeric_limits<float>::max();

  ASSERT_OK_AND_ASSIGN(auto computer,
                       VectorSequenceComputer::Create({&data}, GetParam()));
  std::vector<float> dst(6);
  ASSERT_OK(computer->ComputeNegMinSquareDistance(0, {0, 1, 2}, {1, 3, 2, 2}, 2,
                                                  absl::Span<float>{dst}));
  ASSERT_OK(computer->Release());
  EXPECT_THAT(dst, Pointwise(FloatNear(kEps),
                             {no_found, -4.f, -2.f, no_found, -2.f, -0.f}));
}

TEST(ComputeMaxDotProduct, Large) {
  const int num_dimensions = 2000;
  const int max_num_sequences = 200;
  const int num_examples = 1000;

  const auto input_data =
      BuildSyntheticInputData(num_dimensions, max_num_sequences, num_examples);

  ASSERT_OK_AND_ASSIGN(auto computer_cpu,
                       VectorSequenceComputer::Create({&input_data}, false));
  ASSERT_OK_AND_ASSIGN(auto computer_gpu,
                       VectorSequenceComputer::Create({&input_data}, true));

  const int num_anchors =
      computer_gpu->MaxNumAnchorsInRequest(num_examples).value_or(100);
  EXPECT_GE(num_anchors, 1);

  std::vector<float> dst_cpu(num_examples * num_anchors);
  std::vector<float> dst_gpu(num_examples * num_anchors);

  auto query_data =
      BuildSyntheticQueryData(num_dimensions, num_examples, num_anchors);

  ASSERT_OK(computer_cpu->ComputeMaxDotProduct(0, query_data.selected_examples,
                                               query_data.anchors, num_anchors,
                                               absl::Span<float>{dst_cpu}));

  ASSERT_OK(computer_gpu->ComputeMaxDotProduct(0, query_data.selected_examples,
                                               query_data.anchors, num_anchors,
                                               absl::Span<float>{dst_gpu}));

  ASSERT_OK(computer_cpu->Release());
  ASSERT_OK(computer_gpu->Release());

  EXPECT_THAT(dst_cpu, Pointwise(FloatNear(kEps), dst_gpu));
}

TEST(ComputeNegMinSquareDistance, Large) {
  const int num_dimensions = 2000;
  const int max_num_sequences = 200;
  const int num_examples = 1000;

  const auto input_data =
      BuildSyntheticInputData(num_dimensions, max_num_sequences, num_examples);

  ASSERT_OK_AND_ASSIGN(auto computer_cpu,
                       VectorSequenceComputer::Create({&input_data}, false));
  ASSERT_OK_AND_ASSIGN(auto computer_gpu,
                       VectorSequenceComputer::Create({&input_data}, true));

  const int num_anchors =
      computer_gpu->MaxNumAnchorsInRequest(num_examples).value_or(100);
  EXPECT_GE(num_anchors, 1);

  std::vector<float> dst_cpu(num_examples * num_anchors);
  std::vector<float> dst_gpu(num_examples * num_anchors);

  auto query_data =
      BuildSyntheticQueryData(num_dimensions, num_examples, num_anchors);

  ASSERT_OK(computer_cpu->ComputeNegMinSquareDistance(
      0, query_data.selected_examples, query_data.anchors, num_anchors,
      absl::Span<float>{dst_cpu}));

  ASSERT_OK(computer_gpu->ComputeNegMinSquareDistance(
      0, query_data.selected_examples, query_data.anchors, num_anchors,
      absl::Span<float>{dst_gpu}));

  ASSERT_OK(computer_cpu->Release());
  ASSERT_OK(computer_gpu->Release());

  EXPECT_THAT(dst_cpu, Pointwise(FloatNear(1e-3), dst_gpu));
}

TEST(DISABLED_Benchmark, All) {
  ASSERT_OK(CheckHasGPU());
  const int num_dimensions = 2000;
  const int max_num_sequences = 200;

  const int max_num_examples = 10000;

  for (const int selected_num_examples : {10, 50, 100, 500, 1000, 10000}) {
    int num_repeat = 10000 / selected_num_examples;
    if (num_repeat == 0) {
      num_repeat = 1;
    }

    const auto input_data = BuildSyntheticInputData(
        num_dimensions, max_num_sequences, max_num_examples);

    for (const bool use_gpu : {true, false}) {
      if (!use_gpu) {
        continue;
      }

      ASSERT_OK_AND_ASSIGN(auto computer, VectorSequenceComputer::Create(
                                              {&input_data}, use_gpu));

      const int num_anchors =
          computer->MaxNumAnchorsInRequest(selected_num_examples).value_or(100);

      auto query_data = BuildSyntheticQueryData(
          num_dimensions, selected_num_examples, num_anchors);

      std::vector<float> dst(num_anchors * selected_num_examples);

      ASSERT_OK(computer->ComputeMaxDotProduct(0, query_data.selected_examples,
                                               query_data.anchors, num_anchors,
                                               absl::Span<float>{dst}));

      const auto begin_time = absl::Now();
      for (int repeat_idx = 0; repeat_idx < num_repeat; repeat_idx++) {
        ASSERT_OK(computer->ComputeMaxDotProduct(
            0, query_data.selected_examples, query_data.anchors, num_anchors,
            absl::Span<float>{dst}));
      }
      const auto end_time = absl::Now();
      LOG(INFO) << "num_examples:" << max_num_examples
                << " sel_num_examples:" << selected_num_examples
                << " num_dims:" << num_dimensions
                << " max_num_seqs:" << max_num_sequences
                << " use_gpu:" << use_gpu << " num_repeat:" << num_repeat
                << " num_anchors:" << num_anchors << " wall-time:"
                << (end_time - begin_time) / (num_repeat * num_anchors);
      ASSERT_OK(computer->Release());
    }
  }
}

}  // namespace
}  // namespace yggdrasil_decision_forests::model::decision_tree::gpu
