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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests::model::decision_tree::gpu {
namespace {

// Computes the dot product between two vectors.
float DotProduct(int n, const float* __restrict a, const float* __restrict b) {
  float acc = 0;
  for (int i = 0; i < n; i++) {
    acc += a[i] * b[i];
  }
  return acc;
}

// Computes the squared distance between two vectors. If the squared distance is
// greater than "max_value", return a value greater than "max_value" (but not
// necessary equal to the squared distance).
float SquaredDistance(int n, const float* __restrict a,
                      const float* __restrict b, float max_value) {
  float acc = 0;
  for (int i = 0; i < n; i++) {
    const float v = a[i] - b[i];
    acc += v * v;
    if (acc >= max_value) {
      return acc;
    }
  }
  return acc;
}

}  // namespace

absl::Status AddTwoVectorsCPU(absl::Span<const float> src_1,
                              absl::Span<const float> src_2,
                              absl::Span<float> dst) {
  DCHECK_EQ(src_1.size(), src_2.size());
  DCHECK_EQ(src_1.size(), dst.size());
  for (int i = 0; i < src_1.size(); ++i) {
    dst[i] = src_1[i] + src_2[i];
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<VectorSequenceComputer>>
VectorSequenceComputer::Create(
    const std::vector<
        const dataset::VerticalDataset::NumericalVectorSequenceColumn*>&
        attributes,
    bool use_gpu) {
  if (attributes.empty()) {
    return absl::InvalidArgumentError(
        "At least one attribute should be provided");
  }
  auto c = absl::WrapUnique(new VectorSequenceComputer());

  if (use_gpu) {
    const auto init_gpu_status = CheckHasGPU();
    if (!init_gpu_status.ok()) {
      LOG(INFO) << "Cannot initialize GPU: " << init_gpu_status.message();
      use_gpu = false;
    }
  }

  c->use_gpu_ = use_gpu;
  c->per_attributes_.assign(attributes.size(), {});
  for (int attribute_idx = 0; attribute_idx < attributes.size();
       attribute_idx++) {
    if (attributes[attribute_idx]) {
      c->per_attributes_[attribute_idx].attribute = attributes[attribute_idx];
    }
  }

  if (use_gpu) {
    RETURN_IF_ERROR(c->InitializeGPU());
  } else {
    RETURN_IF_ERROR(c->InitializeCPU());
  }
  return c;
}

VectorSequenceComputer::~VectorSequenceComputer() {
  if (!released_called_) {
    LOG(WARNING) << "Released not called";
    Release().IgnoreError();
  }
}

std::optional<int> VectorSequenceComputer::MaxNumAnchorsInRequest(
    size_t num_examples) {
  if (use_gpu_) {
    DCHECK_LE(num_examples, num_allocated_selected_examples_);
    return std::min(num_allocated_anchors_, num_allocated_dst_ / num_examples);
  }
  return {};
}

absl::Status VectorSequenceComputer::Release() {
  DCHECK(!released_called_);
  released_called_ = true;
  if (use_gpu_) {
    return ReleaseGPU();
  } else {
    return ReleaseCPU();
  }
  return absl::OkStatus();
}

absl::Status VectorSequenceComputer::ReleaseCPU() { return absl::OkStatus(); }

absl::Status VectorSequenceComputer::InitializeCPU() {
  return absl::OkStatus();
}

absl::Status VectorSequenceComputer::ComputeMaxDotProductCPU(
    const int32_t attribute_idx,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<const float> anchors, int num_anchors, absl::Span<float> dst) {
  DCHECK_EQ(selected_examples.size() * num_anchors, dst.size());
  DCHECK_LT(attribute_idx, per_attributes_.size());
  const auto* attribute = per_attributes_[attribute_idx].attribute;
  DCHECK(attribute);
  const int anchor_dim = attribute->vector_length();
  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    size_t local_example_idx = selected_examples.size() * anchor_idx;
    const auto anchor = anchors.subspan(anchor_idx * anchor_dim, anchor_dim);
    for (const auto example_idx : selected_examples) {
      float max_p = std::numeric_limits<float>::lowest();
      if (ABSL_PREDICT_TRUE(!attribute->IsNa(example_idx))) {
        const auto num_vectors = attribute->SequenceLength(example_idx);
        for (int vector_idx = 0; vector_idx < num_vectors; vector_idx++) {
          ASSIGN_OR_RETURN(const auto vector,
                           attribute->GetVector(example_idx, vector_idx));
          const float p =
              DotProduct(vector.size(), vector.data(), anchor.data());
          if (p > max_p) {
            max_p = p;
          }
        }
      }
      dst[local_example_idx++] = max_p;
    }
  }
  return absl::OkStatus();
}

absl::Status VectorSequenceComputer::ComputeNegMinSquareDistanceCPU(
    const int32_t attribute_idx,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<const float> anchors, int num_anchors, absl::Span<float> dst) {
  DCHECK_EQ(selected_examples.size() * num_anchors, dst.size());
  const auto* attribute = per_attributes_[attribute_idx].attribute;
  DCHECK(attribute);
  const int anchor_dim = attribute->vector_length();
  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    size_t local_example_idx = selected_examples.size() * anchor_idx;
    const auto anchor = anchors.subspan(anchor_idx * anchor_dim, anchor_dim);
    for (const auto example_idx : selected_examples) {
      float min_dist2 = std::numeric_limits<float>::max();
      if (ABSL_PREDICT_TRUE(!attribute->IsNa(example_idx))) {
        const auto num_vectors = attribute->SequenceLength(example_idx);
        for (int vector_idx = 0; vector_idx < num_vectors; vector_idx++) {
          ASSIGN_OR_RETURN(const auto vector,
                           attribute->GetVector(example_idx, vector_idx));
          const float dist2 = SquaredDistance(vector.size(), vector.data(),
                                              anchor.data(), min_dist2);
          if (dist2 < min_dist2) {
            min_dist2 = dist2;
          }
        }
      }
      // Note: We negate the values so the vector condition is in the same
      // "direction" as the underlying threshold condition.
      dst[local_example_idx++] = -min_dist2;
    }
  }
  return absl::OkStatus();
}

absl::Status VectorSequenceComputer::ComputeMaxDotProduct(
    const int32_t attribute_idx,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<const float> anchors, const int num_anchors,
    absl::Span<float> dst) {
  DCHECK_EQ(selected_examples.size() * num_anchors, dst.size());
  if (use_gpu_) {
    DCHECK_LE(num_anchors,
              MaxNumAnchorsInRequest(selected_examples.size()).value());
    return ComputeMaxDotProductGPU(attribute_idx, selected_examples, anchors,
                                   num_anchors, dst);
  } else {
    return ComputeMaxDotProductCPU(attribute_idx, selected_examples, anchors,
                                   num_anchors, dst);
  }
}

absl::Status VectorSequenceComputer::ComputeNegMinSquareDistance(
    const int32_t attribute_idx,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<const float> anchors, const int num_anchors,
    absl::Span<float> dst) {
  DCHECK_EQ(selected_examples.size() * num_anchors, dst.size());
  if (use_gpu_) {
    DCHECK_LE(num_anchors,
              MaxNumAnchorsInRequest(selected_examples.size()).value());
    return ComputeNegMinSquareDistanceGPU(attribute_idx, selected_examples,
                                          anchors, num_anchors, dst);
  } else {
    return ComputeNegMinSquareDistanceCPU(attribute_idx, selected_examples,
                                          anchors, num_anchors, dst);
  }
}

#ifdef CUDA_DISABLED
// GPU functions when not compiled with GPU support.

absl::Status CheckHasGPU(bool print_info) {
  return absl::InvalidArgumentError("Not compiled with GPU support");
}

absl::Status VectorSequenceComputer::InitializeGPU() {
  return absl::InvalidArgumentError("Not compiled with GPU support");
}

absl::Status VectorSequenceComputer::ComputeNegMinSquareDistanceGPU(
    const int32_t attribute_idx,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<const float> anchors, int num_anchors, absl::Span<float> dst) {
  return absl::InvalidArgumentError("Not compiled with GPU support");
}

absl::Status VectorSequenceComputer::ReleaseGPU() {
  return absl::InvalidArgumentError("Not compiled with GPU support");
}

absl::Status VectorSequenceComputer::ComputeMaxDotProductGPU(
    const int32_t attribute_idx,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<const float> anchors, int num_anchors, absl::Span<float> dst) {
  return absl::InvalidArgumentError("Not compiled with GPU support");
}

absl::Status AddTwoVectorsGPU(absl::Span<const float> a,
                              absl::Span<const float> b, absl::Span<float> c) {
  return absl::InvalidArgumentError("Not compiled with GPU support");
}

#endif

}  // namespace yggdrasil_decision_forests::model::decision_tree::gpu
