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

// This library contains functions that can be executed either on GPU or on
// their equivalent CPU fall-back implementations.

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_GPU_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_GPU_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests::model::decision_tree::gpu {

// Checks if a GPU is available or fails with an error. If this function is
// called multiple times, it always returns the same value without any extra
// computation (i.e., it is safe to call this methods multiple times).
//
// This function aims to be the central hub for GPU initialization. If other
// parts of the code start to have GPU implementations (e.g. inference), they
// all should call this "Initialize" function.
absl::Status CheckHasGPU(bool print_info = true);

// Adds two vectors of floats using the CPU.
absl::Status AddTwoVectorsCPU(absl::Span<const float> src_1,
                              absl::Span<const float> src_2,
                              absl::Span<float> dst);

// Adds two vectors of floats using the GPU.
absl::Status AddTwoVectorsGPU(absl::Span<const float> a,
                              absl::Span<const float> b, absl::Span<float> c);

// Utility class to compute the projections of vector-sequence conditions either
// on CPU or GPU. When computing on GPU. These classes makes a copy of the
// dataset in the GPU memory and keeps an allocated working memory cache until
// deallocation.
class VectorSequenceComputer {
 public:
  // Initializes. If use_gpu=True and no GPU is available, the tool is
  // initialized to use the CPU (i.e., it does not fail).
  static absl::StatusOr<std::unique_ptr<VectorSequenceComputer>> Create(
      const std::vector<
          const dataset::VerticalDataset::NumericalVectorSequenceColumn*>&
          attributes,
      bool use_gpu);

  ~VectorSequenceComputer();

  // Releases the working memory. No work can be sent after this call.
  absl::Status Release();

  // Computes the maximum of the dot product between the vector-sequence and
  // each anchor.
  //
  // dst[example_idx + anchor_idx * selected_examples.size()] =
  //   \max_{seq_idx} anchors[anchor_idx*anchor_dim_,
  //   (anchor_idx+1)*anchor_dim_] * VectorSequence[example_idx, seq_idx]
  // where:
  //   VectorSequence[example_idx, seq_idx] is the seq_idx-th vector of the
  //   vector sequence of the example_idx-th example.
  //
  // Args:
  //   anchors: A vector of size num_anchors * anchor_dim_. The values
  //     [i*anchor_dim_, (i+1)*anchor_dim_] represent the i-th anchor.
  //   num_anchors: Number of anchors. Should be less than
  //     "MaxNumAnchorsInRequest"
  //   dst: Vector of size selected_examples.size() * num_anchors containing for
  //     each anchor and each example, the maximum of the dot product.
  absl::Status ComputeMaxDotProduct(
      int32_t attribute_idx,
      absl::Span<const UnsignedExampleIdx> selected_examples,
      absl::Span<const float> anchors, int num_anchors, absl::Span<float> dst);

  // Computes the minimum distance between the vector-sequence and each anchor.
  // See "ComputeMaxDotProduct" for the definition of the arguments.
  absl::Status ComputeNegMinSquareDistance(
      int32_t attribute_idx,
      absl::Span<const UnsignedExampleIdx> selected_examples,
      absl::Span<const float> anchors, int num_anchors, absl::Span<float> dst);

  // Maximum number of anchors to query at the same time, if any.
  std::optional<int> MaxNumAnchorsInRequest(size_t num_examples);

  // Is the GPU used for the computation?
  bool use_gpu() const { return use_gpu_; }

 private:
  VectorSequenceComputer() = default;

  absl::Status ComputeMaxDotProductCPU(
      int32_t attribute_idx,
      absl::Span<const UnsignedExampleIdx> selected_examples,
      absl::Span<const float> anchors, int num_anchors, absl::Span<float> dst);

  absl::Status ComputeMaxDotProductGPU(
      int32_t attribute_idx,
      absl::Span<const UnsignedExampleIdx> selected_examples,
      absl::Span<const float> anchors, int num_anchors, absl::Span<float> dst);

  absl::Status ComputeNegMinSquareDistanceCPU(
      int32_t attribute_idx,
      absl::Span<const UnsignedExampleIdx> selected_examples,
      absl::Span<const float> anchors, int num_anchors, absl::Span<float> dst);

  absl::Status ComputeNegMinSquareDistanceGPU(
      int32_t attribute_idx,
      absl::Span<const UnsignedExampleIdx> selected_examples,
      absl::Span<const float> anchors, int num_anchors, absl::Span<float> dst);

  absl::Status InitializeCPU();

  absl::Status InitializeGPU();

  absl::Status ReleaseCPU();

  absl::Status ReleaseGPU();

  // Was "Release" already called?
  bool released_called_ = false;
  // Is GPU used for computation.
  bool use_gpu_;
  // Number of anchors allocated on GPU in: d_anchors_.
  size_t num_allocated_anchors_;
  // Number of example allocated on GPU in: d_selected_examples_.
  size_t num_allocated_selected_examples_;
  // Number of example allocated on GPU in: d_dst_.
  size_t num_allocated_dst_;

  struct PerAttribute {
    // Non-owning pointer to the vector-sequence data. The vector is indexed by
    // attribute idx. Non-available attributes or attributes which are not
    // NumericalVectorSequenceColumn have a null pointer.
    const dataset::VerticalDataset::NumericalVectorSequenceColumn* attribute =
        nullptr;

    // Constant data for GPU containing a copy of the example vector-sequence
    // data.
    size_t* d_item_begins_ = nullptr;
    int32_t* d_item_sizes_ = nullptr;
    float* d_values_ = nullptr;
  };
  std::vector<PerAttribute> per_attributes_;

  // Working data for GPU.
  float* d_anchors_ = nullptr;
  UnsignedExampleIdx* d_selected_examples_ = nullptr;

  // Result of GPU computation.
  float* d_dst_ = nullptr;

  // Mutex protecting the GPU memory.
  utils::concurrency::Mutex gpu_mutex_;
};

}  // namespace yggdrasil_decision_forests::model::decision_tree::gpu

#endif  // THIRD_PARTY_YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_GPU_H_
