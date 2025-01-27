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

#include <cstddef>
#include <cstdint>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"

namespace yggdrasil_decision_forests::model::decision_tree::gpu {

namespace {

constexpr int kThreadPerBlocksExample = 32;

// Adds two vectors.
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// Maximum of dot product kernel.
__global__ void KernelComputeMaxDotProduct(
    size_t *__restrict__ item_begins, int32_t *__restrict__ item_sizes,
    float *__restrict__ values, float *__restrict__ anchors,
    UnsignedExampleIdx *__restrict__ selected_examples, volatile float *dst,
    int num_selected_examples, int num_anchors, int anchor_dim,
    int num_anchor_item_copy_per_thread) {
  int local_example_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int anchor_idx = blockIdx.y;

  // Copy of anchor data in thread shared memory
  extern __shared__ float shared_anchor[];
  if (threadIdx.x * num_anchor_item_copy_per_thread < anchor_dim) {
    for (int i = 0; i < num_anchor_item_copy_per_thread; i++) {
      const int offset = i + num_anchor_item_copy_per_thread * threadIdx.x;
      if (offset < anchor_dim) {
        shared_anchor[offset] = anchors[anchor_idx * anchor_dim + offset];
      }
    }
  }

  if (local_example_idx >= num_selected_examples) {
    return;
  }

  __syncthreads();

  // Compute the dot product
  const int example_idx = selected_examples[local_example_idx];
  const int offset_sequence = item_begins[example_idx];
  const int num_sequences = item_sizes[example_idx];

  float max_p = std::numeric_limits<float>::lowest();
  for (int sequence_idx = 0; sequence_idx < num_sequences; sequence_idx++) {
    float acc = 0;
    const int offset_vector = offset_sequence + sequence_idx * anchor_dim;
    for (int dim_idx = 0; dim_idx < anchor_dim; dim_idx++) {
      acc += shared_anchor[dim_idx] * values[offset_vector + dim_idx];
    }
    if (acc > max_p) {
      max_p = acc;
    }
  }
  dst[local_example_idx + anchor_idx * num_selected_examples] = max_p;
}

// Minimum of distance kernel.
__global__ void KernelComputeNegMinSquareDistance(
    size_t *__restrict__ item_begins, int32_t *__restrict__ item_sizes,
    float *__restrict__ values, float *__restrict__ anchors,
    UnsignedExampleIdx *__restrict__ selected_examples, volatile float *dst,
    int num_selected_examples, int num_anchors, int anchor_dim,
    int num_anchor_item_copy_per_thread) {
  int local_example_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int anchor_idx = blockIdx.y;

  // Copy of anchor data in thread shared memory
  extern __shared__ float shared_anchor[];
  if (threadIdx.x * num_anchor_item_copy_per_thread < anchor_dim) {
    for (int i = 0; i < num_anchor_item_copy_per_thread; i++) {
      const int offset = i + num_anchor_item_copy_per_thread * threadIdx.x;
      if (offset < anchor_dim) {
        shared_anchor[offset] = anchors[anchor_idx * anchor_dim + offset];
      }
    }
  }

  if (local_example_idx >= num_selected_examples) {
    return;
  }

  __syncthreads();

  // Compute the dot product
  const int example_idx = selected_examples[local_example_idx];
  const int offset_sequence = item_begins[example_idx];
  const int num_sequences = item_sizes[example_idx];

  float min_d2 = std::numeric_limits<float>::max();
  for (int sequence_idx = 0; sequence_idx < num_sequences; sequence_idx++) {
    float d2 = 0;
    const int offset_vector = offset_sequence + sequence_idx * anchor_dim;
    for (int dim_idx = 0; dim_idx < anchor_dim; dim_idx++) {
      float v = shared_anchor[dim_idx] - values[offset_vector + dim_idx];
      d2 += v * v;
    }
    if (d2 < min_d2) {
      min_d2 = d2;
    }
  }
  dst[local_example_idx + anchor_idx * num_selected_examples] = -min_d2;
}

// Converts a Cuda status into an absl status.
absl::Status CudaStatus(cudaError_t code) {
  if (code != cudaSuccess) {
    const char *error = cudaGetErrorString(code);
    return absl::InvalidArgumentError(absl::StrCat("Cuda error: ", error));
  }
  return absl::OkStatus();
}

// RETURN_IF_ERROR on Cuda status.
#define RET_CUDA(x) RETURN_IF_ERROR(CudaStatus(x))

// Copies a vector from host to device.
template <typename T>
absl::Status EasyCudaCopyH2D(absl::Span<const T> src, T *dst) {
  return CudaStatus(cudaMemcpy(dst, src.data(), src.size() * sizeof(T),
                               cudaMemcpyHostToDevice));
}

// Copies a vector from device to host.
template <typename T>
absl::Status EasyCudaCopyD2H(T *src, size_t n, absl::Span<T> dst) {
  return CudaStatus(
      cudaMemcpy(dst.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost));
}

// Allocates and copy data from host to device.
template <typename T>
absl::Status EasyCudaAllocAndCopy(absl::Span<const T> src, T **dst) {
  RET_CUDA(cudaMalloc((void **)dst, src.size() * sizeof(T)));
  return EasyCudaCopyH2D<T>(src, *dst);
}

}  // namespace

absl::Status CheckHasGPU(bool print_info) {
  static absl::Status status = [&]() -> absl::Status {
    int driver_version = 0;
    RET_CUDA(cudaDriverGetVersion(&driver_version));
    if (driver_version == 0) {
      return absl::InvalidArgumentError("No matching cuda driver found");
    }
    cudaDeviceProp prop;
    RET_CUDA(cudaGetDeviceProperties(&prop, 0));
    if (print_info) {
      LOG(INFO) << "Using CUDA device: " << prop.name
                << " (driver:" << driver_version << ")";
    }
    return absl::OkStatus();
  }();
  return status;
}

absl::Status AddTwoVectorsGPU(absl::Span<const float> a,
                              absl::Span<const float> b, absl::Span<float> c) {
  DCHECK_EQ(a.size(), b.size());
  DCHECK_EQ(a.size(), c.size());
  size_t n = a.size();

  // Allocate memory on the device
  float *d_a, *d_b, *d_c;
  RET_CUDA(cudaMalloc((void **)&d_a, n * sizeof(float)));
  RET_CUDA(cudaMalloc((void **)&d_b, n * sizeof(float)));
  RET_CUDA(cudaMalloc((void **)&d_c, n * sizeof(float)));

  // Copy data from host to device
  RET_CUDA(
      cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  RET_CUDA(
      cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  // Launch the kernel
  constexpr int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
  RET_CUDA(cudaPeekAtLastError());

  // Copy data from device to host
  RET_CUDA(
      cudaMemcpy(c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

  RET_CUDA(cudaFree(d_a));
  RET_CUDA(cudaFree(d_b));
  RET_CUDA(cudaFree(d_c));

  return absl::OkStatus();
}

absl::Status VectorSequenceComputer::InitializeGPU() {
  num_allocated_anchors_ = 1000;
  const int num_allocated_dst_factor = 10;

  size_t num_allocated_anchor_values = 1;
  for (auto &per_attribute : per_attributes_) {
    if (!per_attribute.attribute) {
      continue;
    }
    const auto *attribute = per_attribute.attribute;
    num_allocated_selected_examples_ = attribute->nrows();
    num_allocated_dst_ = attribute->nrows() * num_allocated_dst_factor;

    // Constants
    RETURN_IF_ERROR(EasyCudaAllocAndCopy<size_t>(
        attribute->item_begins(), &per_attribute.d_item_begins_));
    RETURN_IF_ERROR(EasyCudaAllocAndCopy<int32_t>(
        attribute->item_sizes(), &per_attribute.d_item_sizes_));
    RETURN_IF_ERROR(EasyCudaAllocAndCopy<float>(attribute->bank(),
                                                &per_attribute.d_values_));

    num_allocated_anchor_values =
        std::max(num_allocated_anchor_values,
                 num_allocated_anchors_ * attribute->vector_length());
  }

  // Working data
  RET_CUDA(cudaMalloc((void **)&d_anchors_,
                      num_allocated_anchor_values * sizeof(float)));
  RET_CUDA(cudaMalloc(
      (void **)&d_selected_examples_,
      num_allocated_selected_examples_ * sizeof(UnsignedExampleIdx)));
  RET_CUDA(cudaMalloc((void **)&d_dst_, num_allocated_dst_ * sizeof(float)));

  return absl::OkStatus();
}

absl::Status VectorSequenceComputer::ComputeMaxDotProductGPU(
    const int32_t attribute_idx,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<const float> anchors, int num_anchors, absl::Span<float> dst) {
  utils::concurrency::MutexLock l(&gpu_mutex_);

  const auto &per_attribute = per_attributes_[attribute_idx];
  DCHECK(per_attribute.attribute);
  const int anchor_dim = per_attribute.attribute->vector_length();

  DCHECK_LE(num_anchors,
            MaxNumAnchorsInRequest(selected_examples.size()).value());
  DCHECK_EQ(selected_examples.size() * num_anchors, dst.size());

  RETURN_IF_ERROR(EasyCudaCopyH2D(anchors, d_anchors_));
  RETURN_IF_ERROR(EasyCudaCopyH2D<UnsignedExampleIdx>(selected_examples,
                                                      d_selected_examples_));

  // TODO: Reduce kThreadPerBlocksExample if "num_anchors" *
  // kThreadPerBlocksExample is too large.
  const dim3 grid((selected_examples.size() + kThreadPerBlocksExample - 1) /
                      kThreadPerBlocksExample,
                  num_anchors);
  const dim3 block(kThreadPerBlocksExample, 1);

  const int num_anchor_item_copy_per_thread =
      (anchor_dim + kThreadPerBlocksExample - 1) / kThreadPerBlocksExample;

  KernelComputeMaxDotProduct<<<grid, block, anchor_dim * sizeof(float)>>>(
      per_attribute.d_item_begins_, per_attribute.d_item_sizes_,
      per_attribute.d_values_, d_anchors_, d_selected_examples_, d_dst_,
      selected_examples.size(), num_anchors, anchor_dim,
      num_anchor_item_copy_per_thread);

  RET_CUDA(cudaPeekAtLastError());

  RETURN_IF_ERROR(
      EasyCudaCopyD2H(d_dst_, selected_examples.size() * num_anchors, dst));
  return absl::OkStatus();
}

absl::Status VectorSequenceComputer::ComputeNegMinSquareDistanceGPU(
    const int32_t attribute_idx,
    const absl::Span<const UnsignedExampleIdx> selected_examples,
    absl::Span<const float> anchors, int num_anchors, absl::Span<float> dst) {
  utils::concurrency::MutexLock l(&gpu_mutex_);

  const auto &per_attribute = per_attributes_[attribute_idx];
  DCHECK(per_attribute.attribute);
  const int anchor_dim = per_attribute.attribute->vector_length();

  DCHECK_LE(num_anchors,
            MaxNumAnchorsInRequest(selected_examples.size()).value());
  DCHECK_EQ(selected_examples.size() * num_anchors, dst.size());

  RETURN_IF_ERROR(EasyCudaCopyH2D(anchors, d_anchors_));
  RETURN_IF_ERROR(EasyCudaCopyH2D<UnsignedExampleIdx>(selected_examples,
                                                      d_selected_examples_));

  // TODO: Reduce kThreadPerBlocksExample if "num_anchors" *
  // kThreadPerBlocksExample is too large.
  const dim3 grid((selected_examples.size() + kThreadPerBlocksExample - 1) /
                      kThreadPerBlocksExample,
                  num_anchors);
  const dim3 block(kThreadPerBlocksExample, 1);

  const int num_anchor_item_copy_per_thread =
      (anchor_dim + kThreadPerBlocksExample - 1) / kThreadPerBlocksExample;

  KernelComputeNegMinSquareDistance<<<grid, block,
                                      anchor_dim * sizeof(float)>>>(
      per_attribute.d_item_begins_, per_attribute.d_item_sizes_,
      per_attribute.d_values_, d_anchors_, d_selected_examples_, d_dst_,
      selected_examples.size(), num_anchors, anchor_dim,
      num_anchor_item_copy_per_thread);

  RET_CUDA(cudaPeekAtLastError());

  RETURN_IF_ERROR(
      EasyCudaCopyD2H(d_dst_, selected_examples.size() * num_anchors, dst));
  return absl::OkStatus();
}

absl::Status VectorSequenceComputer::ReleaseGPU() {
  for (auto &per_attribute : per_attributes_) {
    if (!per_attribute.attribute) {
      continue;
    }
    RET_CUDA(cudaFree(per_attribute.d_item_begins_));
    RET_CUDA(cudaFree(per_attribute.d_item_sizes_));
    RET_CUDA(cudaFree(per_attribute.d_values_));
  }
  RET_CUDA(cudaFree(d_anchors_));
  RET_CUDA(cudaFree(d_selected_examples_));
  RET_CUDA(cudaFree(d_dst_));
  return absl::OkStatus();
}

}  // namespace yggdrasil_decision_forests::model::decision_tree::gpu
