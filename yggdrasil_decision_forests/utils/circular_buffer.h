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

// Circular buffer.

#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_CIRCULAR_BUFFER_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_CIRCULAR_BUFFER_H_

#include <vector>

#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace utils {

template <typename T>
class CircularBuffer {
 public:
  CircularBuffer() : CircularBuffer(0) {}

  explicit CircularBuffer(size_t size) {
    capacity_ = size;
    begin_ = 0;
    size_ = 0;
    elements_ = std::allocator<T>().allocate(capacity_);
  }

  ~CircularBuffer() {
    clear();
    std::allocator<T>().deallocate(elements_, capacity_);
  }

  std::vector<T> to_vector() const {
    std::vector<T> tmp;
    tmp.reserve(size_);
    for (size_t i = 0; i < size_; i++) {
      const auto idx = WrapIndexOnce(begin_ + i);
      tmp.push_back(elements_[idx]);
    }
    return tmp;
  }

  void clear() {
    for (size_t i = 0; i < size_; i++) {
      const auto idx = WrapIndexOnce(begin_ + i);
      elements_[idx].~T();
    }
    begin_ = 0;
    size_ = 0;
  }

  void clear_and_resize(size_t new_size) {
    clear();
    if (new_size != capacity_) {
      std::allocator<T>().deallocate(elements_, capacity_);
      capacity_ = new_size;
      elements_ = std::allocator<T>().allocate(capacity_);
    }
  }

  bool full() const { return size_ == capacity_; }

  bool empty() const { return size_ == 0; }

  size_t size() const { return size_; }

  template <typename V>
  void push_front(V&& value) {
    DCHECK_LT(size_, capacity_);
    if (begin_ == 0) {
      begin_ = capacity_;
    }
    begin_--;
    auto* mem = elements_ + begin_;
    *new (mem) T(std::forward<V>(value));
    size_++;
  }

  template <typename V>
  void push_back(V&& value) {
    DCHECK_LT(size_, capacity_);
    auto* mem = elements_ + WrapIndexOnce(begin_ + size_);
    *new (mem) T(std::forward<V>(value));
    size_++;
  }

  void pop_front() {
    DCHECK_GT(size_, 0);
    front().~T();
    begin_++;
    if (begin_ == capacity_) {
      begin_ = 0;
    }
    size_--;
  }

  void pop_back() {
    DCHECK_GT(size_, 0);
    back().~T();
    size_--;
  }

  const T& front() const { return elements_[begin_]; }
  T& front() { return elements_[begin_]; }

  const T& back() const { return elements_[WrapIndexOnce(begin_ + size_ - 1)]; }
  T& back() { return elements_[WrapIndexOnce(begin_ + size_ - 1)]; }

 private:
  size_t WrapIndexOnce(size_t index) const {
    DCHECK_GE(index, 0);
    if (index >= capacity_) {
      index -= capacity_;
    }
    DCHECK_LT(index, capacity_);
    return index;
  }

  size_t capacity_;
  size_t begin_;
  size_t size_;
  T* elements_;
};

}  // namespace utils
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_CIRCULAR_BUFFER_H_
