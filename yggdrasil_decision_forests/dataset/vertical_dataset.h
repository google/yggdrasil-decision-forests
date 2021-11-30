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

// Definition of VerticalDataset, a in-memory storage of data column by column.
//
// Note: VerticalDataset is used to in the implementation of the original Random
// Forest algorithm (see ../algorithm/random_forest.h).
//

#ifndef YGGDRASIL_DECISION_FORESTS_DATASET_VERTICAL_DATASET_H_
#define YGGDRASIL_DECISION_FORESTS_DATASET_VERTICAL_DATASET_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/logging.h"

namespace yggdrasil_decision_forests {
namespace dataset {

// In memory transactional dataset with heterogen column type, stored column by
// column for fast row-wise iteration.
class VerticalDataset {
 public:
  // Row index type.
  typedef int64_t row_t;

  // Abstract representation of a column.
  class AbstractColumn {
   public:
    virtual ~AbstractColumn() {}

    // Type of the column.
    virtual proto::ColumnType type() const = 0;

    // String representation of the value at row "row". "digit_precision"
    // controls the number of printed decimal digits for numerical values.
    virtual std::string ToStringWithDigitPrecision(
        const row_t row, const proto::Column& col_spec,
        int digit_precision) const = 0;

    // String representation of a value with a default of 6 decimal digits
    // precision.
    std::string ToString(const row_t row, const proto::Column& col_spec) const {
      return ToStringWithDigitPrecision(row, col_spec,
                                        /*digit_precision = */ 4);
    }

    // Check if a value is NA (i.e. non-available).
    virtual bool IsNa(const row_t row) const = 0;

    // Add a NA value.
    virtual void AddNA() = 0;

    // Set a missing value.
    virtual void SetNA(const row_t row) = 0;

    // Resize the content of size of the column.
    virtual void Resize(const row_t row) = 0;

    // Reserve the content of the column for fast insertion.
    virtual void Reserve(const row_t row) = 0;

    // Number of rows. Should match nrow_ from the dataset.
    virtual row_t nrows() const = 0;

    // Set the "name" of a column.
    void set_name(absl::string_view name) { name_ = std::string(name); }

    const std::string& name() const { return name_; }

    // Add a new value.
    virtual void AddFromExample(const proto::Example::Attribute& attribute) = 0;

    // Set a value.
    virtual void Set(row_t example_idx,
                     const proto::Example::Attribute& attribute) = 0;

    // Extract the value of an attribute.
    virtual void ExtractExample(row_t example_idx,
                                proto::Example::Attribute* attribute) const = 0;

    // Extract a subset of rows.  The "dst" columns
    // should have the same type as "this".
    virtual void ExtractAndAppend(const std::vector<row_t>& indices,
                                  AbstractColumn* dst) const = 0;

    // Converts the content of a column to another dataspec.
    virtual absl::Status ConvertToGivenDataspec(
        AbstractColumn* dst, const proto::Column& src_spec,
        const proto::Column& dst_spec) const = 0;

    // Cast the column with checks.
    template <typename T>
    T* MutableCast() {
      static_assert(
          std::is_base_of<AbstractColumn, T>::value,
          "The template class argument does not derive  AbstractColumn.");
      T* const casted_column = dynamic_cast<T* const>(this);
      if (!casted_column) {
        LOG(FATAL) << "Column \"" << name() << "\" has type "
                   << proto::ColumnType_Name(type())
                   << " and is not compatible with type " << typeid(T).name();
      }
      return casted_column;
    }

    // Used and reserved memory expressed in bytes.
    virtual std::pair<uint64_t, uint64_t> memory_usage() const = 0;

    // Release the reserved but not used memory.
    virtual void ShrinkToFit() = 0;

   private:
    // Unique column name.
    std::string name_;
  };

  // Storage of scalar values.
  template <typename T>
  class TemplateScalarStorage : public AbstractColumn {
   public:
    using Format = T;

    void Resize(const row_t row) override { values_.resize(row); }
    void Reserve(const row_t row) override { values_.reserve(row); }
    row_t nrows() const override { return values_.size(); };

    // Add a value.
    void Add(const T& value) { values_.push_back(value); }

    // Access to values.
    const std::vector<T>& values() const { return values_; }
    std::vector<T>* mutable_values() { return &values_; }

    void ExtractAndAppend(const std::vector<row_t>& indices,
                          AbstractColumn* dst) const override;

    std::pair<uint64_t, uint64_t> memory_usage() const override {
      return std::pair<uint64_t, uint64_t>(values_.size() * sizeof(T),
                                           values_.capacity() * sizeof(T));
    }

    void ShrinkToFit() override { values_.shrink_to_fit(); }

   private:
    std::vector<T> values_;
  };

  // Storage of multi-dimensional, list or set values.
  template <typename T>
  class TemplateMultiValueStorage : public AbstractColumn {
   public:
    bool IsNa(const row_t row) const override {
      return values_[row].first > values_[row].second;
    }
    void Resize(const row_t row) override { values_.resize(row); }
    void Reserve(const row_t row) override { values_.reserve(row); }
    row_t nrows() const override { return values_.size(); };

    // Add a NA (i.e. missing) value.
    void AddNA() override { values_.emplace_back(1, 0); }

    // Set the value to be missing.
    void SetNA(const row_t row) override { values_[row] = {1, 0}; }

    // Add a value.
    template <typename Iter>
    void Add(Iter begin, Iter end) {
      const size_t begin_idx = bank_.size();
      bank_.insert(bank_.end(), begin, end);
      values_.emplace_back(begin_idx, bank_.size());
    }

    // Set a value.
    template <typename Iter>
    void SetIter(const row_t row, Iter begin, Iter end) {
      const size_t begin_idx = bank_.size();
      bank_.insert(bank_.end(), begin, end);
      values_[row] = {begin_idx, bank_.size()};
    }

    // Add a value.
    void AddVector(const std::vector<T>& values) {
      Add(values.begin(), values.end());
    }

    void ExtractAndAppend(const std::vector<row_t>& indices,
                          AbstractColumn* dst) const override;

    const std::vector<std::pair<size_t, size_t>>& values() const {
      return values_;
    }
    std::vector<std::pair<size_t, size_t>>& mutable_values() { return values_; }

    const std::vector<T>& bank() const { return bank_; }
    std::vector<T>& mutable_bank() { return bank_; }

    typename std::vector<T>::const_iterator begin(const row_t row) const {
      return bank_.begin() + values_[row].first;
    }

    typename std::vector<T>::const_iterator end(const row_t row) const {
      if (values_[row].first > values_[row].second) {
        return bank_.begin() + values_[row].first;
      }
      return bank_.begin() + values_[row].second;
    }

    std::pair<uint64_t, uint64_t> memory_usage() const override {
      return std::pair<uint64_t, uint64_t>(
          bank_.size() * sizeof(T) +
              values_.size() * sizeof(std::pair<size_t, size_t>),
          bank_.capacity() * sizeof(T) +
              values_.capacity() * sizeof(std::pair<size_t, size_t>));
    }

    void ShrinkToFit() override {
      values_.shrink_to_fit();
      bank_.shrink_to_fit();
    }

   private:
    // List of all values in a dense array.
    std::vector<T> bank_;
    // Begin (inclusive) and end (exclusive) index in "bank_".
    // If begin==end, the record is empty.
    // If begin>end, the record is NA.
    std::vector<std::pair<size_t, size_t>> values_;
  };

  class NumericalColumn : public TemplateScalarStorage<float> {
   public:
    proto::ColumnType type() const override {
      return proto::ColumnType::NUMERICAL;
    }
    std::string ToStringWithDigitPrecision(const row_t row,
                                           const proto::Column& col_spec,
                                           int digit_precision) const override;
    bool IsNa(const row_t row) const override {
      return std::isnan(values()[row]);
    }

    void AddNA() override { Add(kNaValue); }

    void SetNA(const row_t row) override { Set(row, kNaValue); }

    void AddFromExample(const proto::Example::Attribute& attribute) override;

    void Set(row_t example_idx,
             const proto::Example::Attribute& attribute) override;

    void Set(row_t example_idx, float value);

    void ExtractExample(row_t example_idx,
                        proto::Example::Attribute* attribute) const override;

    absl::Status ConvertToGivenDataspec(
        AbstractColumn* dst, const proto::Column& src_spec,
        const proto::Column& dst_spec) const override;

    static constexpr float kNaValue = std::numeric_limits<float>::quiet_NaN();
  };

  class BooleanColumn : public TemplateScalarStorage<char> {
   public:
    proto::ColumnType type() const override {
      return proto::ColumnType::BOOLEAN;
    }
    std::string ToStringWithDigitPrecision(const row_t row,
                                           const proto::Column& col_spec,
                                           int digit_precision) const override;
    bool IsNa(const row_t row) const override {
      return values()[row] == kNaValue;
    }

    void AddNA() override { Add(kNaValue); }

    void SetNA(const row_t row) override {
      (*mutable_values())[row] = kNaValue;
    }

    void AddFromExample(const proto::Example::Attribute& attribute) override;

    void Set(row_t example_idx,
             const proto::Example::Attribute& attribute) override;

    void ExtractExample(row_t example_idx,
                        proto::Example::Attribute* attribute) const override;

    absl::Status ConvertToGivenDataspec(
        AbstractColumn* dst, const proto::Column& src_spec,
        const proto::Column& dst_spec) const override;

    bool IsTrue(const row_t row) const { return values()[row] == kTrueValue; }

    // Special value used to represent NA.
    static constexpr char kNaValue = 2;
    // Value representing "true".
    static constexpr char kTrueValue = 1;
    // Value representing "false".
    static constexpr char kFalseValue = 0;
  };

  class DiscretizedNumericalColumn
      : public TemplateScalarStorage<DiscretizedNumericalIndex> {
   public:
    proto::ColumnType type() const override {
      return proto::ColumnType::DISCRETIZED_NUMERICAL;
    }
    std::string ToStringWithDigitPrecision(const row_t row,
                                           const proto::Column& col_spec,
                                           int digit_precision) const override;
    bool IsNa(const row_t row) const override {
      return values()[row] == kNaValue;
    }

    void AddNA() override { Add(kNaValue); }

    void SetNA(const row_t row) override {
      (*mutable_values())[row] = kNaValue;
    }

    void AddFromExample(const proto::Example::Attribute& attribute) override;

    void Set(row_t example_idx,
             const proto::Example::Attribute& attribute) override;

    void ExtractExample(row_t example_idx,
                        proto::Example::Attribute* attribute) const override;

    absl::Status ConvertToGivenDataspec(
        AbstractColumn* dst, const proto::Column& src_spec,
        const proto::Column& dst_spec) const override;

    // Special value used to represent NA.
    static constexpr Format kNaValue = kDiscretizedNumericalMissingValue;
  };

  class CategoricalColumn : public TemplateScalarStorage<int32_t> {
   public:
    proto::ColumnType type() const override {
      return proto::ColumnType::CATEGORICAL;
    }
    std::string ToStringWithDigitPrecision(const row_t row,
                                           const proto::Column& col_spec,
                                           int digit_precision) const override;
    bool IsNa(const row_t row) const override {
      return values()[row] == kNaValue;
    }

    void AddNA() override { Add(kNaValue); }

    void SetNA(const row_t row) override {
      (*mutable_values())[row] = kNaValue;
    }

    void AddFromExample(const proto::Example::Attribute& attribute) override;

    void Set(row_t example_idx,
             const proto::Example::Attribute& attribute) override;

    void ExtractExample(row_t example_idx,
                        proto::Example::Attribute* attribute) const override;

    absl::Status ConvertToGivenDataspec(
        AbstractColumn* dst, const proto::Column& src_spec,
        const proto::Column& dst_spec) const override;

    // Special value used to represent NA.
    static constexpr int kNaValue = -1;
  };

  class NumericalSetColumn : public TemplateMultiValueStorage<float> {
   public:
    proto::ColumnType type() const override {
      return proto::ColumnType::NUMERICAL_SET;
    }
    std::string ToStringWithDigitPrecision(const row_t row,
                                           const proto::Column& col_spec,
                                           int digit_precision) const override;
    void AddFromExample(const proto::Example::Attribute& attribute) override;

    void Set(row_t example_idx,
             const proto::Example::Attribute& attribute) override;

    void ExtractExample(row_t example_idx,
                        proto::Example::Attribute* attribute) const override;

    absl::Status ConvertToGivenDataspec(
        AbstractColumn* dst, const proto::Column& src_spec,
        const proto::Column& dst_spec) const override;
  };

  class CategoricalSetColumn : public TemplateMultiValueStorage<int32_t> {
   public:
    proto::ColumnType type() const override {
      return proto::ColumnType::CATEGORICAL_SET;
    }
    std::string ToStringWithDigitPrecision(const row_t row,
                                           const proto::Column& col_spec,
                                           int digit_precision) const override;
    void AddFromExample(const proto::Example::Attribute& attribute) override;

    void Set(row_t example_idx,
             const proto::Example::Attribute& attribute) override;

    void ExtractExample(row_t example_idx,
                        proto::Example::Attribute* attribute) const override;

    absl::Status ConvertToGivenDataspec(
        AbstractColumn* dst, const proto::Column& src_spec,
        const proto::Column& dst_spec) const override;
  };

  class NumericalListColumn : public TemplateMultiValueStorage<float> {
   public:
    proto::ColumnType type() const override {
      return proto::ColumnType::NUMERICAL_LIST;
    }
    std::string ToStringWithDigitPrecision(const row_t row,
                                           const proto::Column& col_spec,
                                           int digit_precision) const override;
    void AddFromExample(const proto::Example::Attribute& attribute) override;

    void Set(row_t example_idx,
             const proto::Example::Attribute& attribute) override;

    void ExtractExample(row_t example_idx,
                        proto::Example::Attribute* attribute) const override;

    absl::Status ConvertToGivenDataspec(
        AbstractColumn* dst, const proto::Column& src_spec,
        const proto::Column& dst_spec) const override;
  };

  class CategoricalListColumn : public TemplateMultiValueStorage<int32_t> {
   public:
    proto::ColumnType type() const override {
      return proto::ColumnType::CATEGORICAL_LIST;
    }
    std::string ToStringWithDigitPrecision(const row_t row,
                                           const proto::Column& col_spec,
                                           int digit_precision) const override;
    void AddFromExample(const proto::Example::Attribute& attribute) override;

    void Set(row_t example_idx,
             const proto::Example::Attribute& attribute) override;

    void ExtractExample(row_t example_idx,
                        proto::Example::Attribute* attribute) const override;

    absl::Status ConvertToGivenDataspec(
        AbstractColumn* dst, const proto::Column& src_spec,
        const proto::Column& dst_spec) const override;
  };

  class StringColumn : public TemplateScalarStorage<std::string> {
   public:
    proto::ColumnType type() const override {
      return proto::ColumnType::STRING;
    }
    std::string ToStringWithDigitPrecision(const row_t row,
                                           const proto::Column& col_spec,
                                           int digit_precision) const override;

    void Add(const std::string& value) {
      TemplateScalarStorage<std::string>::Add(value);
      is_na_.push_back(false);
    }

    bool IsNa(const row_t row) const override { return is_na_[row]; }

    void AddNA() override {
      TemplateScalarStorage<std::string>::Add("");
      is_na_.push_back(true);
    }

    void SetNA(const row_t row) override {
      (*mutable_values())[row] = "";
      is_na_[row] = true;
    }

    void Resize(const row_t row) override {
      TemplateScalarStorage::Resize(row);
      is_na_.resize(row);
    }
    void Reserve(const row_t row) override {
      TemplateScalarStorage::Reserve(row);
      is_na_.reserve(row);
    }
    void AddFromExample(const proto::Example::Attribute& attribute) override;

    void Set(row_t example_idx,
             const proto::Example::Attribute& attribute) override;

    void Set(row_t example_idx, const absl::string_view value);

    void ExtractExample(row_t example_idx,
                        proto::Example::Attribute* attribute) const override;

    absl::Status ConvertToGivenDataspec(
        AbstractColumn* dst, const proto::Column& src_spec,
        const proto::Column& dst_spec) const override;

   private:
    std::vector<bool> is_na_;
  };

  class HashColumn : public TemplateScalarStorage<uint64_t> {
   public:
    proto::ColumnType type() const override { return proto::ColumnType::HASH; }
    std::string ToStringWithDigitPrecision(const row_t row,
                                           const proto::Column& col_spec,
                                           int digit_precision) const override {
      return absl::StrCat(values()[row]);
    }

    bool IsNa(const row_t row) const override {
      return values()[row] == kNaValue;
    }

    void AddNA() override { Add(kNaValue); }

    void SetNA(const row_t row) override {
      (*mutable_values())[row] = kNaValue;
    }

    void AddFromExample(const proto::Example::Attribute& attribute) override;

    void Set(row_t example_idx, uint64_t value);

    void Set(row_t example_idx,
             const proto::Example::Attribute& attribute) override;

    void ExtractExample(row_t example_idx,
                        proto::Example::Attribute* attribute) const override;

    absl::Status ConvertToGivenDataspec(
        AbstractColumn* dst, const proto::Column& src_spec,
        const proto::Column& dst_spec) const override;

    static constexpr uint64_t kNaValue = 1;
  };

  VerticalDataset() {}
  VerticalDataset& operator=(VerticalDataset&&) = default;
  VerticalDataset(VerticalDataset&&) = default;

  // Number of rows in the dataset.
  row_t nrow() const { return nrow_; }

  // Set the number of rows.
  void set_nrow(const row_t nrow) { nrow_ = nrow; }

  // Number of columns in the dataset.
  int ncol() const { return columns_.size(); }

  // Const access to columns.
  const AbstractColumn* column(int col) const {
    DCHECK(columns_[col].column);
    return columns_[col].column;
  }

  // Mutable access to columns.
  AbstractColumn* mutable_column(int col) {
    DCHECK(columns_[col].owned_column);
    return columns_[col].owned_column.get();
  }

  // Retrieve and cast a column to the specified class. Fails and prints an
  // error message if T is different from the column type.
  //
  // The ownership of the column array is not transferred, don't delete the
  // column.
  template <typename T>
  const T* ColumnWithCast(int col) const;

  // Similar to "ColumnWithCast", but won't check if the column is of the right
  // type.
  //
  // The ownership of the column array is not transferred, don't delete the
  // column.
  template <typename T>
  const T* ColumnWithCastNoCheck(int col) const;

  // Retrieve and cast a column to the specified class. Returns a nullptr  if
  // the type is not compatible.
  template <typename T>
  const T* ColumnWithCastOrNull(int col) const;

  // Retrieve and cast a column to the specified class. Fails if the type is not
  // compatible.
  template <typename T>
  T* MutableColumnWithCast(int col);

  // Retrieve and cast a column to the specified class. Returns a nullptr  if
  // the type is not compatible.
  template <typename T>
  T* MutableColumnWithCastOrNull(int col);

  const proto::DataSpecification& data_spec() const { return data_spec_; }

  proto::DataSpecification* mutable_data_spec() { return &data_spec_; }

  void set_data_spec(const proto::DataSpecification& data_spec) {
    data_spec_ = data_spec;
  }

  // Return the index of a column from its name. If no such column exists,
  // returns -1.
  int ColumnNameToColumnIdx(absl::string_view name) const;

  // Extract a subset of observations from the dataset.
  utils::StatusOr<VerticalDataset> Extract(
      const std::vector<row_t>& indices) const;

  // Copy the dataset while changing its dataspec.
  // When a column is present in the new dataspec but not in the old one:
  //   If this column is present in "required_column_idxs", an error is raised.
  //   If this column is not present in "required_column_idxs", the column is
  //   created and filled with "NA" values.
  utils::StatusOr<VerticalDataset> ConvertToGivenDataspec(
      const proto::DataSpecification& new_data_spec,
      const std::vector<int>& required_column_idxs) const;

  // Add the content of the "src" dataset at the end of this dataset.
  absl::Status Append(const VerticalDataset& src);
  // Add a subset of "src" at the end of this dataset.
  absl::Status Append(const VerticalDataset& src,
                      const std::vector<row_t>& indices);

  // Create the columns of the dataset from the columns specified in the
  // dataspec. This functions should be used if the dataspec was created
  // directly (i.e. using mutable_data_spec), instead of using "AddColumn".
  absl::Status CreateColumnsFromDataspec();

  // Add and initialize a new column to the dataspec.
  utils::StatusOr<AbstractColumn*> AddColumn(const proto::Column& column_spec);

  utils::StatusOr<proto::Column*> AddColumn(const absl::string_view name,
                                            const proto::ColumnType type);

  // Similar to "AddColumn", but replace an existing column.
  utils::StatusOr<AbstractColumn*> ReplaceColumn(
      int column_idx, const proto::Column& column_spec);

  // Add a new column to the list of columns. The dataset takes ownership of the
  // column. Does not update the dataspec.
  void PushBackOwnedColumn(std::unique_ptr<AbstractColumn>&& column);

  // Add a new column to the list of columns. The dataset does not take
  // ownership of the dataset. "column" should NOT be destroyed. Does not update
  // the dataspec.
  void PushBackNotOwnedColumn(const AbstractColumn* column);

  // Test if the column is owned by the dataset.
  bool OwnsColumn(int col) const;

  // Append a new example to the dataset.
  // If "load_columns" is set, only the columns specified in it will be loaded.
  void AppendExample(const proto::Example& example,
                     const absl::optional<std::vector<int>> load_columns = {});
  void AppendExample(
      const std::unordered_map<std::string, std::string>& example);

  // Create a shallow copy of the dataset. The created dataset does not get
  // ownership of the columns.
  VerticalDataset ShallowNonOwningClone() const;

  // Extract an example from the dataset.
  void ExtractExample(row_t example_idx, proto::Example* example) const;

  std::string ValueToString(row_t row, int col) const;

  void Set(row_t row, int col, const proto::Example::Attribute& value);

  // Reserves the memory for "num_rows" examples on each existing columns.
  // It is not required to reserve the memory, but it can speed-up the code
  // (similarly to std::vector:reserve).m
  void Reserve(const row_t num_rows,
               const absl::optional<std::vector<int>>& load_columns = {});

  // Generates a human readable summary of the memory.
  std::string MemorySummary() const;

  // Release the reserved but not used memory.
  //
  // Can be called on a dataset that won't receive new elements.
  // Calls "shrink_to_fit" on the std::vectors.
  void ShrinkToFit();

 private:
  struct ColumnContainer {
    // A column can either be owned or not-owned by the VerticalDataset.
    //
    // If the column is owned: "owned_column" and "column" are pointing to the
    // (same) column (owned by "owned_column"). If the column is not owned:
    // "column" points on the data and "owned_column" is not set.

    const AbstractColumn* column;
    std::unique_ptr<AbstractColumn> owned_column;

    ColumnContainer& operator=(ColumnContainer&&) = default;
    ColumnContainer(ColumnContainer&&) = default;
  };

  // Columns.
  std::vector<ColumnContainer> columns_;

  // Number of records in the dataset.
  row_t nrow_ = 0;

  // Dataspec
  proto::DataSpecification data_spec_;
};

// Converts a map of "column name -> value" into a proto::Example. Each of the
// key of "src" should be a valid column name in "data_spec". In "src",
// values are stored as string (independently of their true semantic) and are
// parsed similarly as a CSV field.
void MapExampleToProtoExample(
    const std::unordered_map<std::string, std::string>& src,
    const proto::DataSpecification& data_spec, proto::Example* dst);

// Reverse of "MapExampleToProtoExample". Convert an example into a map of
// example representations.
utils::StatusOr<std::unordered_map<std::string, std::string>>
ProtoExampleToMapExample(const proto::Example& src,
                         const proto::DataSpecification& data_spec);

// Converts an examples from one dataspec to another.
proto::Example ConvertExampleToGivenDataspec(
    const proto::Example& src, const proto::DataSpecification& src_data_spec,
    const proto::DataSpecification& dst_data_spec);

// Test if "a" is a sub-dataspec of "b". A dataspec "a" if a sub-dataspec of
// dataspec "b" if an example in "b" can be converted into an example in "a". If
// not (i.e. return false), "reason" describes the reason.
bool IsValidSubDataspec(const proto::DataSpecification& a,
                        const proto::DataSpecification& b, std::string* reason);

template <typename T>
void VerticalDataset::TemplateScalarStorage<T>::ExtractAndAppend(
    const std::vector<row_t>& indices, AbstractColumn* dst) const {
  auto* cast_dst =
      dynamic_cast<VerticalDataset::TemplateScalarStorage<T>*>(dst);
  CHECK(cast_dst != nullptr);
  if (values_.empty() && !indices.empty()) {
    LOG(FATAL) << "Trying to extract " << indices.size()
               << " examples from the non-allocated column \"" << name()
               << "\".";
  }
  const size_t indices_size = indices.size();
  const size_t init_dst_nrows = dst->nrows();
  cast_dst->Resize(init_dst_nrows + indices_size);
  for (size_t new_idx = 0; new_idx < indices_size; new_idx++) {
    const auto src_row_idx = indices[new_idx];
    const auto dst_row_idx = new_idx + init_dst_nrows;
    DCHECK_LT(src_row_idx, values_.size());
    if (!IsNa(src_row_idx)) {
      cast_dst->values_[dst_row_idx] = values_[src_row_idx];
    } else {
      cast_dst->SetNA(dst_row_idx);
    }
  }
}

template <typename T>
void VerticalDataset::TemplateMultiValueStorage<T>::ExtractAndAppend(
    const std::vector<row_t>& indices, AbstractColumn* dst) const {
  auto* cast_dst =
      dynamic_cast<VerticalDataset::TemplateMultiValueStorage<T>*>(dst);
  CHECK(cast_dst != nullptr);
  if (values_.empty() && !indices.empty()) {
    LOG(FATAL) << "ExtractAndAppend on an empty column";
  }
  cast_dst->Reserve(dst->nrows() + indices.size());
  for (const auto row_idx : indices) {
    DCHECK_LT(row_idx, values_.size());
    if (!IsNa(row_idx)) {
      cast_dst->Add(bank_.begin() + values_[row_idx].first,
                    bank_.begin() + values_[row_idx].second);
    } else {
      cast_dst->AddNA();
    }
  }
}

template <typename T>
const T* VerticalDataset::ColumnWithCast(int col) const {
  static_assert(std::is_base_of<AbstractColumn, T>::value,
                "The template class argument does not derive AbstractColumn.");
  const auto* abstract_column = column(col);
  const T* const casted_column = dynamic_cast<const T* const>(abstract_column);
  if (!casted_column) {
    LOG(FATAL) << "Column \"" << abstract_column->name() << "\"=" << col
               << " has type "
               << proto::ColumnType_Name(abstract_column->type())
               << " and is not compatible with type " << typeid(T).name();
  }
  return casted_column;
}

template <typename T>
const T* VerticalDataset::ColumnWithCastNoCheck(int col) const {
  static_assert(std::is_base_of<AbstractColumn, T>::value,
                "The template class argument does not derive AbstractColumn.");
  const auto* abstract_column = column(col);
  return static_cast<const T* const>(abstract_column);
}

template <typename T>
const T* VerticalDataset::ColumnWithCastOrNull(int col) const {
  static_assert(std::is_base_of<AbstractColumn, T>::value,
                "The template class argument does not derive AbstractColumn.");
  const auto* abstract_column = column(col);
  return dynamic_cast<const T*>(abstract_column);
}

template <typename T>
T* VerticalDataset::MutableColumnWithCast(int col) {
  static_assert(std::is_base_of<AbstractColumn, T>::value,
                "The template class argument does not derive  AbstractColumn.");
  auto* abstract_column = mutable_column(col);
  T* const casted_column = dynamic_cast<T* const>(abstract_column);
  if (!casted_column) {
    LOG(FATAL) << "Column \"" << abstract_column->name() << "\"=" << col
               << " has type "
               << proto::ColumnType_Name(abstract_column->type())
               << " and is not compatible with type " << typeid(T).name();
  }
  return casted_column;
}

template <typename T>
T* VerticalDataset::MutableColumnWithCastOrNull(int col) {
  static_assert(std::is_base_of<AbstractColumn, T>::value,
                "The template class argument does not derive  AbstractColumn.");
  auto* abstract_column = mutable_column(col);
  return dynamic_cast<T* const>(abstract_column);
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_DATASET_VERTICAL_DATASET_H_
