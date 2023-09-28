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

// This file introduces the "ExampleSets". An example set is an efficient
// structure containing a set of examples, and organized for fast inference.
//
// Given a model, the correspond example set format is obtained with
// ModelClass::ExampleSet. Note: Don't instantiate one of the ExampleSet
// directly as the ExampleSet of a given model is likely to change as inference
// codes are improved and specialized.
//
// Usage example:
//
//   // Initialize.
//   std::unique_ptr<AbstractModel> abstract_model = ...;
//   GradientBoostedTreesBinaryClassificationQuickScorerExtended model;
//   GenericToSpecializedModel(abstract_model, &model);
//   auto feature_1 = model.GetNumericalFeatureId("feature_1");
//   auto feature_2 = model.GetCategoricalFeatureId("feature_2");
//   auto feature_3 = model.GetNumericalFeatureId("feature_3");
//
//   // Allocate 5 examples.
//   GradientBoostedTreesBinaryClassificationQuickScorerExtended::ExampleSet
//   examples(5); examples.FillMissing(model);
//
//   // Set one examples and run the model.
//   int example_idx = 0;
//   examples.SetNumerical(example_idx, feature_1, 1.f, model);
//   examples.SetCategorical(example_idx, feature_2, "hello", model);
//   examples.SetMissingNumerical(example_idx, feature_3, model);
//
//   std::vector<float> predictions;
//   Predict(model, examples, /*num_examples=*/ 1, &predictions);
//
//   // Configure two examples and run the model.
//   examples.Clear();
//   example_idx = 0;
//   examples.SetNumerical(example_idx, feature_1, 1.f, model);
//   examples.SetCategorical(example_idx, feature_2, "hello", model);
//   examples.SetMissingNumerical(example_idx, feature_3, model);
//   example_idx = 1;
//   examples.SetNumerical(example_idx, feature_1, 1.f, model);
//   examples.SetCategorical(example_idx, feature_2, "hello", model);
//   examples.SetMissingNumerical(example_idx, feature_3, model);
//   Predict(model, examples, /*num_examples=*/ 2, &predictions);
//
#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_EXAMPLE_SET_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_EXAMPLE_SET_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace serving {

class AbstractExampleSet;

using dataset::proto::DataSpecification;

// Input feature value that can either be a numerical of a categorical value.
struct NumericalOrCategoricalValue {
  union {
    float numerical_value;
    int categorical_value;
    // A boolean value is threated as an alias of numerical.
    // It is stored as a numerical_value with values 0.0 and 1.0.
  };

  static NumericalOrCategoricalValue Numerical(const float value) {
    NumericalOrCategoricalValue item;
    item.numerical_value = value;
    return item;
  }

  static NumericalOrCategoricalValue Categorical(const int value) {
    NumericalOrCategoricalValue item;
    item.categorical_value = value;
    return item;
  }

  static NumericalOrCategoricalValue Boolean(const bool value) {
    NumericalOrCategoricalValue item;
    item.numerical_value = value ? 1.f : 0.f;
    return item;
  }

  inline bool operator==(const NumericalOrCategoricalValue& other) const {
    // Note: Works because float and ints take the same number of bytes.
    return categorical_value == other.categorical_value;
  }
};

// Gets the value that represent a "missing value" for an example set / an
// inference engine.
//
// For categorical* features, values are replaced by the most frequent value.
// This behavior is compatible with models trained with global imputation.
//
// For numerical features, if missing_numerical_is_na=False, values are replaced
// by the value mean as defined in the dataspec. This behavior is compatible
// with models trained with global imputation.
//
// For numerical features, if missing_numerical_is_na=True, values are replaced
// by NaN.
//
template <typename Value>
absl::StatusOr<Value> GetDefaultValue(const dataset::proto::Column& col_spec,
                                      bool missing_numerical_is_na);

// How is a batch of examples stored in memory.
enum class ExampleFormat {
  // A batch of example stored as example-major feature-minor.
  // i.e. a[i+j*num_example] is the j-th feature of the i-th example.
  FORMAT_EXAMPLE_MAJOR,

  // A batch of example stored as example-minor feature-major.
  // i.e. a[i*num_features+j] is the j-th feature of the i-th example.
  FORMAT_FEATURE_MAJOR,
};

constexpr int kMissingCategoricalSetValue = -1;

// Range of values in an internal buffer.
struct Rangei32 {
  int32_t begin;
  int32_t end;
};
constexpr Rangei32 kUnitBufferRange = {/*.begin =*/0, /*.end =*/1};

struct FeatureDef {
  // Name of a feature.
  std::string name;
  // Type of a feature.
  dataset::proto::ColumnType type;
  // Index of the feature in the dataspec.
  int spec_idx;
  // Internal index of the feature.
  // Note: Two separate features (e.g. different spec_idx) can have the same
  // internal index.
  int internal_idx;
};

// Definition about the unstacked features (accessible as multi dimensional
// features).
struct UnstackedFeature {
  // Internal index of the first unstacked feature.
  int begin_internal_idx;
  // Dataspec column index of the first unstacked feature.
  int begin_spec_idx;
  // Dimension of the original feature i.e. number of unstacked features.
  int size;
  // Index of this struct in "unstacked_features_" i.e. the dense index of
  // unstacked feature used by the model.
  int unstacked_index;
};

std::ostream& operator<<(std::ostream& os, const FeatureDef& feature);

absl::StatusOr<FeatureDef> FindFeatureDef(const std::vector<FeatureDef>& defs,
                                          int spec_feature_idx);

const FeatureDef* FindFeatureDefFromInternalIndex(
    const std::vector<FeatureDef>& defs, int internal_index);

std::vector<std::string> FeatureNames(const std::vector<FeatureDef>& defs);

// Definition of the input features of a model. Used by an ExampleSet.
class FeaturesDefinitionNumericalOrCategoricalFlat {
 public:
  struct NumericalFeatureId {
    int index;
  };
  // A boolean is an alias of NumericalFeatureId. However, code should not rely
  // on it.
  typedef NumericalFeatureId BooleanFeatureId;
  struct CategoricalFeatureId {
    int index;
  };
  struct CategoricalSetFeatureId {
    int index;
  };
  struct MultiDimNumericalFeatureId {
    int index;
  };

  // Gets the feature def of a feature from its name.
  // Returns an invalid status if the feature was not found or if the feature
  // is not used by the model as input.
  absl::StatusOr<const FeatureDef*> FindFeatureDefByName(
      const absl::string_view name) const {
    auto cached_feature_it = feature_def_cache_.find(std::string(name));
    if (cached_feature_it != feature_def_cache_.end()) {
      // The feature was found.
      return cached_feature_it->second;
    }

    // Test if the feature is in the dataspec i.e. the feature was provided
    // during training but ultimately not used by the model.
    bool feature_in_dataspec = false;
    for (const auto& column : data_spec_.columns()) {
      if (column.name() == name) {
        feature_in_dataspec = true;
        break;
      }
    }

    std::string error_snippet;
    if (feature_in_dataspec) {
      absl::SubstituteAndAppend(
          &error_snippet,
          " The column \"$0\" is present in the dataspec but it is not used by "
          "the model (e.g. feature ignored as non-interesting filtered-out "
          "by the training configuration). Use "
          "\"model.features().HasInputFeature()\" or "
          "\"model.features().input_features()\" to check and list the input "
          "features of the model.",
          name);
    }

    return absl::InvalidArgumentError(absl::Substitute(
        "Unknown input feature \"$0\".$1", name, error_snippet));
  }

  // Gets the unstacked feature definition from its name.
  absl::StatusOr<const UnstackedFeature*> FindUnstackedFeatureDefByName(
      const absl::string_view name) const {
    auto it_index = indexed_unstacked_features_.find(std::string(name));
    if (it_index == indexed_unstacked_features_.end()) {
      return absl::InvalidArgumentError(
          absl::Substitute("Unknown unstacked feature $0", name));
    }
    return &unstacked_features_[it_index->second];
  }

  // Get the identifier of a numerical feature.
  absl::StatusOr<NumericalFeatureId> GetNumericalFeatureId(
      const absl::string_view name) const {
    ASSIGN_OR_RETURN(const auto* def, FindFeatureDefByName(name));
    if (def->type != dataset::proto::ColumnType::NUMERICAL &&
        def->type != dataset::proto::ColumnType::DISCRETIZED_NUMERICAL &&
        def->type != dataset::proto::ColumnType::BOOLEAN) {
      return absl::InvalidArgumentError(
          absl::Substitute("Feature $0 is not numerical", name));
    }
    return NumericalFeatureId{def->internal_idx};
  }

  absl::StatusOr<NumericalFeatureId> GetNumericalFeatureId(
      const int feature_spec_idx) const {
    ASSIGN_OR_RETURN(const auto def,
                     FindFeatureDef(fixed_length_features(), feature_spec_idx));
    if (def.type != dataset::proto::ColumnType::NUMERICAL &&
        def.type != dataset::proto::ColumnType::DISCRETIZED_NUMERICAL &&
        def.type != dataset::proto::ColumnType::BOOLEAN) {
      return absl::InvalidArgumentError(
          absl::Substitute("Feature $0 is not numerical", feature_spec_idx));
    }
    return NumericalFeatureId{def.internal_idx};
  }

  absl::StatusOr<BooleanFeatureId> GetBooleanFeatureId(
      const absl::string_view name) const {
    return GetNumericalFeatureId(name);
  }

  absl::StatusOr<BooleanFeatureId> GetBooleanFeatureId(
      const int feature_spec_idx) const {
    return GetNumericalFeatureId(feature_spec_idx);
  }

  // Get the identifier of a categorical feature.
  absl::StatusOr<CategoricalFeatureId> GetCategoricalFeatureId(
      const absl::string_view name) const {
    ASSIGN_OR_RETURN(const auto* def, FindFeatureDefByName(name));
    if (def->type != dataset::proto::ColumnType::CATEGORICAL) {
      return absl::InvalidArgumentError(
          absl::Substitute("Feature $0 is not categorical", name));
    }
    return CategoricalFeatureId{def->internal_idx};
  }

  absl::StatusOr<CategoricalFeatureId> GetCategoricalFeatureId(
      const int feature_spec_idx) const {
    ASSIGN_OR_RETURN(const auto def,
                     FindFeatureDef(fixed_length_features(), feature_spec_idx));
    if (def.type != dataset::proto::ColumnType::CATEGORICAL) {
      return absl::InvalidArgumentError(
          absl::Substitute("Feature $0 is not categorical", feature_spec_idx));
    }
    return CategoricalFeatureId{def.internal_idx};
  }

  // Get the identifier of a categorical-set feature.
  absl::StatusOr<CategoricalSetFeatureId> GetCategoricalSetFeatureId(
      const absl::string_view name) const {
    ASSIGN_OR_RETURN(const auto* def, FindFeatureDefByName(name));
    return CategoricalSetFeatureId{def->internal_idx};
  }

  absl::StatusOr<CategoricalSetFeatureId> GetCategoricalSetFeatureId(
      const int feature_spec_idx) const {
    ASSIGN_OR_RETURN(const auto def, FindFeatureDef(categorical_set_features(),
                                                    feature_spec_idx));
    return CategoricalSetFeatureId{def.internal_idx};
  }

  // Get the identifier of a multi-dimensional numerical feature.
  absl::StatusOr<MultiDimNumericalFeatureId> GetMultiDimNumericalFeatureId(
      const absl::string_view name) const {
    // Get the unstacked feature information.
    const auto it_index = indexed_unstacked_features_.find(std::string(name));
    if (it_index == indexed_unstacked_features_.end()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Unknown feature %s", name));
    }
    const UnstackedFeature& unstack_def = unstacked_features_[it_index->second];
    const auto& begin_def =
        fixed_length_features_[unstack_def.begin_internal_idx];

    // Ensure the feature is numerical.
    if (begin_def.type != dataset::proto::ColumnType::NUMERICAL &&
        begin_def.type != dataset::proto::ColumnType::DISCRETIZED_NUMERICAL) {
      return absl::InvalidArgumentError(
          absl::Substitute("Feature $0 is not numerical", name));
    }

    return MultiDimNumericalFeatureId{it_index->second};
  }

  // List of the name and types of the input features of the model.
  std::vector<FeatureDef> input_features() const;

  const std::vector<UnstackedFeature>& unstacked_features() const;

  // Tests if an attributes is an input feature of the model.
  // If false (i.e. the feature is not used/supported by the model), the
  // functions "Get*FeatureId" will fail.
  bool HasInputFeature(absl::string_view name) const;

  const std::vector<FeatureDef>& fixed_length_features() const {
    return fixed_length_features_;
  }

  const std::vector<FeatureDef>& categorical_set_features() const {
    return categorical_set_features_;
  }

  // Specification of the features.
  const DataSpecification& data_spec() const { return data_spec_; }

  // Specification of the features.
  const std::vector<int>& column_input_features() const {
    return column_input_features_;
  }

  // Representation of missing values for fixed length features.
  const std::vector<NumericalOrCategoricalValue>&
  fixed_length_na_replacement_values() const {
    return fixed_length_feature_missing_values_;
  }

  // Initialize the object.
  absl::Status Initialize(const std::vector<int>& input_features,
                          const DataSpecification& dataspec,
                          bool missing_numerical_is_na = false);

 private:
  // Specialization of "Initialize" for "normal" features.
  absl::Status InitializeNormalFeatures(const std::vector<int>& input_features,
                                        const DataSpecification& dataspec,
                                        bool missing_numerical_is_na);

  // Specialization of "Initialize" for "unstacked" features.
  absl::Status InitializeUnstackedFeatures(
      const std::vector<int>& input_features, const DataSpecification& dataspec,
      const bool missing_numerical_is_na);

  // The name and order of the fixed length input features expected by the
  // model.
  std::vector<FeatureDef> fixed_length_features_;

  // The replacement value for non-available fixed length input features.
  // Specified in the same order as "fixed_length_features". Should have the
  // same size as "fixed_length_features".
  std::vector<NumericalOrCategoricalValue> fixed_length_feature_missing_values_;

  // The name and order of the categorical-set input features expected by the
  // model.
  std::vector<FeatureDef> categorical_set_features_;

  // Data specification.
  DataSpecification data_spec_;

  // Index to the "fixed_length_features_" and "categorical_set_features_" by
  // "name".
  std::unordered_map<std::string, const FeatureDef*> feature_def_cache_;

  // List of "unstacked" features (similar to "unstackeds" in the dataspec).
  std::vector<UnstackedFeature> unstacked_features_;

  // Index  "original name" to its index in "unstacked_features_".
  std::unordered_map<std::string, int> indexed_unstacked_features_;

  // Index of the columns in the dataspec used as input feature to the model.
  std::vector<int> column_input_features_;
};

using FeaturesDefinition = FeaturesDefinitionNumericalOrCategoricalFlat;

// TODO: Remove dependency to TF.
class AbstractExampleSet;

class AbstractExampleSet {
  // Set of examples for fast model inference.
  //
  // When possible, use a child class to avoid virtual call overhead.
  // AbstractExampleSet should only be used when the model type is not known
  // at compilation time.
  //
  // See "ExampleSetNumericalOrCategoricalFlat" for the definition of the
  // functions.
 public:
  virtual ~AbstractExampleSet() = default;

  virtual void SetNumerical(int example_idx,
                            FeaturesDefinition::NumericalFeatureId feature_id,
                            float value,
                            const FeaturesDefinition& features) = 0;

  virtual void SetBoolean(int example_idx,
                          FeaturesDefinition::BooleanFeatureId feature_id,
                          bool value, const FeaturesDefinition& features) = 0;

  virtual void SetCategorical(
      int example_idx, FeaturesDefinition::CategoricalFeatureId feature_id,
      int value, const FeaturesDefinition& features) = 0;

  virtual void SetCategorical(
      int example_idx, FeaturesDefinition::CategoricalFeatureId feature_id,
      const std::string& value, const FeaturesDefinition& features) = 0;

  virtual void SetCategoricalSet(
      int example_idx, FeaturesDefinition::CategoricalSetFeatureId feature_id,
      std::vector<int>::const_iterator value_begin,
      std::vector<int>::const_iterator value_end,
      const FeaturesDefinition& features) = 0;

  void SetCategoricalSet(
      const int example_idx,
      const FeaturesDefinition::CategoricalSetFeatureId feature_id,
      const std::vector<int>& values, const FeaturesDefinition& features) {
    SetCategoricalSet(example_idx, feature_id, values.cbegin(), values.cend(),
                      features);
  }

  virtual void SetCategoricalSet(
      int example_idx, FeaturesDefinition::CategoricalSetFeatureId feature_id,
      const std::vector<std::string>& values,
      const FeaturesDefinition& features) = 0;

  virtual absl::Status SetMultiDimNumerical(
      int example_idx,
      FeaturesDefinition::MultiDimNumericalFeatureId feature_id,
      absl::Span<const float> values, const FeaturesDefinition& features) = 0;

  virtual void SetMissingNumerical(
      int example_idx, FeaturesDefinition::NumericalFeatureId feature_id,
      const FeaturesDefinition& features) = 0;

  virtual void SetMissingBoolean(
      int example_idx, FeaturesDefinition::BooleanFeatureId feature_id,
      const FeaturesDefinition& features) = 0;

  virtual void SetMissingCategorical(
      int example_idx, FeaturesDefinition::CategoricalFeatureId feature_id,
      const FeaturesDefinition& features) = 0;

  virtual void SetMissingCategoricalSet(
      int example_idx, FeaturesDefinition::CategoricalSetFeatureId feature_id,
      const FeaturesDefinition& features) = 0;

  virtual void SetMissingMultiDimNumerical(
      int example_idx,
      FeaturesDefinition::MultiDimNumericalFeatureId feature_id,
      const FeaturesDefinition& features) = 0;

  virtual void FillMissing(const FeaturesDefinition& features) = 0;

  virtual absl::Status Copy(int64_t begin, int64_t end,
                            const FeaturesDefinition& features,
                            AbstractExampleSet* dst) const = 0;

  virtual absl::Status FromProtoExample(const dataset::proto::Example& src,
                                        const int example_idx,
                                        const FeaturesDefinition& features) = 0;

  virtual absl::StatusOr<dataset::proto::Example> ExtractProtoExample(
      const int example_idx, const FeaturesDefinition& features) const = 0;

  virtual void Clear() = 0;

  // Returns an approximation of the set's memory usage.
  virtual uint64_t MemoryUsage() const = 0;
};

// ExampleSet implementation where attribute (feature) values are stored either
// example or feature-wise (based on `format` parameter) in a compact vector.
template <typename Model, ExampleFormat format>
class ExampleSetNumericalOrCategoricalFlat : public AbstractExampleSet {
 public:
  ~ExampleSetNumericalOrCategoricalFlat() override = default;

  using FeaturesDefinition = FeaturesDefinitionNumericalOrCategoricalFlat;

  // Identifiers of the input features.

  using NumericalFeatureId = FeaturesDefinition::NumericalFeatureId;
  using BooleanFeatureId = FeaturesDefinition::BooleanFeatureId;
  using CategoricalFeatureId = FeaturesDefinition::CategoricalFeatureId;
  using CategoricalSetFeatureId = FeaturesDefinition::CategoricalSetFeatureId;
  using MultiDimNumericalFeatureId =
      FeaturesDefinition::MultiDimNumericalFeatureId;

  static constexpr auto kFormat = format;

  // Get the identifier of a numerical feature.
  static absl::StatusOr<NumericalFeatureId> GetNumericalFeatureId(
      const absl::string_view name, const Model& model) {
    return model.features().GetNumericalFeatureId(name);
  }

  static absl::StatusOr<NumericalFeatureId> GetNumericalFeatureId(
      const int feature_spec_idx, const Model& model) {
    return model.features().GetNumericalFeatureId(feature_spec_idx);
  }

  // Get the identifier of a boolean feature.
  static absl::StatusOr<BooleanFeatureId> GetBooleanFeatureId(
      const absl::string_view name, const Model& model) {
    return model.features().GetBooleanFeatureId(name);
  }

  static absl::StatusOr<BooleanFeatureId> GetBooleanFeatureId(
      const int feature_spec_idx, const Model& model) {
    return model.features().GetBooleanFeatureId(feature_spec_idx);
  }

  // Get the identifier of a categorical feature.
  static absl::StatusOr<CategoricalFeatureId> GetCategoricalFeatureId(
      const absl::string_view name, const Model& model) {
    return model.features().GetCategoricalFeatureId(name);
  }

  static absl::StatusOr<CategoricalFeatureId> GetCategoricalFeatureId(
      const int feature_spec_idx, const Model& model) {
    return model.features().GetCategoricalFeatureId(feature_spec_idx);
  }

  // Get the identifier of a categorical-set feature.
  static absl::StatusOr<CategoricalSetFeatureId> GetCategoricalSetFeatureId(
      const absl::string_view name, const Model& model) {
    return model.features().GetCategoricalSetFeatureId(name);
  }

  static absl::StatusOr<CategoricalSetFeatureId> GetCategoricalSetFeatureId(
      const int feature_spec_idx, const Model& model) {
    return model.features().GetCategoricalSetFeatureId(feature_spec_idx);
  }

  // Get the identifier of a multi-dimensional numerical feature.
  static absl::StatusOr<MultiDimNumericalFeatureId>
  GetMultiDimNumericalFeatureId(const absl::string_view name,
                                const Model& model) {
    return model.features().GetMultiDimNumericalFeatureId(name);
  }

  // Tests if the model requires this feature.
  static bool HasInputFeature(const absl::string_view name,
                              const Model& model) {
    return model.features().HasInputFeature(name);
  }

  // Allocate an example set.
  ExampleSetNumericalOrCategoricalFlat(const int num_examples,
                                       const Model& model)
      : fixed_length_features_(num_examples *
                               model.features().fixed_length_features().size()),
        num_examples_(num_examples),
        categorical_set_begins_and_ends_(
            num_examples * model.features().categorical_set_features().size()) {
  }

  // Empty the content of the example set. The example set still contains
  // "num_examples" examples but their values are undefined. This function
  // should be called before an ExampleSet object is re-used.
  void Clear() override {
    // The format does not need any clearing.
    categorical_item_buffer_.clear();
  }

  // Set the value of a numerical feature.
  void SetNumerical(const int example_idx, const NumericalFeatureId feature_id,
                    const float value,
                    const FeaturesDefinition& features) override {
    fixed_length_features_[FixedLengthIndex(example_idx, feature_id.index,
                                            features)]
        .numerical_value = value;
  }
  void SetNumerical(const int example_idx, const NumericalFeatureId feature_id,
                    const float value, const Model& model) {
    SetNumerical(example_idx, feature_id, value, model.features());
  }

  // Set the value of a boolean feature.
  // Note: Boolean features can also be provided as numerical features.
  void SetBoolean(const int example_idx, const BooleanFeatureId feature_id,
                  const bool value,
                  const FeaturesDefinition& features) override {
    SetNumerical(example_idx, NumericalFeatureId{feature_id.index},
                 value ? 1.f : 0.f, features);
  }

  void SetBoolean(const int example_idx, const NumericalFeatureId feature_id,
                  const bool value, const Model& model) {
    SetBoolean(example_idx, feature_id, value, model.features());
  }

  // Get the value of a numerical feature.
  float GetNumerical(const int example_idx, const NumericalFeatureId feature_id,
                     const Model& model) const {
    return fixed_length_features_[FixedLengthIndex(example_idx,
                                                   feature_id.index,
                                                   model.features())]
        .numerical_value;
  }

  float GetBoolean(const int example_idx, const BooleanFeatureId feature_id,
                   const Model& model) const {
    return GetNumerical(example_idx, NumericalFeatureId{feature_id.index},
                        model) >= 0.5f;
  }

  // Set the value of an integer categorical feature.
  void SetCategorical(const int example_idx,
                      const CategoricalFeatureId feature_id, const int value,
                      const FeaturesDefinition& features) override {
#ifndef NDEBUG
    const auto* feature = FindFeatureDefFromInternalIndex(
        features.fixed_length_features(), feature_id.index);
    DCHECK(feature != nullptr);
    const auto& spec =
        features.data_spec().columns(feature->spec_idx).categorical();
    DCHECK_GE(value, -1) << "The categorical integer feature \""
                         << feature->name << "\" should be in [-1,"
                         << spec.number_of_unique_values() << "). Got "
                         << value;
    DCHECK_LT(value, spec.number_of_unique_values())
        << "The categorical integer feature \"" << feature->name
        << "\" should be in [-1," << spec.number_of_unique_values() << "). Got "
        << value;
#endif

    fixed_length_features_[FixedLengthIndex(example_idx, feature_id.index,
                                            features)]
        .categorical_value = value;
  }
  void SetCategorical(const int example_idx,
                      const CategoricalFeatureId feature_id, const int value,
                      const Model& model) {
    SetCategorical(example_idx, feature_id, value, model.features());
  }

  // Get the value of a categorical feature.
  int GetCategoricalInt(const int example_idx,
                        const CategoricalFeatureId feature_id,
                        const Model& model) const {
    return fixed_length_features_[FixedLengthIndex(example_idx,
                                                   feature_id.index,
                                                   model.features())]
        .categorical_value;
  }

  // Set the value of a string categorical feature.
  void SetCategorical(const int example_idx,
                      const CategoricalFeatureId feature_id,
                      const std::string& value,
                      const FeaturesDefinition& features) override {
#ifndef NDEBUG
    const auto* feature = FindFeatureDefFromInternalIndex(
        features.fixed_length_features(), feature_id.index);
    DCHECK(feature != nullptr);
    const auto& spec =
        features.data_spec().columns(feature->spec_idx).categorical();
    DCHECK(!spec.is_already_integerized())
        << "The categorical feature \"" << feature->name
        << "\" should be passed as an integer";
#endif

    const auto int_value = dataset::CategoricalStringToValueWithStatus(
        value,
        features.data_spec().columns(
            features.fixed_length_features()[feature_id.index].spec_idx));
    if (int_value.ok()) {
      SetCategorical(example_idx, feature_id, int_value.value(), features);
    }
  }

  void SetCategorical(const int example_idx,
                      const CategoricalFeatureId feature_id,
                      const std::string& value, const Model& model) {
    SetCategorical(example_idx, feature_id, value, model.features());
  }

  // Get the string representation of a categorical feature.
  std::string GetCategoricalString(const int example_idx,
                                   const CategoricalFeatureId feature_id,
                                   const Model& model) const {
    const auto value = GetCategoricalInt(example_idx, feature_id, model);
    const auto& spec = model.features().data_spec().columns(
        model.features().fixed_length_features()[feature_id.index].spec_idx);
    return dataset::CategoricalIdxToRepresentation(spec, value, false);
  }

  // Set the value of an integer categorical-set feature.

  void SetCategoricalSet(const int example_idx,
                         const CategoricalSetFeatureId feature_id,
                         const std::vector<int>::const_iterator value_begin,
                         const std::vector<int>::const_iterator value_end,
                         const FeaturesDefinition& features) override {
    auto& dst_range = categorical_set_begins_and_ends_[CategoricalSetIndex(
        example_idx, feature_id.index, features)];
    dst_range.begin = categorical_item_buffer_.size();
    categorical_item_buffer_.insert(categorical_item_buffer_.end(), value_begin,
                                    value_end);
    dst_range.end = categorical_item_buffer_.size();
  }

  void SetCategoricalSet(const int example_idx,
                         const CategoricalSetFeatureId feature_id,
                         const std::vector<int>::const_iterator value_begin,
                         const std::vector<int>::const_iterator value_end,
                         const Model& model) {
    SetCategoricalSet(example_idx, feature_id, value_begin, value_end,
                      model.features());
  }

  void SetCategoricalSet(
      const int example_idx,
      const FeaturesDefinition::CategoricalSetFeatureId feature_id,
      const std::vector<int>& values, const Model& model) {
    SetCategoricalSet(example_idx, feature_id, values.cbegin(), values.cend(),
                      model.features());
  }

  void SetCategoricalSet(const int example_idx,
                         const CategoricalSetFeatureId feature_id,
                         const std::vector<std::string>& values,
                         const FeaturesDefinition& features) override {
    auto& dst_range = categorical_set_begins_and_ends_[CategoricalSetIndex(
        example_idx, feature_id.index, features)];
    dst_range.begin = categorical_item_buffer_.size();
    for (const auto& value : values) {
      auto value_idx_or = dataset::CategoricalStringToValueWithStatus(
          value,
          features.data_spec().columns(
              features.categorical_set_features()[feature_id.index].spec_idx));

      if (value_idx_or.ok()) {
        categorical_item_buffer_.push_back(value_idx_or.value());
      }
    }
    dst_range.end = categorical_item_buffer_.size();
  }

  void SetCategoricalSet(const int example_idx,
                         const CategoricalSetFeatureId feature_id,
                         const std::vector<std::string>& values,
                         const Model& model) {
    SetCategoricalSet(example_idx, feature_id, values, model.features());
  }

  absl::Status SetMultiDimNumerical(
      int example_idx, MultiDimNumericalFeatureId feature_id,
      const absl::Span<const float> values,
      const FeaturesDefinition& features) override {
    const UnstackedFeature& unstack_def =
        features.unstacked_features()[feature_id.index];
    if (values.size() != unstack_def.size) {
      return absl::InvalidArgumentError("Wrong number of values.");
    }
    for (int dim_idx = 0; dim_idx < unstack_def.size; dim_idx++) {
      fixed_length_features_[FixedLengthIndex(
                                 example_idx,
                                 unstack_def.begin_internal_idx + dim_idx,
                                 features)]
          .numerical_value = values[dim_idx];
    }
    return absl::OkStatus();
  };

  absl::Status SetMultiDimNumerical(int example_idx,
                                    MultiDimNumericalFeatureId feature_id,
                                    const absl::Span<const float> values,
                                    const Model& model) {
    return SetMultiDimNumerical(example_idx, feature_id, values,
                                model.features());
  }

  // Set a missing value of a numerical feature.
  void SetMissingNumerical(const int example_idx,
                           const NumericalFeatureId feature_id,
                           const FeaturesDefinition& features) override {
    fixed_length_features_[FixedLengthIndex(example_idx, feature_id.index,
                                            features)] =
        features.fixed_length_na_replacement_values()[feature_id.index];
  }
  void SetMissingNumerical(const int example_idx,
                           const NumericalFeatureId feature_id,
                           const Model& model) {
    SetMissingNumerical(example_idx, feature_id, model.features());
  }

  // Set a missing value of a boolean feature.
  void SetMissingBoolean(const int example_idx,
                         const BooleanFeatureId feature_id,
                         const FeaturesDefinition& features) override {
    SetMissingNumerical(example_idx, NumericalFeatureId{feature_id.index},
                        features);
  }

  void SetMissingBoolean(const int example_idx,
                         const BooleanFeatureId feature_id,
                         const Model& model) {
    SetMissingBoolean(example_idx, feature_id, model.features());
  }

  // Set a missing value of a categorical feature.
  void SetMissingCategorical(const int example_idx,
                             const CategoricalFeatureId feature_id,
                             const FeaturesDefinition& features) override {
    fixed_length_features_[FixedLengthIndex(example_idx, feature_id.index,
                                            features)] =
        features.fixed_length_na_replacement_values()[feature_id.index];
  }
  void SetMissingCategorical(const int example_idx,
                             const CategoricalFeatureId feature_id,
                             const Model& model) {
    SetMissingCategorical(example_idx, feature_id, model.features());
  }

  // Set a missing value of a categorical-set feature.
  void SetMissingCategoricalSet(const int example_idx,
                                const CategoricalSetFeatureId feature_id,
                                const FeaturesDefinition& features) override {
    auto& dst_range = categorical_set_begins_and_ends_[CategoricalSetIndex(
        example_idx, feature_id.index, features)];
    dst_range.begin = categorical_item_buffer_.size();
    categorical_item_buffer_.push_back(kMissingCategoricalSetValue);
    dst_range.end = categorical_item_buffer_.size();
  }
  void SetMissingCategoricalSet(const int example_idx,
                                const CategoricalSetFeatureId feature_id,
                                const Model& model) {
    SetMissingCategoricalSet(example_idx, feature_id, model.features());
  }

  // Set a missing value of a multi-dimensional numerical feature.
  void SetMissingMultiDimNumerical(
      const int example_idx, const MultiDimNumericalFeatureId feature_id,
      const FeaturesDefinition& features) override {
    const UnstackedFeature& unstack_def =
        features.unstacked_features()[feature_id.index];
    for (int dim_idx = 0; dim_idx < unstack_def.size; dim_idx++) {
      fixed_length_features_[FixedLengthIndex(
          example_idx, unstack_def.begin_internal_idx + dim_idx, features)] =
          features.fixed_length_na_replacement_values()
              [unstack_def.begin_internal_idx + dim_idx];
    }
  }

  void SetMissingMultiDimNumerical(const int example_idx,
                                   const MultiDimNumericalFeatureId feature_id,
                                   const Model& model) {
    SetMissingMultiDimNumerical(example_idx, feature_id, model.features());
  }

  // Set all the features to be missing.
  void FillMissing(const FeaturesDefinition& features) override;
  void FillMissing(const Model& model) { FillMissing(model.features()); }

  // Copy a subset of examples (from "begin" [inclusive] to "end" [exclusive])
  // from "this" example set to "dst". "dst" should be allocated with at
  // least "end - begin" elements.
  absl::Status Copy(
      int64_t begin, int64_t end, const FeaturesDefinition& features,
      ExampleSetNumericalOrCategoricalFlat<Model, format>* dst) const;

  absl::Status Copy(
      int64_t begin, int64_t end, const Model& model,
      ExampleSetNumericalOrCategoricalFlat<Model, format>* dst) const {
    return Copy(begin, end, model.features(), dst);
  }

  absl::Status Copy(int64_t begin, int64_t end,
                    const FeaturesDefinition& features,
                    AbstractExampleSet* dst) const override {
    auto* casted_dst =
        dynamic_cast<ExampleSetNumericalOrCategoricalFlat<Model, format>*>(dst);
    if (casted_dst == nullptr) {
      return absl::InvalidArgumentError(
          "Cannot copy an ExampleSet to another ExampleSet of a different "
          "type.");
    }
    return Copy(begin, end, features, casted_dst);
  }

  // Number of examples in the example set.
  int NumberOfExamples() const { return num_examples_; }

  // Extracts a proto::Example from the ExampleSet. This method is designed for
  // debugging and should not be used for time sensitive operations.
  //
  // For speed reasons, the conversion data -> ExampleSet can be lossy (i.e. the
  // ExampleSet only keep what the model needs for the inference). Therefore,
  // the conversion ExampleSet to proto::Example is not perfectly accurate:
  //   - Cannot distinguish between missing feature and the missing replacement
  //     value for numerical and categorical features.
  absl::StatusOr<dataset::proto::Example> ExtractProtoExample(
      const int example_idx, const Model& model) const {
    return ExtractProtoExample(example_idx, model.features());
  }

  absl::StatusOr<dataset::proto::Example> ExtractProtoExample(
      const int example_idx, const FeaturesDefinition& features) const override;

  // Set the value of one example from a proto::Example.
  absl::Status FromProtoExample(const dataset::proto::Example& src,
                                int example_idx, const Model& model) {
    return FromProtoExample(src, example_idx, model.features());
  }

  absl::Status FromProtoExample(const dataset::proto::Example& src,
                                int example_idx,
                                const FeaturesDefinition& features) override;

  // The following three methods ("CategoricalAndNumericalValues",
  // "InternalCategoricalSetBeginAndEnds", and
  // "InternalCategoricalSetBeginAndEnds") allow the access to raw content in an
  // ExampleSet. These methods are available for backward compatibility, should
  // be avoided when possible, and will be removed after the serving API V1 is
  // removed.
  const std::vector<NumericalOrCategoricalValue>&
  InternalCategoricalAndNumericalValues() const {
    return fixed_length_features_;
  }

  const std::vector<Rangei32>& InternalCategoricalSetBeginAndEnds() const {
    return categorical_set_begins_and_ends_;
  }

  const std::vector<int32_t>& InternalCategoricalItemBuffer() const {
    return categorical_item_buffer_;
  }

  uint64_t MemoryUsage() const override {
    return sizeof(ExampleSetNumericalOrCategoricalFlat) +
           fixed_length_features_.capacity() *
               sizeof(NumericalOrCategoricalValue) +
           categorical_set_begins_and_ends_.capacity() * sizeof(Rangei32) +
           categorical_item_buffer_.capacity() * sizeof(int32_t);
  }

 private:
  // Index of a fixed-length feature in the fixed-length example buffer.
  size_t FixedLengthIndex(const int example_idx,
                          const int fixed_length_feature_idx,
                          const FeaturesDefinition& features) const {
    if constexpr (format == ExampleFormat::FORMAT_EXAMPLE_MAJOR) {
      return fixed_length_feature_idx +
             example_idx * features.fixed_length_features().size();
    } else if constexpr (format == ExampleFormat::FORMAT_FEATURE_MAJOR) {
      return example_idx +
             fixed_length_feature_idx * static_cast<size_t>(num_examples_);
    } else {
      static_assert(!utils::is_same_v<Model, Model>, "Unsupported format.");
    }
  }

  int CategoricalSetIndex(const int example_idx,
                          const int categorical_set_feature_idx,
                          const FeaturesDefinition& features) const {
    return example_idx + categorical_set_feature_idx * num_examples_;
  }

  // Storage for 32bits fixed length values (currently, numerical and
  // categorical) ordered according to the "format".
  std::vector<NumericalOrCategoricalValue> fixed_length_features_;

  // Number of allocated examples.
  int num_examples_;

  // Categorical-set values. "categorical_set_begins_and_ands_[i]" is the begin
  // and end index, in "categorical_item_buffer_", of the categorical-set
  // values for the "i-th" example.
  std::vector<Rangei32> categorical_set_begins_and_ends_;

  // Buffer of categorical values. Used to store categorical-set values.
  std::vector<int32_t> categorical_item_buffer_;
};  // namespace serving

// Empty model to use for unit testing.
struct EmptyModel {
  using ExampleSet =
      ExampleSetNumericalOrCategoricalFlat<EmptyModel,
                                           ExampleFormat::FORMAT_FEATURE_MAJOR>;

  absl::Status Initialize(const std::vector<int>& input_features,
                          const DataSpecification& dataspec) {
    return mutable_features()->Initialize(input_features, dataspec);
  }

  const ExampleSet::FeaturesDefinition& features() const {
    return intern_features;
  }

  ExampleSet::FeaturesDefinition* mutable_features() {
    return &intern_features;
  }

  ExampleSet::FeaturesDefinition intern_features;
};

// Extracts a set of examples from a vertical dataset.
absl::Status CopyVerticalDatasetToAbstractExampleSet(
    const dataset::VerticalDataset& dataset,
    const dataset::VerticalDataset::row_t begin_example_idx,
    const dataset::VerticalDataset::row_t end_example_idx,
    const FeaturesDefinition& features, AbstractExampleSet* examples);

// Converts a Vertical dataset into an example set.
template <typename Model>
typename absl::StatusOr<typename Model::ExampleSet> VerticalDatasetToExampleSet(
    const dataset::VerticalDataset& dataset, const Model& model) {
  typename Model::ExampleSet examples(dataset.nrow(), model);

  RETURN_IF_ERROR(CopyVerticalDatasetToAbstractExampleSet(
      dataset,
      /*begin_example_idx=*/0,
      /*end_example_idx=*/dataset.nrow(), model.features(), &examples));

  return examples;
}

template <typename Model, ExampleFormat format>
void ExampleSetNumericalOrCategoricalFlat<Model, format>::FillMissing(
    const FeaturesDefinition& features) {
  Clear();

  const auto num_fixed_features = features.fixed_length_features().size();
  for (int fixed_feature_idx = 0; fixed_feature_idx < num_fixed_features;
       fixed_feature_idx++) {
    for (int example_idx = 0; example_idx < num_examples_; example_idx++) {
      fixed_length_features_[FixedLengthIndex(example_idx, fixed_feature_idx,
                                              features)] =
          features.fixed_length_na_replacement_values()[fixed_feature_idx];
    }
  }

  // Add the value kMissingCategoricalSetValue (-1) as the first element
  // of the buffer, and set all categorical set ranges to [0,1).
  categorical_item_buffer_.assign(1, kMissingCategoricalSetValue);
  std::fill(categorical_set_begins_and_ends_.begin(),
            categorical_set_begins_and_ends_.end(), kUnitBufferRange);
}

template <typename Model, ExampleFormat format>
absl::Status ExampleSetNumericalOrCategoricalFlat<Model, format>::Copy(
    int64_t begin, int64_t end, const FeaturesDefinition& features,
    ExampleSetNumericalOrCategoricalFlat<Model, format>* dst) const {
  if (dst->NumberOfExamples() < end - begin) {
    return absl::OutOfRangeError(
        "The destination does not contain enough examples.");
  }
  dst->Clear();

  // Copy of the fixed-length features.
  if constexpr (format == ExampleFormat::FORMAT_EXAMPLE_MAJOR) {
    const auto num_features = features.fixed_length_features().size();
    std::copy(fixed_length_features_.begin() + begin * num_features,
              fixed_length_features_.begin() + end * num_features,
              dst->fixed_length_features_.begin());
  } else if constexpr (format == ExampleFormat::FORMAT_FEATURE_MAJOR) {
    for (const auto& feature : features.fixed_length_features()) {
      const auto it_src_feature = fixed_length_features_.begin() +
                                  feature.internal_idx * NumberOfExamples();
      std::copy(it_src_feature + begin, it_src_feature + end,
                dst->fixed_length_features_.begin() +
                    feature.internal_idx * dst->NumberOfExamples());
    }
  } else {
    static_assert(!std::is_same<Model, Model>::value, "Unsupported format.");
  }

  // Copy of the categorical-set features.
  for (const auto& feature : features.categorical_set_features()) {
    for (int64_t example_idx = begin; example_idx < end; example_idx++) {
      const auto& src_range =
          categorical_set_begins_and_ends_[example_idx +
                                           feature.internal_idx *
                                               NumberOfExamples()];
      dst->SetCategoricalSet(
          example_idx - begin, CategoricalSetFeatureId{feature.internal_idx},
          categorical_item_buffer_.begin() + src_range.begin,
          categorical_item_buffer_.begin() + src_range.end, features);
    }
  }

  return absl::OkStatus();
}

template <typename Model, ExampleFormat format>
absl::Status
ExampleSetNumericalOrCategoricalFlat<Model, format>::FromProtoExample(
    const dataset::proto::Example& src, const int example_idx,
    const FeaturesDefinition& features) {
  for (const auto& feature : features.input_features()) {
    const auto& attribute = src.attributes(feature.spec_idx);
    switch (feature.type) {
      case dataset::proto::ColumnType::NUMERICAL: {
        if (attribute.has_numerical()) {
          SetNumerical(example_idx, NumericalFeatureId{feature.internal_idx},
                       attribute.numerical(), features);
        } else {
          SetMissingNumerical(
              example_idx, NumericalFeatureId{feature.internal_idx}, features);
        }
      } break;

      case dataset::proto::ColumnType::BOOLEAN: {
        if (attribute.has_boolean()) {
          SetBoolean(example_idx, BooleanFeatureId{feature.internal_idx},
                     attribute.boolean(), features);
        } else {
          SetMissingBoolean(example_idx, BooleanFeatureId{feature.internal_idx},
                            features);
        }
      } break;

      case dataset::proto::ColumnType::DISCRETIZED_NUMERICAL: {
        if (attribute.has_discretized_numerical()) {
          ASSIGN_OR_RETURN(const auto value,
                           dataset::DiscretizedNumericalToNumerical(
                               features.data_spec().columns(feature.spec_idx),
                               attribute.discretized_numerical()));
          SetNumerical(example_idx, NumericalFeatureId{feature.internal_idx},
                       value, features);
        } else {
          SetMissingNumerical(
              example_idx, NumericalFeatureId{feature.internal_idx}, features);
        }
      } break;

      case dataset::proto::ColumnType::CATEGORICAL:
        if (attribute.has_categorical()) {
          SetCategorical(example_idx,
                         CategoricalFeatureId{feature.internal_idx},
                         attribute.categorical(), features);
        } else {
          SetMissingCategorical(example_idx,
                                CategoricalFeatureId{feature.internal_idx},
                                features);
        }
        break;

      case dataset::proto::ColumnType::CATEGORICAL_SET:
        if (attribute.has_categorical_set()) {
          const auto& values = attribute.categorical_set().values();
          std::vector<int> copy_values = {values.begin(), values.end()};
          SetCategoricalSet(example_idx,
                            CategoricalSetFeatureId{feature.internal_idx},
                            copy_values.cbegin(), copy_values.cend(), features);
        } else {
          SetMissingCategoricalSet(
              example_idx, CategoricalSetFeatureId{feature.internal_idx},
              features);
        }
        break;

      default:
        return absl::InvalidArgumentError("Non supported feature type.");
    }
  }

  return absl::OkStatus();
}

template <typename Model, ExampleFormat format>
absl::StatusOr<dataset::proto::Example>
ExampleSetNumericalOrCategoricalFlat<Model, format>::ExtractProtoExample(
    const int example_idx, const FeaturesDefinition& features) const {
  dataset::proto::Example example;
  // Allocate the example.
  const auto& data_spec = features.data_spec();
  example.mutable_attributes()->Reserve(data_spec.columns().size());
  for (int col_idx = 0; col_idx < data_spec.columns().size(); col_idx++) {
    example.add_attributes();
  }

  // Extract the value of "fixed length" features.
  for (const auto& feature_id : features.fixed_length_features()) {
    const auto buffer_idx =
        FixedLengthIndex(example_idx, feature_id.internal_idx, features);
    const auto& src_value = fixed_length_features_[buffer_idx];
    const auto na_value =
        features.fixed_length_na_replacement_values()[feature_id.internal_idx];
    if (src_value == na_value) {
      // Note: Actual value equal to the missing value replacement will be seen
      // as missing.
      continue;
    }
    switch (feature_id.type) {
      case dataset::proto::ColumnType::NUMERICAL:
        example.mutable_attributes(feature_id.spec_idx)
            ->set_numerical(src_value.numerical_value);
        break;
      case dataset::proto::ColumnType::BOOLEAN:
        example.mutable_attributes(feature_id.spec_idx)
            ->set_boolean(src_value.numerical_value >= 0.5f);
        break;
      case dataset::proto::ColumnType::DISCRETIZED_NUMERICAL:
        example.mutable_attributes(feature_id.spec_idx)
            ->set_discretized_numerical(
                dataset::NumericalToDiscretizedNumerical(
                    data_spec.columns(feature_id.spec_idx),
                    src_value.numerical_value));
        break;
      case dataset::proto::ColumnType::CATEGORICAL:
        example.mutable_attributes(feature_id.spec_idx)
            ->set_categorical(src_value.categorical_value);
        break;
      default:
        return absl::InvalidArgumentError(
            absl::Substitute("Feature's $0 type is invalid", feature_id.name));
    }
  }

  // Extract the example of categorical set features.
  for (const auto& feature_id : features.categorical_set_features()) {
    auto& dst = *example.mutable_attributes(feature_id.spec_idx)
                     ->mutable_categorical_set();
    const auto range = categorical_set_begins_and_ends_[CategoricalSetIndex(
        example_idx, feature_id.internal_idx, features)];
    for (int value_idx = range.begin; value_idx < range.end; value_idx++) {
      const auto value = categorical_item_buffer_[value_idx];
      if (value == kMissingCategoricalSetValue) {
        example.mutable_attributes(feature_id.spec_idx)
            ->clear_categorical_set();
        break;
      }
      dst.add_values(value);
    }
  }

  // Note: Add support for other feature types here when supported by the
  // ExampleSet.

  return example;
}

}  // namespace serving
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EXAMPLE_SET_H_
