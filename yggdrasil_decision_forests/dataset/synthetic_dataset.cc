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

// Generation of synthetic dataset suited for simpleML.
//
// Features
//   - Classification and regression.
//   - Numerical, categorical {str, int}, categorical-set {str,int}, and boolean
//     features.
//   - Conditional independence of the label and features e.g. patters such as
//     the label and features {F1} are independent under a condition
//     controlled by features {F2}.
//   - Deterministic generation (if using a fixed seed).
//   - Each type of feature contributes more-or-less equally to the label. This
//     impact can be see by looking at the variable importance of a correctly
//     configured learner.
//   - The contribution of a feature is indicated by its index. For example,
//     feature_0 contributes more than feature_1, that contributes more that
//     feature_2.
//
// Algorithm
//   - Each feature is attached to one of few (e.g. 5) accumulators.
//   - The value of an accumulator (for a given example) is defined as the
//     weighted sum of the features attached to it. For categorical and
//     categorical-set features, a random (or fixed) numerical value is attached
//     to each element in the dictionary.
//   - While a feature is used in an accumulator, it can be randomly masked from
//     being in the final examples to simulate missing values.
//   - For a given example, the i-th accumulator rank is the rank of accumulator
//     compared to all the others examples. This rank is stored in a
//     "noisy_accumulator_rank" variable to indicate that noise is added to this
//     rank computation.
//   - One of the accumulator rank define the label value. The label is directly
//     the rank in the case of regression. For classification, each class is
//     assigned a contiguous set of rank value e.g. ranks{1..4}->Class1,
//     ranks{4..7}->Class2, etc. For ranking, the relevance is the label^2 with
//     a random bias and scale constant within each group.
//   - The selection of the accumulator used for the label is defined by a
//     binary decision tree using the other accumulator ranks.
//
#include "yggdrasil_decision_forests/dataset/synthetic_dataset.h"

#include <random>

#include "absl/strings/str_replace.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "yggdrasil_decision_forests/dataset/formats.h"
#include "yggdrasil_decision_forests/dataset/formats.pb.h"
#include "yggdrasil_decision_forests/dataset/tf_example_io_interface.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/csv.h"
#include "yggdrasil_decision_forests/utils/filesystem.h"
#include "yggdrasil_decision_forests/utils/hash.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace dataset {
namespace {

// Short string representation by type.
constexpr char kTypeNumerical[] = "num";
constexpr char kTypeBoolean[] = "bool";
constexpr char kTypeCategoricalStr[] = "cat_str";
constexpr char kTypeCategoricalInt[] = "cat_int";
constexpr char kTypeCategoricalSetStr[] = "cat_set_str";
constexpr char kTypeCategoricalSetInt[] = "cat_set_int";
constexpr char kTypeMultidimensionalNumerical[] = "multidimensional_num";

// Scaling applied to numerical values stored as integer.
constexpr int kIntNumericalScale = 100;

// Example in creation.
struct Example {
  // Example that will be serialized ultimately.
  tensorflow::Example tf_example;

  // Numerical accumulators, each defined as a weighted sum of a random subset
  // of features values (with or without an activation function).
  std::vector<double> accumulators;

  // Rank of the "accumulators" compared to the other examples.
  //
  // For example, without noise, noisy_accumulator_ranks[i] is the number of
  // examples with accumulators[i] smaller that this object accumulators[i].
  std::vector<int> noisy_accumulator_ranks;

  // Group of the example. Only used in ranking. -1 iif. not used.
  int group_idx = -1;
};

// Internal information managed by the generator.
struct GeneratorState {
  // Accumulator where each feature contributes.
  //
  // For example, the i-th numerical feature contributes to the
  // "accumulators[numerical_accumulator_idxs[i]]" accumulator.
  std::vector<int> numerical_accumulator_idxs;
  std::vector<int> categorical_str_accumulator_idxs;
  std::vector<int> categorical_int_accumulator_idxs;
  std::vector<int> categorical_set_str_accumulator_idxs;
  std::vector<int> categorical_set_int_accumulator_idxs;
  std::vector<int> boolean_accumulator_idxs;
  std::vector<int> multidimensional_numerical_accumulator_idxs;
};

// Creates and initializes the random generator.
utils::RandomEngine CreateRandomGenerator(
    const proto::SyntheticDatasetOptions& options) {
  utils::RandomEngine rnd;
  if (options.seed() == -1) {
    std::random_device initializer;
    rnd.seed(initializer());
  } else {
    rnd.seed(options.seed());
  }
  return rnd;
}

// Hashing of an integer that is stable for two executions of the library.
uint64_t StableHash(uint64_t value) {
  return utils::hash::HashInt64ToUint64(value);
}

// Creates a decreasing weight for a list of features.
float DecreasingWeight(const int feature_idx, const int num_features) {
  CHECK_LT(feature_idx, num_features);
  return static_cast<float>(num_features - feature_idx) / num_features;
}

// Creates a symbol "{base}_{index}".
std::string Symbol(const absl::string_view base, const int index) {
  return absl::StrCat(base, "_", index);
}

// Name of a feature.
std::string FeatureName(const proto::SyntheticDatasetOptions& options,
                        const absl::string_view type, const int index) {
  return absl::StrReplaceAll(
      options.feature_name(),
      {{"{type}", type}, {"{index}", absl::StrCat(index)}});
}

tensorflow::Feature* GetFeature(const absl::string_view name,
                                Example* example) {
  return &(*example->tf_example.mutable_features()
                ->mutable_feature())[std::string(name)];
}

void SetNumericalFeature(const absl::string_view name, const float value,
                         Example* example) {
  GetFeature(name, example)->mutable_float_list()->add_value(value);
}

void SetCategoricalStringFeature(const absl::string_view name,
                                 const absl::string_view value,
                                 Example* example) {
  GetFeature(name, example)
      ->mutable_bytes_list()
      ->add_value(std::string(value));
}

void SetCategoricalIntFeature(const absl::string_view name, const int value,
                              Example* example) {
  GetFeature(name, example)->mutable_int64_list()->add_value(value);
}

void SetNumericalIntFeature(const absl::string_view name, const int value,
                            Example* example) {
  return SetCategoricalIntFeature(name, value, example);
}

void SetCategoricalSetStringFeature(const absl::string_view name,
                                    const std::vector<int>& values,
                                    Example* example) {
  auto& dst = *GetFeature(name, example)->mutable_bytes_list();
  for (const auto value : values) {
    dst.add_value(Symbol("V", value));
  }
}

void SetCategoricalSetIntFeature(const absl::string_view name,
                                 const std::vector<int>& values,
                                 Example* example) {
  auto& dst = *GetFeature(name, example)->mutable_int64_list();
  for (const auto value : values) {
    dst.add_value(value);
  }
}

void AddNumericalFeatures(const proto::SyntheticDatasetOptions& options,
                          const GeneratorState& state, Example* example,
                          utils::RandomEngine* rnd) {
  auto uniform = std::uniform_real_distribution<float>();
  for (int feature_idx = 0; feature_idx < options.num_numerical();
       feature_idx++) {
    // Generate a random value.
    const float value = uniform(*rnd);

    // Add to accumulator.
    const float weight = DecreasingWeight(feature_idx, options.num_numerical());
    example->accumulators[state.numerical_accumulator_idxs[feature_idx]] +=
        value * weight;

    // Record feature value.
    if (uniform(*rnd) > options.missing_ratio()) {
      if (options.represent_numerical_as_integer()) {
        SetNumericalIntFeature(
            FeatureName(options, kTypeNumerical, feature_idx),
            value * kIntNumericalScale, example);
      } else {
        SetNumericalFeature(FeatureName(options, kTypeNumerical, feature_idx),
                            value, example);
      }
    }
  }
}

void AddMultidimensionalNumericalFeatures(
    const proto::SyntheticDatasetOptions& options, const GeneratorState& state,
    Example* example, utils::RandomEngine* rnd) {
  auto uniform = std::uniform_real_distribution<float>();
  for (int feature_idx = 0;
       feature_idx < options.num_multidimensional_numerical(); feature_idx++) {
    const float weight =
        DecreasingWeight(feature_idx, options.num_multidimensional_numerical());
    const bool is_missing = uniform(*rnd) < options.missing_ratio();
    const auto feature_name =
        FeatureName(options, kTypeMultidimensionalNumerical, feature_idx);
    for (int dim_idx = 0; dim_idx < options.multidimensional_numerical_dim();
         dim_idx++) {
      // Generate a random value.
      const float value = uniform(*rnd);

      // Add to accumulator.
      example->accumulators
          [state.multidimensional_numerical_accumulator_idxs[feature_idx]] +=
          value * weight / (dim_idx + 1);

      // Record feature value.
      if (!is_missing) {
        if (options.represent_numerical_as_integer()) {
          GetFeature(feature_name, example)
              ->mutable_int64_list()
              ->add_value(value * kIntNumericalScale);
        } else {
          GetFeature(feature_name, example)
              ->mutable_float_list()
              ->add_value(value);
        }
      }
    }
  }
}

void AddBooleanFeatures(const proto::SyntheticDatasetOptions& options,
                        const GeneratorState& state, Example* example,
                        utils::RandomEngine* rnd) {
  auto uniform = std::uniform_real_distribution<float>();
  for (int feature_idx = 0; feature_idx < options.num_boolean();
       feature_idx++) {
    // Generate a random value.
    const float value = uniform(*rnd) > 0.5f;

    // Add to accumulator.
    const float weight = DecreasingWeight(feature_idx, options.num_boolean());
    example->accumulators[state.boolean_accumulator_idxs[feature_idx]] +=
        value * weight;

    // Record feature value.
    if (uniform(*rnd) > options.missing_ratio()) {
      SetNumericalFeature(FeatureName(options, kTypeBoolean, feature_idx),
                          value, example);
    }
  }
}

void AddCategoricalFeatures(const proto::SyntheticDatasetOptions& options,
                            const GeneratorState& state,
                            const bool str_representation, Example* example,
                            utils::RandomEngine* rnd) {
  auto uniform = std::uniform_real_distribution<float>();
  const int vocab_size = options.categorical_vocab_size();
  for (int feature_idx = 0; feature_idx < options.num_categorical();
       feature_idx++) {
    // Select the accumulator.
    double* accumulator;
    if (str_representation) {
      accumulator = &example->accumulators
                         [state.categorical_str_accumulator_idxs[feature_idx]];
    } else {
      accumulator = &example->accumulators
                         [state.categorical_int_accumulator_idxs[feature_idx]];
    }

    // Generate a random value.
    const int value = std::uniform_int_distribution<int>(
        0, /*inclusive*/ vocab_size - 1)(*rnd);
    const int hashed_value = StableHash(value) % vocab_size;
    const float numerical_value = static_cast<float>(hashed_value) / vocab_size;

    // Add to accumulator.
    const float weight =
        DecreasingWeight(feature_idx, options.num_categorical());
    *accumulator += numerical_value * weight;

    // Record feature value.
    if (uniform(*rnd) > options.missing_ratio()) {
      if (str_representation) {
        SetCategoricalStringFeature(
            FeatureName(options, kTypeCategoricalStr, feature_idx),
            Symbol("V", value), example);
      } else {
        SetCategoricalIntFeature(
            FeatureName(options, kTypeCategoricalInt, feature_idx),
            value + (options.zero_categorical_int_value_is_oov() ? 1 : 0),
            example);
      }
    }
  }
}

void AddCategoricalSetFeatures(const proto::SyntheticDatasetOptions& options,
                               const GeneratorState& state,
                               const bool str_representation, Example* example,
                               utils::RandomEngine* rnd) {
  auto uniform = std::uniform_real_distribution<float>();
  const int vocab_size = options.categorical_set_vocab_size();
  for (int feature_idx = 0; feature_idx < options.num_categorical_set();
       feature_idx++) {
    // Select the accumulator.
    double* accumulator;
    if (str_representation) {
      accumulator =
          &example->accumulators
               [state.categorical_set_str_accumulator_idxs[feature_idx]];
    } else {
      accumulator =
          &example->accumulators
               [state.categorical_set_int_accumulator_idxs[feature_idx]];
    }

    const float weight =
        DecreasingWeight(feature_idx, options.num_categorical_set());

    // Generate a random set of values.
    std::vector<int> values;
    for (int value = 0; value < vocab_size; value++) {
      // Is this value included?
      if (uniform(*rnd) * vocab_size >= options.categorical_set_mean_size()) {
        continue;
      }

      values.push_back(value);
      const int hashed_value = StableHash(value) % vocab_size;
      const float numerical_value =
          static_cast<float>(hashed_value) / vocab_size;

      // Add to accumulator.
      *accumulator +=
          numerical_value * weight / options.categorical_set_mean_size();
    }

    // Record feature value.
    if (uniform(*rnd) > options.missing_ratio()) {
      if (str_representation) {
        SetCategoricalSetStringFeature(
            FeatureName(options, kTypeCategoricalSetStr, feature_idx), values,
            example);
      } else {
        SetCategoricalSetIntFeature(
            FeatureName(options, kTypeCategoricalSetInt, feature_idx), values,
            example);
      }
    }
  }
}

utils::StatusOr<std::vector<Example>> CreateFeatures(
    const proto::SyntheticDatasetOptions& options, const GeneratorState& state,
    utils::RandomEngine* rnd) {
  std::vector<Example> examples;
  examples.reserve(options.num_examples());

  for (int example_idx = 0; example_idx < options.num_examples();
       example_idx++) {
    Example example;
    example.accumulators.assign(options.num_accumulators(), {});
    example.noisy_accumulator_ranks.assign(options.num_accumulators(), {});

    // Numerical features.
    AddNumericalFeatures(options, state, &example, rnd);

    // Multidimensional numerical features.
    AddMultidimensionalNumericalFeatures(options, state, &example, rnd);

    // Categorical features.
    AddCategoricalFeatures(options, state, /*str_representation=*/false,
                           &example, rnd);
    AddCategoricalFeatures(options, state, /*str_representation=*/true,
                           &example, rnd);

    // Boolean features.
    AddBooleanFeatures(options, state, &example, rnd);

    // Categorical-set features.
    AddCategoricalSetFeatures(options, state, /*str_representation=*/false,
                              &example, rnd);
    AddCategoricalSetFeatures(options, state, /*str_representation=*/true,
                              &example, rnd);

    examples.push_back(std::move(example));
  }

  return std::move(examples);
}

absl::Status ComputeAccumulatorRanks(
    const proto::SyntheticDatasetOptions& options,
    std::vector<Example>* examples, utils::RandomEngine* rnd) {
  const int rank_noise =
      static_cast<int>(examples->size() * options.label_noise_ratio());
  auto raw_label_rank_noise =
      std::uniform_real_distribution<float>(-rank_noise, rank_noise);

  std::vector<std::pair<float, int>> sorted_labels(examples->size());
  for (int accumulator_idx = 0; accumulator_idx < options.num_accumulators();
       accumulator_idx++) {
    // Gather and sort all the accumulator values.
    for (int example_idx = 0; example_idx < examples->size(); example_idx++) {
      sorted_labels[example_idx].first =
          (*examples)[example_idx].accumulators[accumulator_idx];
      sorted_labels[example_idx].second = example_idx;
    }
    std::sort(sorted_labels.begin(), sorted_labels.end());

    // Update the rank field of each example.
    for (int rank_idx = 0; rank_idx < examples->size(); rank_idx++) {
      // Add some noise to the rank.
      int noisy_rank = rank_idx + raw_label_rank_noise(*rnd);
      if (noisy_rank < 0) {
        noisy_rank = 0;
      }
      if (noisy_rank >= examples->size()) {
        noisy_rank = examples->size() - 1;
      }

      // Record the rank.
      (*examples)[sorted_labels[rank_idx].second]
          .noisy_accumulator_ranks[accumulator_idx] = noisy_rank;
    }
  }
  return absl::OkStatus();
}

// Gets the value as a number in [0,1) from the accumulator ranks and values.
float ComputeLabelValue(const Example& example, const int num_examples) {
  // Transverse the accumulators as a heap binary tree.
  int accumulator_idx = 0;
  while (true) {
    const bool is_leaf = 2 * accumulator_idx + 2 >= example.accumulators.size();
    if (is_leaf) {
      break;
    }
    const bool condition =
        example.noisy_accumulator_ranks[accumulator_idx] * 2 >= num_examples;
    accumulator_idx = 2 * accumulator_idx + (condition ? 2 : 1);
  }

  return static_cast<float>(example.noisy_accumulator_ranks[accumulator_idx]) /
         num_examples;
}

absl::Status CreateLabels(const proto::SyntheticDatasetOptions& options,
                          const GeneratorState& state,
                          std::vector<Example>* examples,
                          utils::RandomEngine* rnd) {
  RETURN_IF_ERROR(ComputeAccumulatorRanks(options, examples, rnd));

  const int rank_noise =
      static_cast<int>(examples->size() * options.label_noise_ratio());
  auto raw_label_rank_noise =
      std::uniform_real_distribution<float>(-rank_noise, rank_noise);

  utils::RandomEngine ranking_group_rnd;

  for (int example_idx = 0; example_idx < examples->size(); example_idx++) {
    auto& example = (*examples)[example_idx];
    const float label = ComputeLabelValue(example, examples->size());

    switch (options.task_case()) {
      case proto::SyntheticDatasetOptions::TASK_NOT_SET:
      case proto::SyntheticDatasetOptions::kClassification: {
        const int class_idx =
            static_cast<int>(label * options.classification().num_classes());
        if (options.classification().store_label_as_str()) {
          SetCategoricalStringFeature(options.label_name(),
                                      Symbol("C", class_idx), &example);
        } else {
          SetCategoricalIntFeature(
              options.label_name(),
              class_idx + (options.zero_categorical_int_value_is_oov() ? 1 : 0),
              &example);
        }
      } break;

      case proto::SyntheticDatasetOptions::kRegression:
        SetNumericalFeature(options.label_name(), label, &example);
        break;

      case proto::SyntheticDatasetOptions::kRanking: {
        example.group_idx = example_idx / options.ranking().group_size();
        ranking_group_rnd.seed(example.group_idx);
        const float group_offset =
            std::uniform_real_distribution<float>()(ranking_group_rnd);
        const float group_scale =
            std::uniform_real_distribution<float>()(ranking_group_rnd);
        const float relevance =
            5.f * std::pow(label * 0.6f * (group_scale * 0.2f + 0.8f) +
                               group_offset * 0.4f,
                           2.f);
        SetNumericalFeature(options.label_name(), relevance, &example);
        SetCategoricalStringFeature(options.ranking().group_name(),
                                    absl::StrCat("G", example.group_idx),
                                    &example);
        break;
      }
    }
  }

  return absl::OkStatus();
}

// Writes the list of examples to a tensorflow.Example container.
absl::Status WriteTFEExamples(const std::vector<Example>& examples,
                              absl::string_view typed_path) {
  ASSIGN_OR_RETURN(auto writer, CreateTFExampleWriter(typed_path, -1));
  for (const auto& example : examples) {
    RETURN_IF_ERROR(writer->Write(example.tf_example));
  }
  return writer->CloseWithStatus();
}

// Writes a list of examples to a CSV file.
absl::Status WriteCsvExamples(const std::vector<Example>& examples,
                              absl::string_view typed_path) {
  // Open the output file.
  const auto path = dataset::GetDatasetPathAndType(typed_path).first;
  ASSIGN_OR_RETURN(auto file_handle, file::OpenOutputFile(path));
  file::OutputFileCloser file_closer(std::move(file_handle));
  utils::csv::Writer writer(file_closer.stream());

  // List the input features i.e. the csv header.
  std::set<std::string> tmp_keys;
  for (const auto& example : examples) {
    for (const auto& feature : example.tf_example.features().feature()) {
      tmp_keys.insert(feature.first);
    }
  }
  std::vector<std::string> keys(tmp_keys.begin(), tmp_keys.end());
  std::sort(keys.begin(), keys.end());

  // Write header.
  RETURN_IF_ERROR(writer.WriteRowStrings(keys));

  constexpr char kMultiDimError[] =
      "The CSV format does not support multi-dimensional features. Use another "
      "format (e.g. TFRecord) or disable multi-dimensional features (e.g. "
      "num_multidimensional_numerical=0 and num_categorical_set=0 in the "
      "config).";

  // Write rows.
  std::vector<std::string> row;
  for (const auto& example : examples) {
    row.clear();
    for (const auto& key : keys) {
      const auto feature_it = example.tf_example.features().feature().find(key);
      if (feature_it == example.tf_example.features().feature().end()) {
        row.push_back("");  // Missing value.
        continue;
      }
      const auto& feature = feature_it->second;
      switch (feature.kind_case()) {
        case tensorflow::Feature::kBytesList:
          if (feature.bytes_list().value_size() != 1) {
            return absl::InvalidArgumentError(kMultiDimError);
          }
          row.push_back(feature.bytes_list().value(0));
          break;
        case tensorflow::Feature::kFloatList:
          if (feature.float_list().value_size() != 1) {
            return absl::InvalidArgumentError(kMultiDimError);
          }
          row.push_back(absl::StrCat(feature.float_list().value(0)));
          break;
        case tensorflow::Feature::kInt64List:
          if (feature.int64_list().value_size() != 1) {
            return absl::InvalidArgumentError(kMultiDimError);
          }
          row.push_back(absl::StrCat(feature.int64_list().value(0)));
          break;
        case tensorflow::Feature::KIND_NOT_SET:
          row.push_back("");  // Missing value.
          break;
      }
    }
    RETURN_IF_ERROR(writer.WriteRowStrings(row));
  }

  return file_closer.Close();
}

absl::Status WriteExamples(const std::vector<Example>& examples,
                           absl::string_view typed_path) {
  // Note: We don't use the generic example writer (CreateExampleWriter) because
  // we don't have (and do not want to create) a dataspec.
  std::string sharded_path;
  dataset::proto::DatasetFormat format;
  std::tie(sharded_path, format) = dataset::GetDatasetPathAndType(typed_path);
  if (format == dataset::proto::DatasetFormat::FORMAT_CSV) {
    return WriteCsvExamples(examples, typed_path);
  } else {
    // Will fail if the format is not based on tensorflow.Example protos.
    return WriteTFEExamples(examples, typed_path);
  }
}

utils::StatusOr<GeneratorState> CreateState(
    const proto::SyntheticDatasetOptions& options, utils::RandomEngine* rnd) {
  GeneratorState state;
  auto accumulator_dist =
      std::uniform_int_distribution<int>(0, options.num_accumulators() - 1);

  for (int i = 0; i < options.num_numerical(); i++) {
    state.numerical_accumulator_idxs.push_back(accumulator_dist(*rnd));
  }

  for (int i = 0; i < options.num_categorical(); i++) {
    state.categorical_str_accumulator_idxs.push_back(accumulator_dist(*rnd));
  }

  for (int i = 0; i < options.num_categorical(); i++) {
    state.categorical_int_accumulator_idxs.push_back(accumulator_dist(*rnd));
  }

  for (int i = 0; i < options.num_categorical_set(); i++) {
    state.categorical_set_str_accumulator_idxs.push_back(
        accumulator_dist(*rnd));
  }

  for (int i = 0; i < options.num_categorical_set(); i++) {
    state.categorical_set_int_accumulator_idxs.push_back(
        accumulator_dist(*rnd));
  }

  for (int i = 0; i < options.num_boolean(); i++) {
    state.boolean_accumulator_idxs.push_back(accumulator_dist(*rnd));
  }

  for (int i = 0; i < options.num_multidimensional_numerical(); i++) {
    state.multidimensional_numerical_accumulator_idxs.push_back(
        accumulator_dist(*rnd));
  }
  return state;
}

}  // namespace

absl::Status GenerateSyntheticDataset(
    const proto::SyntheticDatasetOptions& options,
    absl::string_view typed_path) {
  auto rnd = CreateRandomGenerator(options);
  ASSIGN_OR_RETURN(auto state, CreateState(options, &rnd));
  ASSIGN_OR_RETURN(auto examples, CreateFeatures(options, state, &rnd));
  RETURN_IF_ERROR(CreateLabels(options, state, &examples, &rnd));
  return WriteExamples(examples, typed_path);
}

absl::Status GenerateSyntheticDatasetTrainValidTest(
    const proto::SyntheticDatasetOptions& options,
    const absl::string_view typed_path_train,
    const absl::string_view typed_path_valid,
    const absl::string_view typed_path_test, const float ratio_valid,
    const float ratio_test) {
  CHECK_GE(ratio_valid, 0.0);
  CHECK_LE(ratio_valid, 1.0);
  CHECK_GE(ratio_test, 0.0);
  CHECK_LE(ratio_test, 1.0);
  CHECK_GE(1.0 - ratio_test - ratio_valid, 0.0);
  CHECK_LE(1.0 - ratio_test - ratio_valid, 1.0);

  if (typed_path_valid.empty() && ratio_valid > 0) {
    return absl::InvalidArgumentError(
        "\"valid\" cannot be empty if \"ratio_valid\" >0.");
  }

  if (typed_path_test.empty() && ratio_test > 0) {
    return absl::InvalidArgumentError(
        "\"test\" cannot be empty if \"ratio_test\" >0.");
  }

  auto rnd = CreateRandomGenerator(options);
  ASSIGN_OR_RETURN(auto state, CreateState(options, &rnd));
  ASSIGN_OR_RETURN(auto examples, CreateFeatures(options, state, &rnd));
  RETURN_IF_ERROR(CreateLabels(options, state, &examples, &rnd));

  auto uniform = std::uniform_real_distribution<float>();
  std::vector<Example> example_train;
  std::vector<Example> example_test;
  std::vector<Example> example_valid;

  // Random generator seeded with the example group. Only used for ranking.
  utils::RandomEngine ground_rnd;

  for (const auto& example : examples) {
    // Destination of the example among train, valid and test.
    float dst;

    // Ensure that examples from the same group are written to the same
    // destination.
    if (options.has_ranking()) {
      ground_rnd.seed(example.group_idx);
      dst = uniform(ground_rnd);
    } else {
      dst = uniform(rnd);
    }

    if (dst < ratio_valid) {
      example_valid.push_back(example);
    } else if (dst < ratio_valid + ratio_test) {
      example_test.push_back(example);
    } else {
      example_train.push_back(example);
    }
  }

  RETURN_IF_ERROR(WriteExamples(example_train, typed_path_train));
  if (!typed_path_valid.empty()) {
    RETURN_IF_ERROR(WriteExamples(example_valid, typed_path_valid));
  }
  if (!typed_path_test.empty()) {
    RETURN_IF_ERROR(WriteExamples(example_test, typed_path_test));
  }
  return absl::OkStatus();
}

}  // namespace dataset
}  // namespace yggdrasil_decision_forests
