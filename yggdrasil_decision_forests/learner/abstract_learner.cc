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

#include "yggdrasil_decision_forests/learner/abstract_learner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset_io.h"
#include "yggdrasil_decision_forests/dataset/weight.h"
#include "yggdrasil_decision_forests/dataset/weight.pb.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/metric/metric.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/concurrency.h"
#include "yggdrasil_decision_forests/utils/fold_generator.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"
#include "yggdrasil_decision_forests/utils/synchronization_primitives.h"
#include "yggdrasil_decision_forests/utils/uid.h"

namespace yggdrasil_decision_forests {
namespace model {

absl::Status AbstractLearner::LinkTrainingConfig(
    const proto::TrainingConfig& training_config,
    const dataset::proto::DataSpecification& data_spec,
    proto::TrainingConfigLinking* config_link) {
  // Label.
  int32_t label;
  if (!training_config.has_label()) {
    STATUS_FATAL("No label specified in the training config. Aborting.");
  }
  RETURN_IF_ERROR(dataset::GetSingleColumnIdxFromName(
      training_config.label(), data_spec, &label,
      "Retrieving label column failed. "));
  config_link->set_label(label);
  config_link->set_num_label_classes(
      data_spec.columns(label).categorical().number_of_unique_values());

  // CV group.
  int32_t cv_group = -1;
  if (training_config.has_cv_group()) {
    RETURN_IF_ERROR(dataset::GetSingleColumnIdxFromName(
        training_config.cv_group(), data_spec, &cv_group,
        "Retrieving cross-validation group column failed. "));
  }
  config_link->set_cv_group(cv_group);

  // Ranking group.
  int32_t ranking_group = -1;
  if (training_config.task() == proto::RANKING) {
    if (!training_config.has_ranking_group())
      return absl::InvalidArgumentError(
          "\"ranking_group\" should be specified for a ranking task.");
    RETURN_IF_ERROR(dataset::GetSingleColumnIdxFromName(
        training_config.ranking_group(), data_spec, &ranking_group,
        "Retrieving ranking_group column failed. "));
  } else {
    if (training_config.has_ranking_group())
      return absl::InvalidArgumentError(
          "\"ranking_group\" should not be specified for a non ranking task.");
  }
  config_link->set_ranking_group(ranking_group);

  // Uplift treatment.
  int32_t uplift_treatment = -1;
  if (training_config.task() == proto::CATEGORICAL_UPLIFT ||
      training_config.task() == proto::NUMERICAL_UPLIFT) {
    if (!training_config.has_uplift_treatment())
      return absl::InvalidArgumentError(
          "\"uplift_treatment\" should be specified for an uplift task.");
    RETURN_IF_ERROR(dataset::GetSingleColumnIdxFromName(
        training_config.uplift_treatment(), data_spec, &uplift_treatment,
        "Retrieving uplift_treatment column failed. "));
  } else {
    if (training_config.has_uplift_treatment())
      return absl::InvalidArgumentError(
          "\"uplift_treatment\" should not be specified for non uplift task.");
  }
  config_link->set_uplift_treatment(uplift_treatment);

  // Weights.
  if (training_config.has_weight_definition()) {
    RETURN_IF_ERROR(dataset::GetLinkedWeightDefinition(
        training_config.weight_definition(), data_spec,
        config_link->mutable_weight_definition()));
  }

  // List the model input features.
  std::vector<int32_t> feature_idxs;
  if (training_config.features().empty()) {
    YDF_LOG(INFO) << "No input feature explicitly specified. Using all the "
                     "available input features.";
    feature_idxs.assign(data_spec.columns_size(), 0);
    std::iota(feature_idxs.begin(), feature_idxs.end(), 0);
  } else {
    dataset::GetMultipleColumnIdxFromName(
        {training_config.features().begin(), training_config.features().end()},
        data_spec, &feature_idxs);
  }

  // Remove the label from the input features.
  auto it_label_result =
      std::find(feature_idxs.begin(), feature_idxs.end(), label);
  if (it_label_result != feature_idxs.end()) {
    YDF_LOG(INFO) << "The label \"" << training_config.label()
                  << "\" was removed from the input feature set.";
    feature_idxs.erase(it_label_result);
  }

  // Remove the rank group from the input features.
  if (ranking_group != -1) {
    auto it_ranking_group_result =
        std::find(feature_idxs.begin(), feature_idxs.end(), ranking_group);
    if (it_ranking_group_result != feature_idxs.end()) {
      YDF_LOG(INFO) << "The ranking_group \"" << training_config.ranking_group()
                    << "\" was removed from the input feature set.";
      feature_idxs.erase(it_ranking_group_result);
    }
  }

  // Remove the uplift treatment from the input features.
  if (uplift_treatment != -1) {
    auto it_uplift_treatment_result =
        std::find(feature_idxs.begin(), feature_idxs.end(), uplift_treatment);
    if (it_uplift_treatment_result != feature_idxs.end()) {
      YDF_LOG(INFO) << "The uplift_treatment \""
                    << training_config.uplift_treatment()
                    << "\" was removed from the input feature set.";
      feature_idxs.erase(it_uplift_treatment_result);
    }
  }

  // Remove the cv group from the input features.
  if (cv_group != -1) {
    auto it_cv_group_result =
        std::find(feature_idxs.begin(), feature_idxs.end(), cv_group);
    if (it_cv_group_result != feature_idxs.end()) {
      YDF_LOG(INFO) << "The cv_group \"" << training_config.cv_group()
                    << "\" was removed from the input feature set.";
      feature_idxs.erase(it_cv_group_result);
    }
  }

  // Weights
  if (config_link->has_weight_definition()) {
    auto it_weight_result =
        std::find(feature_idxs.begin(), feature_idxs.end(),
                  config_link->weight_definition().attribute_idx());
    if (it_weight_result != feature_idxs.end()) {
      YDF_LOG(INFO) << "The weight column \""
                    << training_config.weight_definition().attribute()
                    << "\" was removed from the input feature set.";
      feature_idxs.erase(it_weight_result);
    }
  }

  // Remove features that only contain missing values.
  const auto is_feature_empty = [&data_spec](const int feature_idx) {
    const auto& feature_col = data_spec.columns(feature_idx);
    bool is_fully_missing =
        data_spec.created_num_rows() > 0 &&
        feature_col.count_nas() == data_spec.created_num_rows();

    // Not fully missing values, but that are like missing statistically.
    is_fully_missing |= feature_col.has_numerical() &&
                        std::isnan(feature_col.numerical().mean());

    if (is_fully_missing) {
      YDF_LOG(WARNING) << "Remove feature \"" << feature_col.name()
                       << "\" because it only contains missing values.";
      return true;
    }
    return false;
  };
  feature_idxs.erase(std::remove_if(feature_idxs.begin(), feature_idxs.end(),
                                    is_feature_empty),
                     feature_idxs.end());

  *config_link->mutable_features() = {feature_idxs.begin(), feature_idxs.end()};

  // Index numerical features
  config_link->clear_numerical_features();
  absl::flat_hash_set<int> numerical_features;
  for (const auto feature_idx : feature_idxs) {
    if (data_spec.columns(feature_idx).type() == dataset::proto::NUMERICAL) {
      config_link->add_numerical_features(feature_idx);
      numerical_features.insert(feature_idx);
    }
  }

  // Allocate per-attributes array
  config_link->clear_per_columns();
  for (int i = 0; i < data_spec.columns_size(); i++) {
    config_link->add_per_columns();
  }

  // Monotonicity constraints
  for (const auto& src : training_config.monotonic_constraints()) {
    if (src.feature().empty()) {
      return absl::InvalidArgumentError(
          "Empty \"feature\" in a monotonicity constraint");
    }
    std::vector<int32_t> feature_idxs;
    dataset::GetMultipleColumnIdxFromName({src.feature()}, data_spec,
                                          &feature_idxs);
    if (feature_idxs.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat(src.feature(), " does not match any input features"));
    }
    for (const int src_feature : feature_idxs) {
      if (numerical_features.find(src_feature) == numerical_features.end()) {
        // Build error message.
        std::vector<std::string> str_numerical_features;
        str_numerical_features.reserve(numerical_features.size());
        for (const auto feature_idx : numerical_features) {
          str_numerical_features.push_back(
              data_spec.columns(feature_idx).name());
        }

        return absl::InvalidArgumentError(absl::Substitute(
            "Feature \"$0\" caught by regular expression \"$1\" is not a "
            "numerical input feature of the "
            "model. Make sure this "
            "feature is also defined as input feature of the model, and that "
            "it is numerical. The numerical input features are: [$2].",
            data_spec.columns(src_feature).name(), src.feature(),
            absl::StrJoin(str_numerical_features, ", ")));
      }
      auto* dst = config_link->mutable_per_columns(src_feature);
      *dst->mutable_monotonic_constraint() = src;
    }
  }

  return absl::OkStatus();
}

std::unique_ptr<AbstractModel> AbstractLearner::Train(
    const dataset::VerticalDataset& train_dataset) const {
  return TrainWithStatus(train_dataset).value();
}

std::unique_ptr<AbstractModel> AbstractLearner::Train(
    const absl::string_view typed_path,
    const dataset::proto::DataSpecification& data_spec) const {
  return TrainWithStatus(typed_path, data_spec).value();
}

absl::StatusOr<std::unique_ptr<AbstractModel>> AbstractLearner::TrainWithStatus(
    const absl::string_view typed_path,
    const dataset::proto::DataSpecification& data_spec,
    const absl::optional<std::string>& typed_valid_path) const {
  // List the columns used for the training.
  // Only these columns will be loaded.
  proto::TrainingConfigLinking link_config;
  RETURN_IF_ERROR(AbstractLearner::LinkTrainingConfig(training_config_,
                                                      data_spec, &link_config));
  auto dataset_loading_config = OptimalDatasetLoadingConfig(link_config);
  dataset_loading_config.num_threads = deployment().num_io_threads();

  dataset::VerticalDataset train_dataset;
  RETURN_IF_ERROR(LoadVerticalDataset(typed_path, data_spec, &train_dataset,
                                      /*required_columns=*/{},
                                      dataset_loading_config));

  RETURN_IF_ERROR(dataset::CheckNumExamples(train_dataset.nrow()));

  dataset::VerticalDataset valid_dataset_data;
  absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
      valid_dataset;
  if (typed_valid_path.has_value()) {
    RETURN_IF_ERROR(LoadVerticalDataset(
        typed_valid_path.value(), data_spec, &valid_dataset_data,
        /*required_columns=*/{}, dataset_loading_config));
    valid_dataset = valid_dataset_data;
  }
  return TrainWithStatus(train_dataset, valid_dataset);
}

absl::Status CheckGenericHyperParameterSpecification(
    const proto::GenericHyperParameters& params,
    const proto::GenericHyperParameterSpecification& spec) {
  std::set<std::string> already_defined_params;
  for (const auto& param : params.fields()) {
    if (already_defined_params.find(param.name()) !=
        already_defined_params.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The param \"", param.name(), "\" is defined multiple times."));
    }
    already_defined_params.insert(param.name());

    const auto spec_field_it = spec.fields().find(param.name());
    if (spec_field_it == spec.fields().end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown param \"", param.name(), "\"."));
    }

    switch (spec_field_it->second.Type_case()) {
      case proto::GenericHyperParameterSpecification::Value::TYPE_NOT_SET:
        return absl::InternalError("Missing generic hyper parameter type.");
      case proto::GenericHyperParameterSpecification::Value::kCategorical: {
        if (!param.value().has_categorical()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "The parameter \"", param.name(), "\" should be a categorical."));
        }
        const auto& possible_values =
            spec_field_it->second.categorical().possible_values();
        if (std::find(possible_values.begin(), possible_values.end(),
                      param.value().categorical()) == possible_values.end()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Unknown value \"", param.value().categorical(),
                           "\" for the parameter \"", param.name(), "\"."));
        }
      } break;
      case proto::GenericHyperParameterSpecification::Value::kInteger: {
        if (!param.value().has_integer()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "The parameter \"", param.name(), "\" should be an integer."));
        }
        const auto& integer = spec_field_it->second.integer();
        if (integer.has_minimum() &&
            param.value().integer() < integer.minimum()) {
          return absl::InvalidArgumentError(
              absl::StrCat("The parameter \"", param.name(),
                           "\" is smaller than the minimum permitted value."));
        }
        if (integer.has_maximum() &&
            param.value().integer() > integer.maximum()) {
          return absl::InvalidArgumentError(
              absl::StrCat("The parameter \"", param.name(),
                           "\" is larger than the maximum permitted value."));
        }
      } break;
      case proto::GenericHyperParameterSpecification::Value::kReal: {
        if (!param.value().has_real()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "The parameter \"", param.name(), "\" should be a real."));
        }
        const auto& real = spec_field_it->second.real();
        if (real.has_minimum() && param.value().real() < real.minimum()) {
          return absl::InvalidArgumentError(
              absl::StrCat("The parameter \"", param.name(),
                           "\" is smaller than the minimum permitted value."));
        }
        if (real.has_maximum() && param.value().real() > real.maximum()) {
          return absl::InvalidArgumentError(
              absl::StrCat("The parameter \"", param.name(),
                           "\" is larger than the maximum permitted value."));
        }
      } break;
      case proto::GenericHyperParameterSpecification::Value::kCategoricalList: {
        if (!param.value().has_categorical_list()) {
          return absl::InvalidArgumentError(
              absl::StrCat("The parameter \"", param.name(),
                           "\" should be a categorical_list."));
        }
      } break;
    }
  }
  return absl::OkStatus();
}

absl::Status AbstractLearner::CheckConfiguration(
    const dataset::proto::DataSpecification& data_spec,
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const model::proto::DeploymentConfig& deployment) {
  const auto& label_col_spec = data_spec.columns(config_link.label());
  // Check the type of the label column.
  switch (config.task()) {
    case model::proto::Task::UNDEFINED:
      return absl::InvalidArgumentError(
          "The \"task\" field is not defined in the TrainingConfig proto.");
      break;
    case model::proto::Task::CLASSIFICATION:
      if (label_col_spec.type() != dataset::proto::ColumnType::CATEGORICAL) {
        return absl::InvalidArgumentError(absl::StrCat(
            "The label column \"", config.label(),
            "\" should be CATEGORICAL for a CLASSIFICATION "
            "Task. Note: BOOLEAN columns should be set as CATEGORICAL using a "
            "dataspec guide, even for a binary classification task."));
      }
      break;
    case model::proto::Task::REGRESSION:
      if (label_col_spec.type() != dataset::proto::ColumnType::NUMERICAL) {
        return absl::InvalidArgumentError(
            absl::StrCat("The label column \"", config.label(),
                         "\" should be NUMERICAL for a REGRESSION task."));
      }
      break;
    case model::proto::Task::RANKING: {
      if (label_col_spec.type() != dataset::proto::ColumnType::NUMERICAL) {
        return absl::InvalidArgumentError(
            absl::StrCat("The label column \"", config.label(),
                         "\" should be NUMERICAL for a RANKING task."));
      }
      if (!config_link.has_ranking_group() || config_link.ranking_group() < 0) {
        return absl::InvalidArgumentError(
            "The \"ranking_group\" is not defined but required for a RANKING "
            "task.");
      }
      const auto& ranking_group_col_spec =
          data_spec.columns(config_link.ranking_group());
      if (ranking_group_col_spec.type() !=
              dataset::proto::ColumnType::CATEGORICAL &&
          ranking_group_col_spec.type() != dataset::proto::ColumnType::HASH) {
        return absl::InvalidArgumentError(
            "The \"ranking_group\" column must be CATEGORICAL or HASH.");
      }
      if (ranking_group_col_spec.type() ==
          dataset::proto::ColumnType::CATEGORICAL) {
        YDF_LOG(WARNING) << "The grouping column \"" << config.ranking_group()
                         << "\" is of CATEGORICAL type. The STRING type is "
                            "generally a better choice.";
        if (ranking_group_col_spec.categorical().min_value_count() != 1) {
          return absl::InvalidArgumentError(
              "The \"ranking_group\" column must have a \"min_value_count\" "
              "of 1 in the dataspec guide. This ensures that rare groups are "
              "not pruned.");
        }
        if (ranking_group_col_spec.categorical()
                .max_number_of_unique_values() != -1) {
          return absl::InvalidArgumentError(
              "The \"ranking_group\" column must have a "
              "\"max_number_of_unique_values\" "
              "of -1 in the dataspec guide. This ensures that rare groups are "
              "not pruned.");
        }
      }
    } break;
    case model::proto::Task::CATEGORICAL_UPLIFT: {
      if (label_col_spec.type() != dataset::proto::ColumnType::CATEGORICAL) {
        return absl::InvalidArgumentError(
            "The label column should be CATEGORICAL for an CATEGORICAL_UPLIFT "
            "task.");
      }
      if (!config_link.has_uplift_treatment() ||
          config_link.uplift_treatment() < 0) {
        return absl::InvalidArgumentError(
            "The \"uplift_treatment\" is not defined but required for an "
            "UPLIFT task.");
      }
      const auto& uplift_treatment_col_spec =
          data_spec.columns(config_link.uplift_treatment());
      if (uplift_treatment_col_spec.type() !=
          dataset::proto::ColumnType::CATEGORICAL) {
        return absl::InvalidArgumentError(
            "The \"uplift_treatment\" column must be CATEGORICAL.");
      }
      const auto num_treatment_classes =
          uplift_treatment_col_spec.categorical().number_of_unique_values();
      if (num_treatment_classes > 3) {
        return absl::InvalidArgumentError(
            "Uplift only supports binary treatments.");
      }
    } break;
    case model::proto::Task::NUMERICAL_UPLIFT: {
      if (label_col_spec.type() != dataset::proto::ColumnType::NUMERICAL) {
        return absl::InvalidArgumentError(
            "The label column should be NUMERICAL for an NUMERICAL_UPLIFT "
            "task.");
      }
      if (!config_link.has_uplift_treatment() ||
          config_link.uplift_treatment() < 0) {
        return absl::InvalidArgumentError(
            "The \"uplift_treatment\" is not defined but required for an "
            "UPLIFT task.");
      }
      const auto& uplift_treatment_col_spec =
          data_spec.columns(config_link.uplift_treatment());
      if (uplift_treatment_col_spec.type() !=
          dataset::proto::ColumnType::CATEGORICAL) {
        return absl::InvalidArgumentError(
            "The \"uplift_treatment\" column must be CATEGORICAL.");
      }
      const auto num_treatment_classes =
          uplift_treatment_col_spec.categorical().number_of_unique_values();
      if (num_treatment_classes > 3) {
        return absl::InvalidArgumentError(
            "Uplift only supports binary treatments.");
      }
    } break;
  }
  // Check the label don't contains NaN.
  if (label_col_spec.count_nas() != 0) {
    return absl::InvalidArgumentError(
        absl::Substitute("The label column \"$0\" should not contain NaN / "
                         "missing values. $1 missing values are found.",
                         config.label(), label_col_spec.count_nas()));
  }
  if (deployment.num_threads() < 0) {
    return absl::InvalidArgumentError("The number of threads should be >= 0");
  }
  return absl::OkStatus();
}

absl::Status AbstractLearner::SetHyperParameters(
    const proto::GenericHyperParameters& generic_hyper_params) {
  ASSIGN_OR_RETURN(const auto h_param_spec,
                   GetGenericHyperParameterSpecification());
  RETURN_IF_ERROR(CheckGenericHyperParameterSpecification(generic_hyper_params,
                                                          h_param_spec));
  utils::GenericHyperParameterConsumer consumer(generic_hyper_params);
  RETURN_IF_ERROR(SetHyperParametersImpl(&consumer));
  return consumer.CheckThatAllHyperparametersAreConsumed();
}

absl::Status AbstractLearner::SetHyperParametersImpl(
    utils::GenericHyperParameterConsumer* generic_hyper_params) {
  {
    const auto hparam =
        generic_hyper_params->Get(kHParamMaximumTrainingDurationSeconds);
    if (hparam.has_value()) {
      if (hparam.value().value().real() >= 0) {
        training_config_.set_maximum_training_duration_seconds(
            hparam.value().value().real());
      } else {
        training_config_.clear_maximum_training_duration_seconds();
      }
    }
  }

  {
    const auto hparam =
        generic_hyper_params->Get(kHParamMaximumModelSizeInMemoryInBytes);
    if (hparam.has_value()) {
      if (hparam.value().value().real() >= 0) {
        training_config_.set_maximum_model_size_in_memory_in_bytes(
            hparam.value().value().real());
      } else {
        training_config_.clear_maximum_model_size_in_memory_in_bytes();
      }
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamRandomSeed);
    if (hparam.has_value()) {
      training_config_.set_random_seed(hparam.value().value().integer());
    }
  }

  {
    const auto hparam = generic_hyper_params->Get(kHParamPureServingModel);
    if (hparam.has_value()) {
      training_config_.set_pure_serving_model(
          hparam.value().value().categorical() == kTrue);
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<model::proto::GenericHyperParameterSpecification>
AbstractLearner::GetGenericHyperParameterSpecification() const {
  model::proto::GenericHyperParameterSpecification hparam_def;
  const proto::TrainingConfig default_config;

  {
    auto& param = hparam_def.mutable_fields()->operator[](
        kHParamMaximumTrainingDurationSeconds);
    param.mutable_real()->set_default_value(-1);
    param.mutable_documentation()->set_proto_path(
        "learner/abstract_learner.proto");
    param.mutable_documentation()->set_description(
        R"(Maximum training duration of the model expressed in seconds. Each learning algorithm is free to use this parameter at it sees fit. Enabling maximum training duration makes the model training non-deterministic.)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](
        kHParamMaximumModelSizeInMemoryInBytes);
    param.mutable_real()->set_default_value(-1);
    param.mutable_documentation()->set_proto_path(
        "learner/abstract_learner.proto");
    param.mutable_documentation()->set_description(
        R"(Limit the size of the model when stored in ram. Different algorithms can enforce this limit differently. Note that when models are compiled into an inference, the size of the inference engine is generally much smaller than the original model.)");
  }

  {
    auto& param = hparam_def.mutable_fields()->operator[](kHParamRandomSeed);
    param.mutable_integer()->set_default_value(default_config.random_seed());
    param.mutable_documentation()->set_proto_path(
        "learner/abstract_learner.proto");
    param.mutable_documentation()->set_description(
        R"(Random seed for the training of the model. Learners are expected to be deterministic by the random seed.)");
  }

  {
    auto& param =
        hparam_def.mutable_fields()->operator[](kHParamPureServingModel);
    param.mutable_categorical()->set_default_value(
        default_config.pure_serving_model() ? kTrue : kFalse);
    param.mutable_categorical()->add_possible_values(kTrue);
    param.mutable_categorical()->add_possible_values(kFalse);
    param.mutable_documentation()->set_proto_path(
        "learner/abstract_learner.proto");
    param.mutable_documentation()->set_description(
        R"(Clear the model from any information that is not required for model serving. This includes debugging, model interpretation and other meta-data. The size of the serialized model can be reduced significatively (50% model size reduction is common). This parameter has no impact on the quality, serving speed or RAM usage of model serving.)");
  }

  return hparam_def;
}

metric::proto::EvaluationResults EvaluateLearner(
    const model::AbstractLearner& learner,
    const dataset::VerticalDataset& dataset,
    const utils::proto::FoldGenerator& fold_generator,
    const metric::proto::EvaluationOptions& evaluation_options,
    const proto::DeploymentConfig& deployment_evaluation) {
  return EvaluateLearnerOrStatus(learner, dataset, fold_generator,
                                 evaluation_options, deployment_evaluation)
      .value();
}

absl::StatusOr<metric::proto::EvaluationResults> EvaluateLearnerOrStatus(
    const model::AbstractLearner& learner,
    const dataset::VerticalDataset& dataset,
    const utils::proto::FoldGenerator& fold_generator,
    const metric::proto::EvaluationOptions& evaluation_options,
    const proto::DeploymentConfig& deployment_evaluation) {
  // Make sure the computation distribution is supported.
  switch (deployment_evaluation.execution_case()) {
    case proto::DeploymentConfig::EXECUTION_NOT_SET:
    case proto::DeploymentConfig::kLocal:
      break;
    default:
      STATUS_FATAL("\"EvaluateLearner\" only support local deployment config.");
      break;
  }

  // Initialize the folds.
  utils::FoldList folds;
  RETURN_IF_ERROR(
      utils::GenerateFoldsConstDataset(fold_generator, dataset, &folds));
  const int num_folds = utils::NumberOfFolds(fold_generator, folds);

  // Get the label column specification.
  int32_t label_col_idx;
  RETURN_IF_ERROR(dataset::GetSingleColumnIdxFromName(
      learner.training_config().label(), dataset.data_spec(), &label_col_idx));
  const auto& label_col_spec = dataset.data_spec().columns(label_col_idx);

  // Protects "aggregated_evaluation".
  utils::concurrency::Mutex evaluation_mutex;
  metric::proto::EvaluationResults aggregated_evaluation;
  absl::Status status_train_and_evaluate;

  // Trains and evaluates a single model on the "fold_idx".
  const auto train_and_evaluate =
      [&aggregated_evaluation, &evaluation_mutex, &label_col_spec, &dataset,
       &folds, &learner, &evaluation_options, &status_train_and_evaluate](
          const int fold_idx, utils::RandomEngine* rnd) {
        metric::proto::EvaluationResults evaluation;
        {
          utils::concurrency::MutexLock lock(&evaluation_mutex);
          if (!status_train_and_evaluate.ok()) {
            return;
          }
          status_train_and_evaluate.Update(metric::InitializeEvaluation(
              evaluation_options, label_col_spec, &evaluation));
          if (!status_train_and_evaluate.ok()) {
            return;
          }
        }

        // Extract the training and testing dataset.
        const auto testing_dataset = dataset.Extract(folds[fold_idx]).value();
        const auto training_indices =
            utils::MergeIndicesExceptOneFold(folds, fold_idx);
        const auto training_dataset = dataset.Extract(training_indices).value();
        // Train a model.
        auto model = learner.TrainWithStatus(training_dataset).value();
        // Evaluate the model.
        auto status_append = model->AppendEvaluation(
            testing_dataset, evaluation_options, rnd, &evaluation);
        // Aggregate the evaluations.
        {
          utils::concurrency::MutexLock lock(&evaluation_mutex);
          status_train_and_evaluate.Update(status_append);
          status_train_and_evaluate.Update(metric::MergeEvaluation(
              evaluation_options, evaluation, &aggregated_evaluation));
        }
      };

  utils::RandomEngine rnd;
  RETURN_IF_ERROR(metric::InitializeEvaluation(
      evaluation_options, label_col_spec, &aggregated_evaluation));
  {
    yggdrasil_decision_forests::utils::concurrency::ThreadPool pool(
        "Evaluator", deployment_evaluation.num_threads());
    pool.StartWorkers();
    for (int fold_idx = 0; fold_idx < num_folds; fold_idx++) {
      pool.Schedule([&train_and_evaluate, fold_idx, seed{rnd()}]() {
        utils::RandomEngine rnd(seed);
        train_and_evaluate(fold_idx, &rnd);
      });
    }
  }

  RETURN_IF_ERROR(metric::FinalizeEvaluation(evaluation_options, label_col_spec,
                                             &aggregated_evaluation));
  return aggregated_evaluation;
}

void InitializeModelWithAbstractTrainingConfig(
    const proto::TrainingConfig& training_config,
    const proto::TrainingConfigLinking& training_config_linking,
    AbstractModel* model) {
  model->set_label_col_idx(training_config_linking.label());

  if (training_config.task() == proto::Task::RANKING) {
    model->set_ranking_group_col(training_config_linking.ranking_group());
  }

  if (training_config.task() == proto::Task::CATEGORICAL_UPLIFT ||
      training_config.task() == proto::Task::NUMERICAL_UPLIFT) {
    model->set_uplift_treatment_col(training_config_linking.uplift_treatment());
  }

  model->set_task(training_config.task());
  model->mutable_input_features()->assign(
      training_config_linking.features().begin(),
      training_config_linking.features().end());
  if (training_config_linking.has_weight_definition()) {
    model->set_weights(training_config_linking.weight_definition());
  }

  InitializeModelMetadataWithAbstractTrainingConfig(training_config, model);
}

void InitializeModelMetadataWithAbstractTrainingConfig(
    const proto::TrainingConfig& training_config, AbstractModel* model) {
  auto* dst = model->mutable_metadata();
  dst->Import(training_config.metadata());

  // Owner
  if (dst->owner().empty()) {
    auto opt_username = utils::UserName();
    if (opt_username.has_value()) {
      dst->set_owner(std::move(opt_username).value());
    }
  }

  // Date
  if (dst->created_date() == 0) {
    dst->set_created_date(absl::ToUnixSeconds(absl::Now()));
  }

  // UID
  if (dst->uid() == 0) {
    dst->set_uid(utils::GenUniqueIdUint64());
  }

  // Framework
  if (dst->framework().empty()) {
    dst->set_framework("Yggdrasil c++");
  }
}

absl::Status AbstractLearner::CheckCapabilities() const {
  // All the learners require a label.
  if (training_config().label().empty()) {
    return absl::InvalidArgumentError("\"label\" field required.");
  }
  const auto capabilities = Capabilities();

  // Maximum training duration.
  if (!capabilities.support_max_training_duration() &&
      training_config().has_maximum_training_duration_seconds()) {
    return absl::InvalidArgumentError(
        absl::Substitute("The learner $0 does not support the "
                         "\"maximum_training_duration_seconds\" flag.",
                         training_config().learner()));
  }

  // Maximum model size.
  if (!capabilities.support_max_model_size_in_memory() &&
      training_config().has_maximum_model_size_in_memory_in_bytes()) {
    return absl::InvalidArgumentError(
        absl::Substitute("The learner $0 does not support the "
                         "\"maximum_model_size_in_memory_in_bytes\" flag.",
                         training_config().learner()));
  }

  // Monotonic constraints
  if (!capabilities.support_monotonic_constraints() &&
      training_config().monotonic_constraints_size() > 0) {
    return absl::InvalidArgumentError(absl::Substitute(
        "The learner $0 does not support monotonic constraints.",
        training_config().learner()));
  }

  return absl::OkStatus();
}

absl::StatusOr<proto::HyperParameterSpace>
AbstractLearner::PredefinedHyperParameterSpace() const {
  return absl::InvalidArgumentError(
      absl::Substitute("Learner $0 does not provide a default hyper-parameter "
                       "space for optimization. You should define the set of "
                       "hyper-parameters to optimize manually.",
                       training_config_.learner()));
}

absl::Status CopyProblemDefinition(const proto::TrainingConfig& src,
                                   proto::TrainingConfig* dst) {
  if (src.has_label()) {
    if (dst->has_label() && dst->label() != src.label()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Invalid label. $0 != $1", src.label(), dst->label()));
    } else {
      dst->set_label(src.label());
    }
  }

  if (src.has_task()) {
    if (dst->has_task() && dst->task() != src.task()) {
      return absl::InvalidArgumentError(
          absl::Substitute("Invalid task. $0 != $1", src.task(), dst->task()));
    } else {
      dst->set_task(src.task());
    }
  }

  if (src.has_cv_group()) {
    if (dst->has_cv_group() && dst->cv_group() != src.cv_group()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Invalid cv_group. $0 != $1", src.cv_group(), dst->cv_group()));
    } else {
      dst->set_cv_group(src.cv_group());
    }
  }

  if (src.has_ranking_group()) {
    if (dst->has_ranking_group() &&
        dst->ranking_group() != src.ranking_group()) {
      return absl::InvalidArgumentError(
          absl::Substitute("Invalid ranking_group. $0 != $1",
                           src.ranking_group(), dst->ranking_group()));
    } else {
      dst->set_ranking_group(src.ranking_group());
    }
  }

  if (src.has_uplift_treatment()) {
    if (dst->has_uplift_treatment() &&
        dst->uplift_treatment() != src.uplift_treatment()) {
      return absl::InvalidArgumentError(
          absl::Substitute("Invalid uplift_treatment. $0 != $1",
                           src.uplift_treatment(), dst->uplift_treatment()));
    } else {
      dst->set_uplift_treatment(src.uplift_treatment());
    }
  }

  if (src.has_weight_definition()) {
    if (dst->has_weight_definition() &&
        dst->weight_definition().DebugString() !=
            src.weight_definition().DebugString()) {
      return absl::InvalidArgumentError("Invalid weight_definition.");
    } else {
      *dst->mutable_weight_definition() = src.weight_definition();
    }
  }

  if (src.features_size() > 0 && dst->features_size() == 0) {
    *dst->mutable_features() = src.features();
  }

  return absl::OkStatus();
}

dataset::LoadConfig OptimalDatasetLoadingConfig(
    const proto::TrainingConfigLinking& link_config) {
  dataset::LoadConfig load_config;
  load_config.load_columns = {link_config.features().begin(),
                              link_config.features().end()};
  if (link_config.has_label() && link_config.label() >= 0) {
    load_config.load_columns->push_back(link_config.label());
  }
  if (link_config.has_cv_group() && link_config.cv_group() >= 0) {
    load_config.load_columns->push_back(link_config.cv_group());
  }
  if (link_config.has_ranking_group() && link_config.ranking_group() >= 0) {
    load_config.load_columns->push_back(link_config.ranking_group());
  }
  if (link_config.has_uplift_treatment() &&
      link_config.uplift_treatment() >= 0) {
    load_config.load_columns->push_back(link_config.uplift_treatment());
  }

  if (link_config.has_weight_definition()) {
    load_config.load_columns->push_back(
        link_config.weight_definition().attribute_idx());
  }

  // Filter the examples with zero weight.
  if (link_config.has_weight_definition() &&
      link_config.weight_definition().has_numerical()) {
    const auto weight_attribute =
        link_config.weight_definition().attribute_idx();
    load_config.load_example =
        [weight_attribute](const dataset::proto::Example& example) {
          return example.attributes(weight_attribute).numerical() > 0.f;
        };
  }
  return load_config;
}

}  // namespace model
}  // namespace yggdrasil_decision_forests
