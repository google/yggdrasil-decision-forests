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

#include "absl/status/status.h"
#if defined YGG_TFRECORD_PREDICTIONS
#include "yggdrasil_decision_forests/utils/sharded_io_tfrecord.h"
#endif

#include "yggdrasil_decision_forests/dataset/data_spec.h"
#include "yggdrasil_decision_forests/dataset/example_writer.h"
#include "yggdrasil_decision_forests/dataset/formats.h"
#include "yggdrasil_decision_forests/utils/distribution.h"
#include "yggdrasil_decision_forests/utils/distribution.pb.h"
#include "yggdrasil_decision_forests/utils/evaluation.h"
#include "yggdrasil_decision_forests/utils/status_macros.h"

namespace yggdrasil_decision_forests {
namespace utils {

absl::StatusOr<PredictionFormat> ParsePredictionFormat(
    absl::string_view value) {
  if (value == "kRaw") {
    return PredictionFormat::kRaw;
  }
  if (value == "kSimple") {
    return PredictionFormat::kSimple;
  }
  if (value == "kRich") {
    return PredictionFormat::kRich;
  }
  if (value == "kFull") {
    return PredictionFormat::kFull;
  }
  return absl::InvalidArgumentError(absl::StrCat("Unknown format ", value));
}

absl::Status ExportPredictions(
    const std::vector<model::proto::Prediction>& predictions,
    model::proto::Task task, const dataset::proto::Column& label_column,
    absl::string_view typed_prediction_path,
    const int num_records_by_shard_in_output,
    const absl::optional<std::string> prediction_key,
    const PredictionFormat format) {
  // Determines the container for the predictions.
  std::string prediction_path, prediction_format;
  ASSIGN_OR_RETURN(std::tie(prediction_format, prediction_path),
                   dataset::SplitTypeAndPath(typed_prediction_path));

#if defined YGG_TFRECORD_PREDICTIONS
  if (prediction_format == "tfrecord+pred") {
    // Save the prediction as a tfrecord of proto::Predictions.
    auto prediction_writer =
        absl::make_unique<TFRecordShardedWriter<model::proto::Prediction>>();
    RETURN_IF_ERROR(prediction_writer->Open(prediction_path,
                                            num_records_by_shard_in_output));
    for (const auto& prediction : predictions) {
      RETURN_IF_ERROR(prediction_writer->Write(prediction));
    }
    return absl::OkStatus();
  }
#endif

  // Save the prediction as a collection (e.g. tfrecord or csv) of
  // proto::Examples.
  ASSIGN_OR_RETURN(auto dataspec, PredictionDataspec(task, label_column,
                                                     prediction_key, format));
  ASSIGN_OR_RETURN(auto writer, dataset::CreateExampleWriter(
                                    typed_prediction_path, dataspec,
                                    num_records_by_shard_in_output));
  dataset::proto::Example prediction_as_example;
  for (const auto& prediction : predictions) {
    // Convert the prediction into an example.
    RETURN_IF_ERROR(PredictionToExample(task, label_column, prediction,
                                        &prediction_as_example, prediction_key,
                                        format));
    RETURN_IF_ERROR(writer->Write(prediction_as_example));
  }
  return absl::OkStatus();
}

absl::Status PredictionToExample(
    model::proto::Task task, const dataset::proto::Column& label_col,
    const model::proto::Prediction& prediction,
    dataset::proto::Example* prediction_as_example,
    const absl::optional<std::string> prediction_key,
    const PredictionFormat format) {
  prediction_as_example->clear_attributes();
  switch (task) {
    case model::proto::Task::CLASSIFICATION: {
      const int num_label_values =
          static_cast<int>(label_col.categorical().number_of_unique_values());
      if (num_label_values !=
          prediction.classification().distribution().counts_size()) {
        return absl::InvalidArgumentError("Wrong number of classes.");
      }

      // Note: The kRaw format is the only format that does not contain the
      // output class.
      if (format != PredictionFormat::kRaw) {
        prediction_as_example->add_attributes()->set_categorical(
            prediction.classification().value());
      }

      // Predicted probability of predicted class.
      if (format == PredictionFormat::kRich) {
        DCHECK_GT(prediction.classification().distribution().sum(), 0);
        const float prediction_proba =
            prediction.classification().distribution().counts(
                prediction.classification().value()) /
            prediction.classification().distribution().sum();
        prediction_as_example->add_attributes()->set_numerical(
            prediction_proba);
      }

      // Predicted probability of all the classes.
      if (format == PredictionFormat::kRaw ||
          format == PredictionFormat::kFull) {
        const int num_label_values =
            label_col.categorical().number_of_unique_values();
        if (num_label_values !=
            prediction.classification().distribution().counts_size()) {
          return absl::InvalidArgumentError("Wrong number of classes.");
        }
        // We don't export the prediction for the out-of-dictionary item label
        // value (value 0). See kOutOfDictionaryItemKey in "data_spec.h" for
        // details.
        DCHECK_GT(prediction.classification().distribution().sum(), 0);
        for (int label_value = 1; label_value < num_label_values;
             ++label_value) {
          const float prediction_proba =
              prediction.classification().distribution().counts(label_value) /
              prediction.classification().distribution().sum();
          prediction_as_example->add_attributes()->set_numerical(
              prediction_proba);
        }
      }
    } break;
    case model::proto::Task::REGRESSION:
      prediction_as_example->add_attributes()->set_numerical(
          prediction.regression().value());
      break;
    case model::proto::Task::RANKING:
      prediction_as_example->add_attributes()->set_numerical(
          prediction.ranking().relevance());
      break;
    case model::proto::Task::CATEGORICAL_UPLIFT: {
      const int num_label_values =
          static_cast<int>(label_col.categorical().number_of_unique_values());
      if (num_label_values - 2 != prediction.uplift().treatment_effect_size()) {
        return absl::InvalidArgumentError("Wrong number of effects.");
      }
      // There are two excluded label values:
      // 0: Out-of-vocabulary.
      // 1: Treatment.
      for (int label_value = 2; label_value < num_label_values; label_value++) {
        const int effect_idx = label_value - 2;
        prediction_as_example->add_attributes()->set_numerical(
            prediction.uplift().treatment_effect(effect_idx));
      }
    } break;
    default:
      return absl::InvalidArgumentError("Non supported class");
  }

  // Export the prediction keys.
  // Column spec for the prediction key.
  if (prediction_key.has_value()) {
    prediction_as_example->add_attributes()->set_text(prediction.example_key());
  }

  return absl::OkStatus();
}

absl::Status ExampleToPrediction(
    model::proto::Task task, const dataset::proto::Column& label_col,
    const dataset::proto::Example& prediction_as_example,
    model::proto::Prediction* prediction) {
  switch (task) {
    case model::proto::Task::CLASSIFICATION: {
      utils::IntegerDistributionFloat distribution;
      const int num_label_values =
          static_cast<int>(label_col.categorical().number_of_unique_values());
      distribution.SetNumClasses(num_label_values);
      if (prediction_as_example.attributes_size() != num_label_values - 1) {
        return absl::InvalidArgumentError("Wrong number of predictions.");
      }
      // Skip the out-of-vocabulary (0) item.
      for (int label_value = 1; label_value < num_label_values; label_value++) {
        const int score_col_idx = label_value - 1;
        if (!prediction_as_example.attributes(score_col_idx).has_numerical()) {
          return absl::InvalidArgumentError("The prediction is not numerical");
        }
        const float score =
            prediction_as_example.attributes(score_col_idx).numerical();
        distribution.Add(label_value, score);
      }
      prediction->mutable_classification()->set_value(distribution.TopClass());
      distribution.Save(
          prediction->mutable_classification()->mutable_distribution());
    } break;
    case model::proto::Task::REGRESSION:
      if (prediction_as_example.attributes_size() != 1) {
        return absl::InvalidArgumentError("Wrong number of predictions.");
      }
      if (!prediction_as_example.attributes(0).has_numerical()) {
        return absl::InvalidArgumentError("The prediction is not numerical");
      }
      prediction->mutable_regression()->set_value(
          prediction_as_example.attributes(0).numerical());
      break;
    case model::proto::Task::RANKING:
      if (prediction_as_example.attributes_size() != 1) {
        return absl::InvalidArgumentError("Wrong number of predictions.");
      }
      if (!prediction_as_example.attributes(0).has_numerical()) {
        return absl::InvalidArgumentError("The prediction is not numerical");
      }
      prediction->mutable_ranking()->set_relevance(
          prediction_as_example.attributes(0).numerical());
      break;
    case model::proto::Task::CATEGORICAL_UPLIFT: {
      const int num_label_values =
          static_cast<int>(label_col.categorical().number_of_unique_values());
      prediction->mutable_uplift()->clear_treatment_effect();
      if (prediction_as_example.attributes_size() != num_label_values - 2) {
        return absl::InvalidArgumentError("Wrong number of predictions.");
      }
      // Skip the out-of-vocabulary (0) and control (1) item.
      for (int label_value = 2; label_value < num_label_values; label_value++) {
        const int effect_idx = label_value - 2;
        if (!prediction_as_example.attributes(effect_idx).has_numerical()) {
          return absl::InvalidArgumentError("The prediction is not numerical");
        }
        const float effect =
            prediction_as_example.attributes(effect_idx).numerical();
        prediction->mutable_uplift()->add_treatment_effect(effect);
      }
    } break;
    default:
      YDF_LOG(FATAL) << "Non supported task.";
      break;
  }

  return absl::OkStatus();
}

absl::StatusOr<dataset::proto::DataSpecification> PredictionDataspec(
    const model::proto::Task task, const dataset::proto::Column& label_col,
    const absl::optional<std::string> prediction_key,
    const PredictionFormat format) {
  dataset::proto::DataSpecification dataspec;

  switch (task) {
    case model::proto::Task::CLASSIFICATION: {
      // Note: label_value starts at 1 since we don't predict the OOV
      // (out-of-dictionary) item.
      const int num_label_values =
          static_cast<int>(label_col.categorical().number_of_unique_values());

      if (format != PredictionFormat::kRaw) {
        auto* col = dataset::AddColumn(label_col.name(),
                                       dataset::proto::ColumnType::CATEGORICAL,
                                       &dataspec);
        *col->mutable_categorical() = label_col.categorical();
      }

      // Predicted probability of predicted class.
      if (format == PredictionFormat::kRich) {
        dataset::AddColumn(/*name=*/absl::StrCat("Conf.", label_col.name()),
                           dataset::proto::ColumnType::NUMERICAL, &dataspec);
      }

      // Predicted probability of all the classes.
      if (format == PredictionFormat::kRaw ||
          format == PredictionFormat::kFull) {
        // We don't export the prediction for the out-of-dictionary item label
        // value (value 0). See kOutOfDictionaryItemKey in "data_spec.h" for
        // details.
        for (int label_value = 1; label_value < num_label_values;
             ++label_value) {
          dataset::AddColumn(
              /*name=*/absl::StrCat(dataset::CategoricalIdxToRepresentation(
                  label_col, label_value)),
              dataset::proto::ColumnType::NUMERICAL, &dataspec);
        }
      }
    } break;
    case model::proto::Task::REGRESSION:
    case model::proto::Task::RANKING:
      dataset::AddColumn(label_col.name(),
                         dataset::proto::ColumnType::NUMERICAL, &dataspec);
      break;
    case model::proto::Task::CATEGORICAL_UPLIFT: {
      const int num_label_values =
          static_cast<int>(label_col.categorical().number_of_unique_values());
      // There are two excluded label values:
      // 0: Out-of-vocabulary.
      // 1: Control.
      for (int label_value = 2; label_value < num_label_values; label_value++) {
        dataset::AddColumn(absl::StrCat(dataset::CategoricalIdxToRepresentation(
                               label_col, label_value)),
                           dataset::proto::ColumnType::NUMERICAL, &dataspec);
      }
    } break;
    default:
      YDF_LOG(FATAL) << "Non supported task.";
      break;
  }

  // Column spec for the prediction key.
  if (prediction_key.has_value()) {
    dataset::AddColumn(prediction_key.value(),
                       dataset::proto::ColumnType::STRING, &dataspec);
  }

  return dataspec;
}

}  // namespace utils
}  // namespace yggdrasil_decision_forests
