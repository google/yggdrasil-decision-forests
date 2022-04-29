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

// Abstract classes for models and model builders (called learners).
//
// FutureWork(gbm): Make this file and the "AbstractModel" minimalistic. Move
// the help methods in a separate file e.g. "abstract_model_utils".

#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_ABSTRACT_MODEL_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_ABSTRACT_MODEL_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/dataset/weight.pb.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.pb.h"
#include "yggdrasil_decision_forests/model/fast_engine_factory.h"
#include "yggdrasil_decision_forests/model/metadata.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"
#include "yggdrasil_decision_forests/utils/compatibility.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/random.h"
#include "yggdrasil_decision_forests/utils/registration.h"

namespace yggdrasil_decision_forests {
namespace model {

struct ModelIOOptions {
  // The prefix of files used by the model.
  //
  // For saving a model, if `file_prefix` is not set, an empty string is used.
  // For loading a model, if `file_prefix` is not set, the model prefix is
  // auto-detected (if possible) based on the existing files in the given
  // directory.
  absl::optional<std::string> file_prefix;
};

class AbstractModel {
 public:
  virtual ~AbstractModel() {}

  // It is likely that you want to use the function "SaveModel" from
  // "model_library.h" instead of this function.
  //
  // Save the model into a directory. The model controls the format of the model
  // (i.e. what file is written and what they contains) but it should not create
  // files called "header.pb" nor "data_spec.pb" (see kModelHeaderFileName and
  // kModelDataSpecFileName) as these filenames are reserved for the model meta
  // information.
  // If no file prefix is given through `io_options`, the empty string is used.
  virtual absl::Status Save(absl::string_view directory,
                            const ModelIOOptions& io_options) const = 0;

  // It is likely that you want to use the function "LoadModel" from
  // "model_library.h" instead of this function.
  //
  // Load the model from a directory. Should match the format created by "Save".
  // Derived classes may require a file prefix be given through `io_options`.
  virtual absl::Status Load(absl::string_view directory,
                            const ModelIOOptions& io_options) = 0;

  // Creates an inference engine able to run the model more efficiently
  // than by calling "Predict". Once the inference engine created, the model
  // can be discarded. If no inference engine is available for the model,
  // an error is returned. If multiple inference engines are available,
  // the faster one will be selected.
  //
  // Inference engines are added as separate dependencies. For example,
  // ../serving/decision_forest:register_engines contains multiple basic
  // inference engines for decision forest models.
  //
  // Because "BuildFastEngine" uses virtual calls, this solution is slower
  // than selecting directly the inference engine at compile time.
  utils::StatusOr<std::unique_ptr<serving::FastEngine>> BuildFastEngine() const;

  // List the fast engines compatible with the model.
  std::vector<std::unique_ptr<FastEngineFactory>> ListCompatibleFastEngines()
      const;

  // If set to "False", "BuildFastEngine" won't return an engine, even if one if
  // available.
  void SetAllowFastEngine(const bool allow_fast_engine) {
    allow_fast_engine_ = allow_fast_engine;
  }

  // Check that the model is valid. The inference on a non-valid model is non
  // defined.
  //
  // This function is called implicitly when importing and exporting a model.
  virtual absl::Status Validate() const;

  // Set the dataspec of the model.
  void set_data_spec(const dataset::proto::DataSpecification& v) {
    data_spec_ = v;
  }

  // Get the dataspec in the model.
  const dataset::proto::DataSpecification& data_spec() const {
    return data_spec_;
  }

  // Get the mutable dataspec in the model.
  dataset::proto::DataSpecification* mutable_data_spec() { return &data_spec_; }

  // Set the model's task.
  void set_task(const proto::Task task) { task_ = task; }

  // Get the task of the model.
  const proto::Task& task() const { return task_; }

  // Set the model target column.
  void set_label_col_idx(int label_col_idx) { label_col_idx_ = label_col_idx; }

  // Get the model target column.
  int label_col_idx() const { return label_col_idx_; }

  // Name of the label column.
  std::string label() const {
    CHECK_GE(label_col_idx_, 0);
    CHECK_LT(label_col_idx_, data_spec_.columns_size());
    return data_spec_.columns(label_col_idx_).name();
  }

  // Set the model ranking group column (e.g. query id).
  void set_ranking_group_col(int ranking_group_col_idx) {
    ranking_group_col_idx_ = ranking_group_col_idx;
  }

  // Get the model ranking group column.
  int ranking_group_col_idx() const { return ranking_group_col_idx_; }

  // Set the model uplift treatment column.
  void set_uplift_treatment_col(int uplift_treatment_col_idx) {
    uplift_treatment_col_idx_ = uplift_treatment_col_idx;
  }

  // Get the model uplift treatment column.
  int uplift_treatment_col_idx() const { return uplift_treatment_col_idx_; }

  // Column spec of the label.
  const dataset::proto::Column& label_col_spec() const {
    return data_spec().columns(label_col_idx());
  }

  // Get the weights used during training..
  absl::optional<dataset::proto::LinkedWeightDefinition> weights() const {
    return weights_;
  }

  // Set training weights.
  void set_weights(const dataset::proto::LinkedWeightDefinition& weights) {
    weights_ = weights;
  }

  // Column spec of the label.
  const dataset::proto::Column& LabelColumnSpec() const {
    return data_spec_.columns(label_col_idx_);
  }

  const std::string& name() const { return name_; }

  // Export an abstract model to a proto.
  static void ExportProto(const AbstractModel& model,
                          proto::AbstractModel* proto);

  // Load an abstract model from a proto.
  static void ImportProto(const proto::AbstractModel& proto,
                          AbstractModel* model);

  // Evaluates the model on a dataset. Returns a finalized EvaluationResults.
  //
  // If specified, "predictions" will be populated with the predictions.
  metric::proto::EvaluationResults Evaluate(
      const dataset::VerticalDataset& dataset,
      const metric::proto::EvaluationOptions& option, utils::RandomEngine* rnd,
      std::vector<model::proto::Prediction>* predictions = nullptr) const;

  // Evaluates the model on a dataset stored in disk. `typed_path` defines
  // the type and the path pattern of the files, as described in
  // `yggdrasil_decision_forests/datasets/format.h` file.
  // This method is preferable when the number of examples is large since they
  // do not have to be all first loaded into memory.
  // Returns a finalized EvaluationResults.
  // Evaluates the model on a dataset. Returns a finalized EvaluationResults.
  // The random generator "rnd" is used boostrapping of confidence intervals and
  // sub-sampling evaluation (if configured in "option").
  metric::proto::EvaluationResults Evaluate(
      const absl::string_view typed_path,
      const metric::proto::EvaluationOptions& option,
      utils::RandomEngine* rnd) const;

  // Similar to "Evaluate", but allow to override the evaluation objective.
  metric::proto::EvaluationResults EvaluateOverrideType(
      const dataset::VerticalDataset& dataset,
      const metric::proto::EvaluationOptions& option,
      const proto::Task override_task, const int override_label_col_idx,
      const int override_group_col_idx, utils::RandomEngine* rnd,
      std::vector<model::proto::Prediction>* predictions = nullptr) const;

  // Evaluates the model and appends the results to an initialized and
  // non-finalized EvaluationResults.
  //
  // If specified, "predictions" will be populated with the predictions.
  void AppendEvaluation(
      const dataset::VerticalDataset& dataset,
      const metric::proto::EvaluationOptions& option, utils::RandomEngine* rnd,
      metric::proto::EvaluationResults* eval,
      std::vector<model::proto::Prediction>* predictions = nullptr) const;

  // Similar as "AppendEvaluation" above. But operate on dataset stored on disk.
  // This method is preferable when the number of examples is large since they
  // do not have to be all loaded in memory as the same time.
  void AppendEvaluation(const absl::string_view typed_path,
                        const metric::proto::EvaluationOptions& option,
                        utils::RandomEngine* rnd,
                        metric::proto::EvaluationResults* eval) const;

  // Similar to "AppendEvaluation", but allow to override the evaluation
  // objective.
  void AppendEvaluationOverrideType(
      const dataset::VerticalDataset& dataset,
      const metric::proto::EvaluationOptions& option,
      const proto::Task override_task, const int override_label_col_idx,
      const int override_group_col_idx, utils::RandomEngine* rnd,
      metric::proto::EvaluationResults* eval,
      std::vector<model::proto::Prediction>* predictions = nullptr) const;

  // Generates the predictions of the model.
  void AppendPredictions(
      const dataset::VerticalDataset& dataset, const bool add_ground_truth,
      std::vector<model::proto::Prediction>* predictions) const;

  // Apply the model on an example defined as a VerticalDataset and a row
  // index. Requires for the dataset to have the same structure as the training
  // dataset. The model representation is expected to be generic and the
  // inference code is expected to be slower than the optimized serving code
  // available in "serving:all".
  //
  // Does not set the ground truth and the weight fields in "prediction".
  virtual void Predict(const dataset::VerticalDataset& dataset,
                       dataset::VerticalDataset::row_t row_idx,
                       proto::Prediction* prediction) const = 0;

  // Apply the model on a proto::Example. The model representation is expected
  // to be generic and the inference code is expected to be slower than the
  // optimized serving code available in "serving:all".
  //
  // "proto::Example" is the native generic example format for simple ml. This
  // is different from the "tensorflow::Example". Conversion from
  // "tensorflow::Example" to "proto::Example" can be done with the function
  // "TfExampleToExample".
  //
  // Does not set the ground truth and the weight fields in "prediction".
  virtual void Predict(const dataset::proto::Example& example,
                       proto::Prediction* prediction) const = 0;

  // Set the ground truth values in a Prediction proto. Ground truth values
  // can be defined by fields like label or example weight. This depends on the
  // model "type" (see Prediction proto). This step is required to evaluate the
  // prediction of one example (done in metric::AddPrediction in "metric.h").
  //
  // Both version requires that either the example (this version) or dataset
  // (next version) contains the label value.
  void SetGroundTruth(const dataset::proto::Example& example,
                      proto::Prediction* prediction) const;

  // Set the ground truth values (see description above) of one Prediction
  // proto from the specified row in the given dataset.
  void SetGroundTruth(const dataset::VerticalDataset& dataset,
                      dataset::VerticalDataset::row_t row_idx,
                      proto::Prediction* prediction) const;

  // Generates a human readable description of the statistics and structure of
  // the model. If "full_definition" is true, the entire model definition is
  // printed. In case of large model, this can represent a lot of data.
  virtual void AppendDescriptionAndStatistics(bool full_definition,
                                              std::string* description) const;

  // Simplified syntax to "AppendDescriptionAndStatistics".
  std::string DescriptionAndStatistics(bool full_definition = false) const;

  // Returns the list of the variable importance according to the model.
  //
  // When derived and in most cases, this function should merge the results
  // with its parent implementation.
  virtual std::vector<std::string> AvailableVariableImportances() const;

  // Returns a sorted list of variable importances (the most important first).
  // "key" should be an element of the result of "AvailableVariableImportances".
  //
  // Note: The model does not have to return a variable importance for all the
  // input features available at training time. If the model does not use a
  // feature, it does not have to return a variable importance for this feature.
  //
  // When derived, this function should also call its parent implementation.
  virtual utils::StatusOr<std::vector<proto::VariableImportance>>
  GetVariableImportance(absl::string_view key) const;

  // Create a user readable description of all the variable importance metrics
  // of the model.
  void AppendAllVariableImportanceDescription(std::string* description) const;

  // Evaluation of the performance of the model estimated during training.
  // Depending on the machine learning algorithm, the semantic of this
  // estimation can change.
  //
  // This evaluation (often called "validation") can be used to guide the
  // training and tuning of the model. For this reason, this evaluation is only
  // indicative and should not be used to compare models.
  virtual metric::proto::EvaluationResults ValidationEvaluation() const;

  // Estimates the memory usage of the model in RAM. The serialized or the
  // compiled version of the model can be much smaller.
  virtual absl::optional<size_t> ModelSizeInBytes() const { return {}; }

  // Estimates the memory usage of the attributes defined in the "AbstractModel"
  // object.
  size_t AbstractAttributesSizeInBytes() const;

  // List of input features of the model.
  const std::vector<int>& input_features() const { return input_features_; }
  std::vector<int>* mutable_input_features() { return &input_features_; }

  // Copy the meta data of the model i.e. the attributes common to all models.
  void CopyAbstractModelMetaData(AbstractModel* dst) const;

  absl::flat_hash_map<std::string, proto::VariableImportanceSet>*
  mutable_precomputed_variable_importances() {
    return &precomputed_variable_importances_;
  }

  const absl::flat_hash_map<std::string, proto::VariableImportanceSet>&
  precomputed_variable_importances() const {
    return precomputed_variable_importances_;
  }

  bool classification_outputs_probabilities() const {
    return classification_outputs_probabilities_;
  }

  void set_classification_outputs_probabilities(bool value) {
    classification_outputs_probabilities_ = value;
  }

  // Computes a set of variable importances available in
  // "AvailableVariableImportances", and store the result in the model. Querying
  // those variable importances will return the cached values (instead of
  // possibly re-computing those variables from the model structure).
  // If the variable is already cached, it will be ignored.
  absl::Status PrecomputeVariableImportances(
      const std::vector<std::string>& variable_importances);

  // Metadata accessors.
  //
  // Note: The use of "MetaData" (instead of "proto::MetaData") is a temporary
  // change. Do not depend on it.
  const MetaData& metadata() const { return metadata_; }
  MetaData* mutable_metadata() { return &metadata_; }

  // Hyperparameter tuning logs.
  const absl::optional<proto::HyperparametersOptimizerLogs>&
  hyperparameter_optimizer_logs() const {
    return hyperparameter_optimizer_logs_;
  }
  absl::optional<proto::HyperparametersOptimizerLogs>*
  mutable_hyperparameter_optimizer_logs() {
    return &hyperparameter_optimizer_logs_;
  }

 protected:
  explicit AbstractModel(const absl::string_view name) : name_(name) {}

  void AppendEvaluationWithEngine(
      const dataset::VerticalDataset& dataset,
      const metric::proto::EvaluationOptions& option,
      const dataset::proto::LinkedWeightDefinition& weight_links,
      const serving::FastEngine& engine, utils::RandomEngine* rnd,
      std::vector<model::proto::Prediction>* predictions,
      metric::proto::EvaluationResults* eval) const;

  // Prints informations about the hyper-parameter optimizer logs.
  void AppendHyperparameterOptimizerLogs(std::string* description) const;

  // Checks if the ModelIOOptions are sufficient to load the model.
  //
  // At this time, this function checks if a prefix if given.
  static absl::Status ValidateModelIOOptions(const ModelIOOptions& io_options);

  // A string uniquely identifying the model type . Used to determine
  // model types during serialization. This should match the registered names in
  // ":model_library".
  std::string name_;

  // Dataset specification.
  dataset::proto::DataSpecification data_spec_;

  // Modeling task (e.g. Classification, regression).
  proto::Task task_ = proto::Task::UNDEFINED;

  // Column idx of the label.
  int label_col_idx_ = -1;

  // Column index of groups (e.g. queries) in ranking.
  int ranking_group_col_idx_ = -1;

  // Column index of uplift treatment.
  int uplift_treatment_col_idx_ = -1;

  // Example weight used during training. If not specified, all the examples
  // have the same weight.
  absl::optional<dataset::proto::LinkedWeightDefinition> weights_;

  // Input features of the model.
  std::vector<int> input_features_;

  absl::flat_hash_map<std::string, proto::VariableImportanceSet>
      precomputed_variable_importances_;

  // Allow for fast engine to run.
  bool allow_fast_engine_ = true;

  // If true, the output of a task=CLASSIFICATION model is a probability and can
  // be used accordingly (e.g. averaged, clamped to [0,1]). If false, the output
  // of the task=CLASSIFICATION model might not be a probability.
  bool classification_outputs_probabilities_ = true;

  // TODO(gbm): Use proto::Metadata.
  // Note: Cannot use proto::Metadata with the version of protobuf linked by TF.
  MetaData metadata_;

  absl::optional<proto::HyperparametersOptimizerLogs>
      hyperparameter_optimizer_logs_;

  // Note: New fields should be registered in:
  // - The proto serialization functions.
  // - The "CopyAbstractModelMetaData" method.
};

REGISTRATION_CREATE_POOL(AbstractModel);

#define REGISTER_AbstractModel(name, key) \
  REGISTRATION_REGISTER_CLASS(name, key, AbstractModel)

// Set the ground truth in a "proto::Prediction".
//
// See the definition of AbstractModel::SetGroundTruth(...) for mode details.
// Unlike AbstractModel::SetGroundTruth, the two following SetGroundTruth
// functions do not require for a model to exist. When possible use
// "model.SetGroundTruth(...)" instead.

// In case of non-ranking task (e.g. regression), "ranking_group_col_idx" should
// be set to  "kNoRankingGroup".
constexpr int kNoRankingGroup = -1;
constexpr int kNoUpliftTreatmentGroup = -1;

// Indices of the columns needed to set the ground truth.
struct GroundTruthColumnIndices {
  // These fields correspond to the fields defined in AbstractModel.
  const int label_col_idx;
  const int group_col_idx;
  const int uplift_treatment_col_idx;

  GroundTruthColumnIndices(
      const int label_col_idx, const int group_col_idx = kNoRankingGroup,
      const int uplift_treatment_col_idx = kNoUpliftTreatmentGroup)
      : label_col_idx(label_col_idx),
        group_col_idx(group_col_idx),
        uplift_treatment_col_idx(uplift_treatment_col_idx) {}
};

// Note: The "task" defines how the label are interpreted and how the
// predictions are evaluated. The task should correspond to the model emitting
// the predictions, or be compatible with it (e.g. a ranking model can be
// evaluated with task=REGRESSION).

// See comments above.
void SetGroundTruth(const dataset::VerticalDataset& dataset,
                    dataset::VerticalDataset::row_t row_idx,
                    const GroundTruthColumnIndices& columns, proto::Task task,
                    proto::Prediction* prediction);

// See comments above.
void SetGroundTruth(const dataset::proto::Example& example,
                    const GroundTruthColumnIndices& columns, proto::Task task,
                    proto::Prediction* prediction);

// Converts a prediction from one type to another.
void ChangePredictionType(proto::Task src_task, proto::Task dst_task,
                          const proto::Prediction& src_pred,
                          proto::Prediction* dst_pred);

// Create a user readable description of the set of the variable importances of
// a model as returned by "GetVariableImportance".
void AppendVariableImportanceDescription(
    const std::vector<proto::VariableImportance>& variable_importances,
    const dataset::proto::DataSpecification& data_spec,
    const int leading_spaces, std::string* description);

// Merge the variable importance of "src" to the variable importances of "dst".
// The final variable importance is: src * weight_src + dst * (1 - weight_src).
// If an item is not present in "src" or "dst", its importance is assumed to be
// 0 for this container. The output "dst" is sorted in decreasing order of
// importance.
void MergeVariableImportance(const std::vector<proto::VariableImportance>& src,
                             double weight_src,
                             std::vector<proto::VariableImportance>* dst);

// Content accumulator for predictions.
// The final prediction is defined as \sum_i src_factor_i * src_i, where "i"
// correspond to the successive calls to "Add".
class PredictionMerger {
 public:
  // Initialize the merged with a target prediction.
  explicit PredictionMerger(proto::Prediction* dst) : dst_(dst) {}

  // Add a prediction to dst. Note: "dst" should not be used before "Merge" is
  // called.
  void Add(const proto::Prediction& src, float src_factor);

  // Finalize the addition of the predictions. Should be called before "dst" is
  // used.
  void Merge();

  // "Scales" the predictions. This is similar to multiply all the "src_factor"
  // of the "Add" method by the "scale" parameter.
  //
  // Scaling all the predictions have a different semantic for different tasks
  // but can always be understood as the "accumulation" of the predictions from
  // different sub-predictions.
  //
  //   Classification: Has no effect (multiply the numerator and denominator
  //     used to compute the final probabilities).
  //   Regression: Multiplies the prediction value by "scale".
  //   Ranking:  Multiplies the prediction value by
  //   "scale". Does not impact the predicted ranking.
  static void ScalePrediction(float scale, proto::Prediction* dst);

 private:
  proto::Prediction* dst_;
};

// Converts a prediction generated by a fast engine into a proto Prediction.
void FloatToProtoPrediction(const std::vector<float>& src_prediction,
                            int example_idx, const proto::Task task,
                            int num_prediction_dimensions,
                            proto::Prediction* dst_prediction);

}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_ABSTRACT_MODEL_H_
