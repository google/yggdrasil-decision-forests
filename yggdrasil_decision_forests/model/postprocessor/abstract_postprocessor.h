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

#ifndef YGGDRASIL_DECISION_FORESTS_MODEL_POSTPROCESSOR_ABSTRACT_POSTPROCESSOR_H_
#define YGGDRASIL_DECISION_FORESTS_MODEL_POSTPROCESSOR_ABSTRACT_POSTPROCESSOR_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/dataset/example.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/metric/metric.pb.h"
#include "yggdrasil_decision_forests/model/postprocessor/postprocessor.pb.h"
#include "yggdrasil_decision_forests/model/prediction.pb.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace postprocessor {

class AbstractPostprocessor {
 public:
  virtual ~AbstractPostprocessor() = default;

  // Adds the postprocessor's output to the prediction for a given row in the
  // dataset and the prediction from the model. This method primarily exists to
  // decouple the internal, override-able implementation from the public API.
  void Process(
      const dataset::VerticalDataset& dataset,
      dataset::VerticalDataset::row_t row_idx,
      yggdrasil_decision_forests::model::proto::Prediction* prediction) const;

  // Adds the postprocessor's output to the prediction for a given example and
  // the prediction from the model. This method primarily exists to decouple the
  // internal, override-able implementation from the public API.
  void Process(
      const dataset::proto::Example& example,
      yggdrasil_decision_forests::model::proto::Prediction* prediction) const;

  void Process(const serving::AbstractExampleSet& example, int num_examples,
               std::vector<float>* predictions) const;

  // Exports the postprocessor information to a proto.
  void ExportProto(proto::Postprocessor* proto) const;

  // Appends a description of the postprocessor to the given string.
  void AppendDescription(std::string* description) const;

  absl::Status InitializeEvaluation(
      const metric::proto::EvaluationOptions& option,
      const dataset::proto::Column& label_column,
      metric::proto::EvaluationResults* eval);

  absl::Status FinalizeEvaluation(
      const metric::proto::EvaluationOptions& option,
      const dataset::proto::Column& label_column,
      metric::proto::EvaluationResults* eval);

  absl::Status AppendEvaluation(const metric::proto::EvaluationOptions& option,
                                const model::proto::Prediction& pred,
                                utils::RandomEngine* rnd,
                                metric::proto::EvaluationResults* eval) const;

  // Returns whether the postprocessor is enabled.
  bool enabled() const { return enabled_; }
  // Enables the postprocessor.
  void enable() { enabled_ = true; }
  // Disables the postprocessor.
  void disable() { enabled_ = false; }

 protected:
  // Internal implementation of the Process method for a VerticalDataset.
  virtual void ProcessImpl(const dataset::VerticalDataset& dataset,
                           dataset::VerticalDataset::row_t row_idx,
                           yggdrasil_decision_forests::model::proto::Prediction*
                               prediction) const = 0;

  // Internal implementation of the Process method for an Example.
  virtual void ProcessImpl(const dataset::proto::Example& example,
                           yggdrasil_decision_forests::model::proto::Prediction*
                               prediction) const = 0;
  virtual void ProcessImpl(const serving::AbstractExampleSet& example,
                           int num_examples,
                           std::vector<float>* predictions) const = 0;

  // Internal implementation of the ExportProto method.
  virtual void ExportProtoImpl(proto::Postprocessor* proto) const = 0;

  // Internal implementation of the AppendDescription method.
  virtual void AppendDescriptionImpl(std::string* description) const = 0;

  virtual absl::Status InitializeEvaluationImpl(
      const metric::proto::EvaluationOptions& option,
      const dataset::proto::Column& label_column,
      metric::proto::EvaluationResults* eval) = 0;

  virtual absl::Status FinalizeEvaluationImpl(
      const metric::proto::EvaluationOptions& option,
      const dataset::proto::Column& label_column,
      metric::proto::EvaluationResults* eval) = 0;

  virtual absl::Status AppendEvaluationImpl(
      const metric::proto::EvaluationOptions& option,
      const model::proto::Prediction& pred, utils::RandomEngine* rnd,
      metric::proto::EvaluationResults* eval) const = 0;

  bool enabled_ = true;
};

}  // namespace postprocessor
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_MODEL_POSTPROCESSOR_ABSTRACT_POSTPROCESSOR_H_
