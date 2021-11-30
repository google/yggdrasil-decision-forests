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

// Semi-fast generic inference engine which does not require for the model to
// known as compilation time and is faster than simpleML's "generic inference"
// engine.
//
// If inference speed is a central concern and the model can be known at compile
// time, use one of the specialized inference engines instead.
//
// Usage example:
//   std::unique_ptr<AbstractModel> model = ...
//   const auto engine = model.BuildFastEngine();
//   const auto examples = engine->AllocateExamples(5);
//   examples->SetNumericalFeature(...);
//   std::vector<float> predictions;
//   engine.Predict(examples, 1, &predictions);

#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_FAST_ENGINE_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_FAST_ENGINE_H_

#include "yggdrasil_decision_forests/serving/example_set.h"

namespace yggdrasil_decision_forests {
namespace serving {

class FastEngine {
 public:
  virtual ~FastEngine() = default;

  // Allocates a set of examples. The "num_examples" argument of the "Predict"
  // method should be less or equal to the "num_examples" of "AllocateExamples".
  //
  // After being allocated, examples are in an undefined state, and the user is
  // expected to set (i.e. using the "Set*" functions) all the feature values
  // before calling the "Predict" function. If a feature value is missing, you
  // should set it as missing using the "SetMissing*" functions.
  //
  // Example (simplified):
  //   Allocate
  //   SetNumerical("a", 5.f)
  //   SetNumerical("b", 1.f)
  //   SetMissingNumerical("c")
  //   SetMissingNumerical("d")
  //   Predict
  //
  // The "FillMissing" function sets all the features values to missing. You can
  // also call the "FillMissing" function after the example allocation (or in
  // between Prediction calls, if you are re-using the same allocated examples),
  // and before setting any feature value to set all the features to an initial
  // state of missing. "FillMissing" has a cost, but it is more efficient than
  // calling "SetMissing" on all features individually.
  //
  // Example (equivalent to the previous one; simplified):
  //   Allocate
  //   FillMissing()
  //   SetNumerical("a", 5.f)
  //   SetNumerical("b", 1.f)
  //   Predict
  //
  virtual std::unique_ptr<AbstractExampleSet> AllocateExamples(
      int num_examples) const = 0;

  // Applies the model on a set of examples.
  // After the function call, "predictions" will be of size "num_examples *
  // NumPredictionDimension()".
  virtual void Predict(const AbstractExampleSet& examples, int num_examples,
                       std::vector<float>* predictions) const = 0;

  // Applies the model on a set of examples and returns the index of the active
  // leaf of each tree.
  //
  // This method should be called with "leaves" containing exactly "num_rows x
  // num_trees" elements. The leaf indices are stored in example-major
  // tree-minor.
  virtual absl::Status GetLeaves(const AbstractExampleSet& examples,
                                 const int num_examples,
                                 absl::Span<int32_t> leaves) const {
    return absl::UnimplementedError("GetLeaves not implemented");
  }

  // Number of dimensions of the output predictions.
  // 1 for regression, ranking and binary classification with compact format.
  // number of classes for classification.
  virtual int NumPredictionDimension() const = 0;

  // List of features used by the model.
  virtual const serving::FeaturesDefinition& features() const = 0;
};

}  // namespace serving
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_FAST_ENGINE_H_
