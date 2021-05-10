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

// Utility class to convert a fast inference engine that consume ExampleSets
// into a fast generic engine deriving the "FastEngine" class.
//
#ifndef YGGDRASIL_DECISION_FORESTS_SERVING_EXAMPLE_SET_MODEL_WRAPPER_H_
#define YGGDRASIL_DECISION_FORESTS_SERVING_EXAMPLE_SET_MODEL_WRAPPER_H_

#include "absl/status/status.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/serving/example_set.h"
#include "yggdrasil_decision_forests/serving/fast_engine.h"

namespace yggdrasil_decision_forests {
namespace serving {

// Utility class to wrap a fast ExampleSet model into a FastGenericEngine.
template <typename Model,
          void (*PredictCall)(const Model&, const typename Model::ExampleSet&,
                              int, std::vector<float>*)>
class ExampleSetModelWrapper : public FastEngine {
 public:
  // Loads the model in the engine. The "src" model can be discarded after that.
  template <typename SourceModel>
  absl::Status LoadModel(const SourceModel& src) {
    return GenericToSpecializedModel(src, &model_);
  }

  std::unique_ptr<AbstractExampleSet> AllocateExamples(
      int num_examples) const override {
    return absl::make_unique<typename Model::ExampleSet>(num_examples, model_);
  }

  void Predict(const AbstractExampleSet& examples, int num_examples,
               std::vector<float>* predictions) const override {
    const auto& casted_examples =
        dynamic_cast<const typename Model::ExampleSet&>(examples);
    PredictCall(model_, casted_examples, num_examples, predictions);
  }

  template <class...>
  using void_t = void;

  template <class T, class = void_t<>>
  struct extract_num_classes {
    int operator()(const T& m) { return 1; }
  };

  template <class T>
  struct extract_num_classes<T, void_t<decltype(T::num_classes)>> {
    int operator()(const T& m) { return m.num_classes; }
  };

  int NumPredictionDimension() const override {
    return extract_num_classes<Model>()(model_);
  }

  const serving::FeaturesDefinition& features() const override {
    return model_.features();
  }

 private:
  Model model_;
};

}  // namespace serving
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_SERVING_EXAMPLE_SET_MODEL_WRAPPER_H_
