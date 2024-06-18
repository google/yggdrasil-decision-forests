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

// Isolation Forest learner.
#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_ISOLATION_FOREST_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_ISOLATION_FOREST_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "yggdrasil_decision_forests/dataset/types.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.h"
#include "yggdrasil_decision_forests/learner/abstract_learner.pb.h"
#include "yggdrasil_decision_forests/learner/isolation_forest/isolation_forest.pb.h"
#include "yggdrasil_decision_forests/model/abstract_model.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"
#include "yggdrasil_decision_forests/utils/hyper_parameters.h"
#include "yggdrasil_decision_forests/utils/random.h"

namespace yggdrasil_decision_forests::model::isolation_forest {

class IsolationForestLearner : public AbstractLearner {
 public:
  explicit IsolationForestLearner(
      const model::proto::TrainingConfig& training_config);

  inline static constexpr char kRegisteredName[] = "ISOLATION_FOREST";
  inline static constexpr char kHParamNumTrees[] = "num_trees";
  inline static constexpr char kHParamSubsampleRatio[] = "subsample_ratio";
  inline static constexpr char kHParamSubsampleCount[] = "subsample_count";
  // TODO: Add all hyper-parameters.

  absl::StatusOr<std::unique_ptr<AbstractModel>> TrainWithStatusImpl(
      const dataset::VerticalDataset& train_dataset,
      absl::optional<std::reference_wrapper<const dataset::VerticalDataset>>
          valid_dataset) const override;

  absl::Status SetHyperParametersImpl(
      utils::GenericHyperParameterConsumer* generic_hyper_params) override;

  absl::StatusOr<model::proto::GenericHyperParameterSpecification>
  GetGenericHyperParameterSpecification() const override;

  model::proto::LearnerCapabilities Capabilities() const override {
    model::proto::LearnerCapabilities capabilities;
    capabilities.set_require_label(false);
    return capabilities;
  }
};

REGISTER_AbstractLearner(IsolationForestLearner,
                         IsolationForestLearner::kRegisteredName);

namespace internal {

struct Configuration {
  model::proto::TrainingConfig training_config;
  model::proto::TrainingConfigLinking config_link;
  // "if_config" is a non-owning pointer to a sub-component of
  // "training_config".
  proto::IsolationForestTrainingConfig* if_config = nullptr;
};

// Gets the number of examples used to grow each tree.
SignedExampleIdx GetNumExamplesPerTrees(
    const proto::IsolationForestTrainingConfig& if_config,
    SignedExampleIdx num_training_examples);

// Sample examples to grow a tree.
std::vector<UnsignedExampleIdx> SampleExamples(
    UnsignedExampleIdx num_examples, UnsignedExampleIdx num_examples_to_sample,
    utils::RandomEngine* rnd);

// Default maximum depth hyper-parameter according to the number of examples
// used to grow each tree.
int DefaultMaximumDepth(UnsignedExampleIdx num_examples_per_trees);

// Finds a split (i.e. condition) for a node.
//
// A split is randomly sampled and returned.
// A valid split always branches one training examples in each branch. If not
// valid split can be generated, "FindSplit" returns false and not split is set.
// If a valid split is sampled, the condition of "node" is set and the function
// returns true.
//
// This function currently only implement the original isolation forest
// algorithm: Only split of the form "X >= threshold" are generated. The
// threshold is uniformly sampled between the minimum and maximum values
// observed in the training examples reaching this node
absl::StatusOr<bool> FindSplit(
    const Configuration& config, const dataset::VerticalDataset& train_dataset,
    const std::vector<UnsignedExampleIdx>& selected_examples,
    decision_tree::NodeWithChildren* node, utils::RandomEngine* rnd);

}  // namespace internal

}  // namespace yggdrasil_decision_forests::model::isolation_forest
#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_ISOLATION_FOREST_H_
