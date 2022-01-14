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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/training.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/decision_tree/splitter_accumulator.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache/dataset_cache_common.h"
#include "yggdrasil_decision_forests/learner/distributed_decision_tree/splitter.h"
#include "yggdrasil_decision_forests/utils/bitmap.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace {

// Creates a vector of accumulator initializer. One for each open node.
template <typename AccumulatorInitializer>
utils::StatusOr<std::vector<AccumulatorInitializer>>
CreateAccumulatorInitializerList(
    const FindBestSplitsCommonArgs& common,
    utils::StatusOr<AccumulatorInitializer> (AbstractLabelAccessor::*creator)(
        const decision_tree::proto::LabelStatistics&) const) {
  std::vector<AccumulatorInitializer> initializers;
  initializers.reserve(common.label_stats_per_open_node.size());
  for (int open_node_idx = 0;
       open_node_idx < common.label_stats_per_open_node.size();
       open_node_idx++) {
    ASSIGN_OR_RETURN(auto initializer,
                     (common.label_accessor.*creator)(
                         common.label_stats_per_open_node[open_node_idx]));
    initializers.push_back(std::move(initializer));
  }
  return initializers;
}

// Creates a bitmap indicating that a node is a "target node" i.e. that the
// "feature" is a candidate feature for this node.
std::vector<bool> BuildTargetNodeMap(
    const std::vector<std::vector<int>>& features_per_open_node,
    const FeatureIndex feature) {
  std::vector<bool> is_target_node(features_per_open_node.size());
  for (int node_idx = 0; node_idx < features_per_open_node.size(); node_idx++) {
    const auto& node_features = features_per_open_node[node_idx];
    is_target_node[node_idx] =
        std::find(node_features.begin(), node_features.end(), feature) !=
        node_features.end();
  }
  return is_target_node;
}

// Collections the label statistics of all the training examples.
template <typename Filler>
absl::Status TemplatedAggregateLabelStatistics(
    const Filler& filler, decision_tree::proto::LabelStatistics* label_stats,
    utils::concurrency::ThreadPool* thread_pool) {
  const size_t num_examples = filler.num_examples();

  const size_t num_threads = thread_pool->num_threads();
  std::vector<typename Filler::Accumulator> accumulators(num_threads);
  for (size_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    filler.InitializeAndZeroAccumulator(&accumulators[thread_idx]);
  }

  decision_tree::ConcurrentForLoop(
      num_threads, thread_pool, num_examples,
      [&accumulators, &filler](size_t block_idx, size_t begin_idx,
                               size_t end_idx) -> void {
        auto& accumulator = accumulators[block_idx];
        for (ExampleIndex example_idx = begin_idx; example_idx < end_idx;
             example_idx++) {
          filler.Add(example_idx, &accumulator);
        }
      });

  /*
   utils::concurrency::BlockingCounter blocker(num_threads);
    size_t begin_idx = 0;
    const size_t block_size = (num_examples + num_threads - 1) / num_threads;
    for (size_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
      const auto end_idx = std::min(begin_idx + block_size, num_examples);
      thread_pool->Schedule([thread_idx, &accumulators, &filler, begin_idx,
                             end_idx, &blocker]() -> void {
        auto& accumulator = accumulators[thread_idx];
        for (ExampleIndex example_idx = begin_idx; example_idx < end_idx;
             example_idx++) {
          filler.Add(example_idx, &accumulator);
        }
        blocker.DecrementCount();
      });
      begin_idx += block_size;
    }
    blocker.Wait();
  */

  for (size_t thread_idx = 1; thread_idx < num_threads; thread_idx++) {
    accumulators[thread_idx].Add(&accumulators.front());
  }

  accumulators.front().ExportLabelStats(label_stats);
  label_stats->set_num_examples(num_examples);
  return absl::OkStatus();
}

}  // namespace

ExampleToNodeMap CreateExampleToNodeMap(ExampleIndex num_examples) {
  return ExampleToNodeMap(num_examples, 0);
}

utils::StatusOr<std::unique_ptr<TreeBuilder>> TreeBuilder::Create(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const decision_tree::proto::DecisionTreeTrainingConfig& dt_config,
    const LabelAccessorType label_accessor_type,
    SetLeafValueFromLabelStatsFunctor set_leaf_functor) {
  auto in_construction_tree = absl::WrapUnique<TreeBuilder>(new TreeBuilder(
      config, config_link, dt_config, label_accessor_type, set_leaf_functor));
  in_construction_tree->tree_.CreateRoot();
  in_construction_tree->open_nodes_.push_back(
      in_construction_tree->tree_.mutable_root());
  return in_construction_tree;
}

TreeBuilder::TreeBuilder(
    const model::proto::TrainingConfig& config,
    const model::proto::TrainingConfigLinking& config_link,
    const decision_tree::proto::DecisionTreeTrainingConfig& dt_config,
    const LabelAccessorType label_accessor_type,
    SetLeafValueFromLabelStatsFunctor set_leaf_functor)
    : config_(config),
      config_link_(config_link),
      dt_config_(dt_config),
      label_accessor_type_(label_accessor_type),
      set_leaf_functor_(set_leaf_functor) {}

absl::Status TreeBuilder::AggregateLabelStatistics(
    const AbstractLabelAccessor& label_accessor,
    decision_tree::proto::LabelStatistics* label_stats,
    utils::concurrency::ThreadPool* thread_pool) const {
  return distributed_decision_tree::AggregateLabelStatistics(
      label_accessor, config_.task(), label_accessor_type_, label_stats,
      thread_pool);
}

absl::Status AggregateLabelStatistics(
    const AbstractLabelAccessor& label_accessor, const model::proto::Task task,
    const LabelAccessorType label_accessor_type,
    decision_tree::proto::LabelStatistics* label_stats,
    utils::concurrency::ThreadPool* thread_pool) {
  switch (task) {
    case model::proto::Task::CLASSIFICATION: {
      if (label_accessor_type != LabelAccessorType::kAutomatic) {
        return absl::InternalError("Unexpected label accessor");
      }
      ASSIGN_OR_RETURN(auto filler,
                       label_accessor.CreateClassificationLabelFiller());
      return TemplatedAggregateLabelStatistics<>(filler, label_stats,
                                                 thread_pool);
    }
    case model::proto::Task::REGRESSION: {
      switch (label_accessor_type) {
        case LabelAccessorType::kAutomatic: {
          ASSIGN_OR_RETURN(auto filler,
                           label_accessor.CreateRegressionLabelFiller());
          return TemplatedAggregateLabelStatistics<>(filler, label_stats,
                                                     thread_pool);
        }
        case LabelAccessorType::kNumericalWithHessian: {
          ASSIGN_OR_RETURN(
              auto filler,
              label_accessor.CreateRegressionWithHessianLabelFiller());
          return TemplatedAggregateLabelStatistics<>(filler, label_stats,
                                                     thread_pool);
        }
        default:
          return absl::InternalError("Unexpected label accessor");
      }
    }
    default:
      return absl::InvalidArgumentError("Non supported task");
  }
}

absl::Status TreeBuilder::FindBestSplits(
    const FindBestSplitsCommonArgs& common) const {
  if (open_nodes_.size() != common.features_per_open_node.size()) {
    return absl::InternalError("Wrong number of elements");
  }

  // Allocates the splits.
  common.best_splits->assign(open_nodes_.size(), {});

  // List all the features tested by at least one node.
  absl::flat_hash_set<FeatureIndex> all_features;
  for (const auto& features : common.features_per_open_node) {
    all_features.insert(features.begin(), features.end());
  }

  // Find the best split per node and per feature.
  for (const FeatureIndex feature : all_features) {
    RETURN_IF_ERROR(
        FindBestSplitsWithFeature(common, feature, /*num_threads=*/1));
  }

  return absl::OkStatus();
}

absl::Status TreeBuilder::FindBestSplitsWithThreadPool(
    const FindBestSplitsCommonArgs& common,
    const std::vector<int>& unique_active_features,
    utils::concurrency::ThreadPool* thread_pool,
    utils::concurrency::Mutex* mutex,
    utils::concurrency::BlockingCounter* counter, absl::Status* status) const {
  if (open_nodes_.size() != common.features_per_open_node.size()) {
    return absl::InternalError("Wrong number of elements");
  }

  DCHECK(thread_pool != nullptr);

  // Allocates the splits.
  common.best_splits->assign(open_nodes_.size(), {});

  if (unique_active_features.empty()) {
    return absl::OkStatus();
  }
  const auto sub_num_threads = std::max<int>(
      1, (thread_pool->num_threads() + unique_active_features.size() - 1) /
             unique_active_features.size());

  // Find the best split per node and per feature.
  absl::Status worker_status;
  for (const FeatureIndex feature : unique_active_features) {
    thread_pool->Schedule([/*ptr*/ status, /*ptr*/ mutex, /*value*/ feature,
                           /*value*/ common, /*ptr*/ counter, this,
                           sub_num_threads]() {
      // Did another worker already failed?
      {
        utils::concurrency::MutexLock l(mutex);
        if (!status->ok()) {
          return;
        }
      }

      // Find the split.

      // Copy of common arguments to have an independent
      // space for the output argument "best_splits".
      auto local_common = common;
      SplitPerOpenNode local_splits(open_nodes_.size());
      local_common.best_splits = &local_splits;

      const auto local_status =
          FindBestSplitsWithFeature(local_common, feature, sub_num_threads);

      // Merge the result.
      {
        utils::concurrency::MutexLock l(mutex);
        status->Update(local_status);
        if (local_status.ok()) {
          status->Update(
              MergeBestSplits(*local_common.best_splits, common.best_splits));
        }
        // Note: Making sure the mutex lock is destroyed before the mutex.
      }
      counter->DecrementCount();
    });
  }

  return absl::OkStatus();
}

absl::Status TreeBuilder::FindBestSplitsWithFeature(
    const FindBestSplitsCommonArgs& common, const FeatureIndex feature,
    int num_theads) const {
  // Compute the set of target nodes.
  const auto is_target_node =
      BuildTargetNodeMap(common.features_per_open_node, feature);

  const auto column_spec = common.dataspec.columns(feature);

  if (dt_config_.has_sparse_oblique_split()) {
    return absl::InvalidArgumentError(
        "Oblique splits not implemented with distributed training. Disable "
        "oblique splits (i.e. sparse_oblique_split=false) or disable "
        "distributed training.");
  }
  if (dt_config_.numerical_split().type() !=
      decision_tree::proto::NumericalSplit::EXACT) {
    return absl::InvalidArgumentError(
        "Non-exact numerical splits not implemented. Force exact splits (i.e. "
        "numerical_split=EXACT) or disable distributed training.");
  }

  switch (column_spec.type()) {
    case dataset::proto::ColumnType::NUMERICAL:
      if (common.dataset->meta_data()
              .columns(feature)
              .numerical()
              .discretized()) {
        RETURN_IF_ERROR(FindBestSplitsWithFeatureDiscretizedNumerical(
            common, feature, is_target_node, num_theads));
      } else {
        RETURN_IF_ERROR(FindBestSplitsWithFeatureSortedNumerical(
            common, feature, is_target_node));
      }
      break;

    case dataset::proto::ColumnType::CATEGORICAL:
      RETURN_IF_ERROR(FindBestSplitsWithFeatureCategorical(common, feature,
                                                           is_target_node));
      break;

    case dataset::proto::ColumnType::BOOLEAN:
      RETURN_IF_ERROR(
          FindBestSplitsWithFeatureBoolean(common, feature, is_target_node));
      break;

    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Feature type ", dataset::proto::ColumnType_Name(column_spec.type()),
          " not implemented for feature \"", column_spec.name(), "\""));
  }
  return absl::OkStatus();
}

absl::Status TreeBuilder::FindBestSplitsWithFeatureSortedNumerical(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node) const {
  switch (config_.task()) {
    case model::proto::CLASSIFICATION: {
      if (label_accessor_type_ != LabelAccessorType::kAutomatic) {
        return absl::InternalError("Unexpected label accessor");
      }
      ASSIGN_OR_RETURN(const auto filler,
                       common.label_accessor.CreateClassificationLabelFiller());
      ASSIGN_OR_RETURN(
          const auto initializers,
          CreateAccumulatorInitializerList<>(
              common, &AbstractLabelAccessor::
                          CreateClassificationAccumulatorInitializer));
      return TemplatedFindBestSplitsWithSortedNumericalFeature<>(
          common, feature, is_target_node, filler, initializers,
          dt_config_.min_examples());
    }

    case model::proto::REGRESSION: {
      switch (label_accessor_type_) {
        case LabelAccessorType::kAutomatic: {
          ASSIGN_OR_RETURN(const auto filler,
                           common.label_accessor.CreateRegressionLabelFiller());
          ASSIGN_OR_RETURN(
              const auto initializers,
              CreateAccumulatorInitializerList<>(
                  common, &AbstractLabelAccessor::
                              CreateRegressionAccumulatorInitializer));
          return TemplatedFindBestSplitsWithSortedNumericalFeature<>(
              common, feature, is_target_node, filler, initializers,
              dt_config_.min_examples());
        }

        case LabelAccessorType::kNumericalWithHessian: {
          ASSIGN_OR_RETURN(
              const auto filler,
              common.label_accessor.CreateRegressionWithHessianLabelFiller());
          ASSIGN_OR_RETURN(
              const auto initializers,
              CreateAccumulatorInitializerList<>(
                  common,
                  &AbstractLabelAccessor::
                      CreateRegressionWithHessianAccumulatorInitializer));
          return TemplatedFindBestSplitsWithSortedNumericalFeature<>(
              common, feature, is_target_node, filler, initializers,
              dt_config_.min_examples());
        }

        default:
          return absl::InternalError("Unexpected label accessor");
      }
    }

    default:
      return absl::InvalidArgumentError(
          absl::Substitute("The task $0 is not supported for numerical "
                           "features and distributed training. The "
                           "supported tasks are CLASSIFICATION, REGRESSION.",
                           model::proto::Task_Name(config_.task())));
  }
}

absl::Status TreeBuilder::FindBestSplitsWithFeatureDiscretizedNumerical(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node, int num_threads) const {
  switch (config_.task()) {
    case model::proto::CLASSIFICATION: {
      if (label_accessor_type_ != LabelAccessorType::kAutomatic) {
        return absl::InternalError("Unexpected label accessor");
      }
      ASSIGN_OR_RETURN(const auto filler,
                       common.label_accessor.CreateClassificationLabelFiller());
      ASSIGN_OR_RETURN(
          const auto initializers,
          CreateAccumulatorInitializerList<>(
              common, &AbstractLabelAccessor::
                          CreateClassificationAccumulatorInitializer));
      return TemplatedFindBestSplitsWithDiscretizedNumericalFeatureMultiThreading<>(
          common, feature, is_target_node, filler, initializers,
          dt_config_.min_examples(), num_threads);
    }

    case model::proto::REGRESSION: {
      switch (label_accessor_type_) {
        case LabelAccessorType::kAutomatic: {
          ASSIGN_OR_RETURN(const auto filler,
                           common.label_accessor.CreateRegressionLabelFiller());
          ASSIGN_OR_RETURN(
              const auto initializers,
              CreateAccumulatorInitializerList<>(
                  common, &AbstractLabelAccessor::
                              CreateRegressionAccumulatorInitializer));
          return TemplatedFindBestSplitsWithDiscretizedNumericalFeatureMultiThreading<>(
              common, feature, is_target_node, filler, initializers,
              dt_config_.min_examples(), num_threads);
        }

        case LabelAccessorType::kNumericalWithHessian: {
          ASSIGN_OR_RETURN(
              const auto filler,
              common.label_accessor.CreateRegressionWithHessianLabelFiller());
          ASSIGN_OR_RETURN(
              const auto initializers,
              CreateAccumulatorInitializerList<>(
                  common,
                  &AbstractLabelAccessor::
                      CreateRegressionWithHessianAccumulatorInitializer));
          return TemplatedFindBestSplitsWithDiscretizedNumericalFeatureMultiThreading<>(
              common, feature, is_target_node, filler, initializers,
              dt_config_.min_examples(), num_threads);
        }

        default:
          return absl::InternalError("Unexpected label accessor");
      }
    }

    default:
      return absl::InvalidArgumentError(
          absl::Substitute("The task $0 is not supported for discretized "
                           "numerical features and distributed training. The "
                           "supported tasks are CLASSIFICATION, REGRESSION.",
                           model::proto::Task_Name(config_.task())));
  }
  return absl::OkStatus();
}

absl::Status TreeBuilder::FindBestSplitsWithFeatureCategorical(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node) const {
  if (dt_config_.categorical().algorithm_case() !=
          decision_tree::proto::Categorical::kCart &&
      dt_config_.categorical().algorithm_case() !=
          decision_tree::proto::Categorical::ALGORITHM_NOT_SET) {
    return absl::InvalidArgumentError(
        "Only the CART categorical splitter is implemented for distributed "
        "training. Set categorical.algorithm_case=CART or disable distributed "
        "training.");
  }
  // TODO(gbm): Implement the random splitter.

  switch (config_.task()) {
    case model::proto::CLASSIFICATION: {
      if (label_accessor_type_ != LabelAccessorType::kAutomatic) {
        return absl::InternalError("Unexpected label accessor");
      }
      ASSIGN_OR_RETURN(const auto filler,
                       common.label_accessor.CreateClassificationLabelFiller());
      ASSIGN_OR_RETURN(
          const auto initializers,
          CreateAccumulatorInitializerList<>(
              common, &AbstractLabelAccessor::
                          CreateClassificationAccumulatorInitializer));
      return TemplatedFindBestSplitsWithClassificationAndCategoricalFeature<>(
          common, feature, is_target_node, filler, initializers,
          dt_config_.min_examples());
    }

    case model::proto::REGRESSION: {
      switch (label_accessor_type_) {
        case LabelAccessorType::kAutomatic: {
          ASSIGN_OR_RETURN(const auto filler,
                           common.label_accessor.CreateRegressionLabelFiller());
          ASSIGN_OR_RETURN(
              const auto initializers,
              CreateAccumulatorInitializerList<>(
                  common, &AbstractLabelAccessor::
                              CreateRegressionAccumulatorInitializer));
          return TemplatedFindBestSplitsWithRegressionAndCategoricalFeature<>(
              common, feature, is_target_node, filler, initializers,
              dt_config_.min_examples());
        }

        case LabelAccessorType::kNumericalWithHessian: {
          ASSIGN_OR_RETURN(
              const auto filler,
              common.label_accessor.CreateRegressionWithHessianLabelFiller());
          ASSIGN_OR_RETURN(
              const auto initializers,
              CreateAccumulatorInitializerList<>(
                  common,
                  &AbstractLabelAccessor::
                      CreateRegressionWithHessianAccumulatorInitializer));
          return TemplatedFindBestSplitsWithRegressionAndCategoricalFeature<>(
              common, feature, is_target_node, filler, initializers,
              dt_config_.min_examples());
        }

        default:
          return absl::InternalError("Unexpected label accessor");
      }
    }

    default:
      return absl::InvalidArgumentError(
          absl::Substitute("The task $0 is not supported for categorical "
                           "features and distributed training. The "
                           "supported tasks are CLASSIFICATION, REGRESSION.",
                           model::proto::Task_Name(config_.task())));
  }
}

absl::Status TreeBuilder::FindBestSplitsWithFeatureBoolean(
    const FindBestSplitsCommonArgs& common, FeatureIndex feature,
    const std::vector<bool>& is_target_node) const {
  switch (config_.task()) {
    case model::proto::CLASSIFICATION: {
      if (label_accessor_type_ != LabelAccessorType::kAutomatic) {
        return absl::InternalError("Unexpected label accessor");
      }
      ASSIGN_OR_RETURN(const auto filler,
                       common.label_accessor.CreateClassificationLabelFiller());
      ASSIGN_OR_RETURN(
          const auto initializers,
          CreateAccumulatorInitializerList<>(
              common, &AbstractLabelAccessor::
                          CreateClassificationAccumulatorInitializer));
      return TemplatedFindBestSplitsWithClassificationAndBooleanFeature<>(
          common, feature, is_target_node, filler, initializers,
          dt_config_.min_examples());
    }

    case model::proto::REGRESSION: {
      switch (label_accessor_type_) {
        case LabelAccessorType::kAutomatic: {
          ASSIGN_OR_RETURN(const auto filler,
                           common.label_accessor.CreateRegressionLabelFiller());
          ASSIGN_OR_RETURN(
              const auto initializers,
              CreateAccumulatorInitializerList<>(
                  common, &AbstractLabelAccessor::
                              CreateRegressionAccumulatorInitializer));
          return TemplatedFindBestSplitsWithRegressionAndBooleanFeature<>(
              common, feature, is_target_node, filler, initializers,
              dt_config_.min_examples());
        }

        case LabelAccessorType::kNumericalWithHessian: {
          ASSIGN_OR_RETURN(
              const auto filler,
              common.label_accessor.CreateRegressionWithHessianLabelFiller());
          ASSIGN_OR_RETURN(
              const auto initializers,
              CreateAccumulatorInitializerList<>(
                  common,
                  &AbstractLabelAccessor::
                      CreateRegressionWithHessianAccumulatorInitializer));
          return TemplatedFindBestSplitsWithRegressionAndBooleanFeature<>(
              common, feature, is_target_node, filler, initializers,
              dt_config_.min_examples());
        }

        default:
          return absl::InternalError("Unexpected label accessor");
      }
    }

    default:
      return absl::InvalidArgumentError(
          absl::Substitute("The task $0 is not supported for boolean features "
                           "and distributed training. The "
                           "supported tasks are CLASSIFICATION, REGRESSION.",
                           model::proto::Task_Name(config_.task())));
  }
}

absl::Status MergeBestSplits(const SplitPerOpenNode& src,
                             SplitPerOpenNode* const dst) {
  if (src.size() != dst->size()) {
    return absl::InternalError("Unexpected number of open nodes");
  }
  for (int split_idx = 0; split_idx < src.size(); split_idx++) {
    if (src[split_idx].condition.split_score() >
        (*dst)[split_idx].condition.split_score()) {
      (*dst)[split_idx] = src[split_idx];
    }
  }
  return absl::OkStatus();
}

int NumValidSplits(const SplitPerOpenNode& splits) {
  int num_splits = 0;
  for (const auto& split : splits) {
    if (IsSplitValid(split)) {
      num_splits++;
    }
  }
  return num_splits;
}

bool IsSplitValid(const Split& split) {
  return split.condition.has_condition();
}

absl::Status SetLeafValue(
    const decision_tree::proto::LabelStatistics& label_stats,
    decision_tree::proto::Node* leaf) {
  // This function creates the same label values as "SetLabelDistribution" in
  // "learner/decision_tree/training.cc". However, it uses a pre-computed label
  // statistic instead of a vertical dataset.

  switch (label_stats.type_case()) {
    case decision_tree::proto::LabelStatistics::kClassification:
      *leaf->mutable_classifier()->mutable_distribution() =
          label_stats.classification().labels();
      leaf->mutable_classifier()->set_top_value(
          utils::TopClass(label_stats.classification().labels()));
      break;

    case decision_tree::proto::LabelStatistics::kRegression:
      *leaf->mutable_regressor()->mutable_distribution() =
          label_stats.regression().labels();
      leaf->mutable_regressor()->set_top_value(
          utils::Mean(label_stats.regression().labels()));
      break;

    default:
      return absl::InternalError(
          "Label statistics no support by default SetLeafValue");

    case decision_tree::proto::LabelStatistics::TYPE_NOT_SET:
      return absl::InternalError("Empty label stats");
  }
  return absl::OkStatus();
}

NodeRemapping TreeBuilder::CreateClosingNodeRemapping() const {
  return NodeRemapping{open_nodes_.size(), {kClosedNode, kClosedNode}};
}

utils::StatusOr<NodeRemapping> TreeBuilder::ApplySplitToTree(
    const SplitPerOpenNode& splits) {
  if (open_nodes_.size() != splits.size()) {
    return absl::InternalError("Wrong number of internal nodes");
  }
  NodeRemapping remapping(open_nodes_.size());
  std::vector<decision_tree::NodeWithChildren*> new_open_nodes;
  for (int split_idx = 0; split_idx < splits.size(); split_idx++) {
    const auto& split = splits[split_idx];
    auto& node = *open_nodes_[split_idx];
    if (IsSplitValid(split)) {
      // Non-leaf node.

      node.CreateChildren();
      *node.mutable_node()->mutable_condition() = split.condition;
      node.mutable_node()->set_num_pos_training_examples_without_weight(
          split.condition.num_pos_training_examples_without_weight());
      remapping[split_idx] = {
          static_cast<NodeIndex>(new_open_nodes.size()),
          static_cast<NodeIndex>(new_open_nodes.size() + 1)};
      new_open_nodes.push_back(node.mutable_neg_child());
      new_open_nodes.push_back(node.mutable_pos_child());

      RETURN_IF_ERROR(
          set_leaf_functor_(splits[split_idx].label_statistics[0],
                            node.mutable_neg_child()->mutable_node()));
      RETURN_IF_ERROR(
          set_leaf_functor_(splits[split_idx].label_statistics[1],
                            node.mutable_pos_child()->mutable_node()));

      node.FinalizeAsNonLeaf(true, true);
    } else {
      // Turning the node into a leaf.
      remapping[split_idx] = {kClosedNode, kClosedNode};
      node.FinalizeAsLeaf(true);
    }
  }

  if (new_open_nodes.size() >= std::numeric_limits<NodeIndex>::max()) {
    return absl::InvalidArgumentError("Maximum node limit exceeded");
  }
  open_nodes_ = new_open_nodes;
  return remapping;
}

absl::Status TreeBuilder::SetRootValue(
    const decision_tree::proto::LabelStatistics& label_stats) {
  return set_leaf_functor_(label_stats, tree_.mutable_root()->mutable_node());
}

absl::Status EvaluateSplits(const ExampleToNodeMap& example_to_node,
                            const SplitPerOpenNode& splits,
                            SplitEvaluationPerOpenNode* split_evaluation,
                            dataset_cache::DatasetCacheReader* dataset,
                            utils::concurrency::ThreadPool* thread_pool) {
  // Group the split per feature.
  absl::flat_hash_map<FeatureIndex, std::vector<int>> split_per_feature;
  for (int split_idx = 0; split_idx < splits.size(); split_idx++) {
    const auto& split = splits[split_idx];
    if (!IsSplitValid(split)) {
      continue;
    }
    const FeatureIndex feature = split.condition.attribute();
    if (!dataset->has_feature(feature)) {
      continue;
    }
    split_per_feature[feature].push_back(split_idx);
  }

  split_evaluation->assign(splits.size(), {});

  const auto process =
      [&](const FeatureIndex feature_idx,
          const std::vector<int>& active_node_idxs) -> absl::Status {
    const auto& col_metadata = dataset->meta_data().columns(feature_idx);
    switch (col_metadata.type_case()) {
      case dataset_cache::proto::CacheMetadata_Column::kNumerical:
        RETURN_IF_ERROR(EvaluateSplitsPerNumericalFeature(
            example_to_node, splits, feature_idx, active_node_idxs,
            split_evaluation, dataset));
        break;

      case dataset_cache::proto::CacheMetadata_Column::kCategorical:
        RETURN_IF_ERROR(EvaluateSplitsPerCategoricalFeature(
            example_to_node, splits, feature_idx, active_node_idxs,
            split_evaluation, dataset));
        break;

      case dataset_cache::proto::CacheMetadata_Column::kBoolean:
        RETURN_IF_ERROR(EvaluateSplitsPerBooleanFeature(
            example_to_node, splits, feature_idx, active_node_idxs,
            split_evaluation, dataset));
        break;

      case dataset_cache::proto::CacheMetadata_Column::TYPE_NOT_SET:
        return absl::InternalError("Non set split");
    }
    return absl::OkStatus();
  };

  utils::concurrency::Mutex mutex;
  utils::concurrency::BlockingCounter blocker(split_per_feature.size());
  absl::Status status;
  for (const auto& feature_and_split_idx : split_per_feature) {
    thread_pool->Schedule([&, feature_idx = feature_and_split_idx.first,
                           &splits = feature_and_split_idx.second]() {
      auto local_status = process(feature_idx, splits);
      {
        utils::concurrency::MutexLock l(&mutex);
        status.Update(local_status);
      }
      blocker.DecrementCount();
    });
  }
  blocker.Wait();
  return status;
}

absl::Status EvaluateSplitsPerNumericalFeature(
    const ExampleToNodeMap& example_to_node, const SplitPerOpenNode& splits,
    FeatureIndex feature, const std::vector<int>& active_node_idxs,
    SplitEvaluationPerOpenNode* split_evaluation,
    dataset_cache::DatasetCacheReader* dataset) {
  // Initializer the active nodes.
  std::vector<int> node_idx_to_active_node_idx(splits.size(), -1);
  struct ActiveNode {
    utils::bitmap::BitWriter writer;
    float threshold;
    // Total number of expected elements.
    size_t num_elements = 0;

#ifndef NDEBUG
    // Only use to check the validity of the code.
    // Number of elements written so far.
    size_t num_written_elements = 0;
#endif
  };
  std::vector<ActiveNode> active_nodes;
  active_nodes.reserve(active_node_idxs.size());
  for (const auto node_idx : active_node_idxs) {
    node_idx_to_active_node_idx[node_idx] = active_nodes.size();

    const auto& condition = splits[node_idx].condition.condition();
    float threshold;
    switch (condition.type_case()) {
      case decision_tree::proto::Condition::kHigherCondition:
        threshold = splits[node_idx]
                        .condition.condition()
                        .higher_condition()
                        .threshold();
        DCHECK(!std::isnan(threshold));
        break;
      default:
        return absl::InternalError("Unexpected condition type");
    }

    const auto num_elements = static_cast<uint64_t>(
        splits[node_idx].condition.num_training_examples_without_weight());
    ActiveNode active_node{
        /*writer=*/{num_elements, &(*split_evaluation)[node_idx]},
        /*threshold=*/threshold,
        /*num_elements=*/num_elements,
    };
    active_node.writer.AllocateAndZeroBitMap();
    active_nodes.push_back(std::move(active_node));
  }

  // Scan the dataset and evaluate the split.
  ASSIGN_OR_RETURN(auto value_it,
                   dataset->InOrderNumericalFeatureValueIterator(feature));
  ExampleIndex example_idx = 0;
  while (true) {
    RETURN_IF_ERROR(value_it->Next());
    const auto values = value_it->Values();
    if (values.empty()) {
      break;
    }

    for (auto value : values) {
      const auto node_idx = example_to_node[example_idx];
      if (node_idx != kClosedNode) {
        const auto active_node_idx = node_idx_to_active_node_idx[node_idx];
        if (active_node_idx >= 0) {
          auto& active_node = active_nodes[active_node_idx];

#ifndef NDEBUG
          DCHECK_LT(active_node.num_written_elements, active_node.num_elements);
          active_node.num_written_elements++;
#endif
          active_node.writer.Write(value >= active_node.threshold);
        }
      }
      example_idx++;
    }
  }
  RETURN_IF_ERROR(value_it->Close());

  // Finalize the writers.
  for (auto& active_node : active_nodes) {
#ifndef NDEBUG
    DCHECK_EQ(active_node.num_written_elements, active_node.num_elements);
#endif
    active_node.writer.Finish();
  }

  return absl::OkStatus();
}

absl::Status EvaluateSplitsPerCategoricalFeature(
    const ExampleToNodeMap& example_to_node, const SplitPerOpenNode& splits,
    FeatureIndex feature, const std::vector<int>& active_node_idxs,
    SplitEvaluationPerOpenNode* split_evaluation,
    dataset_cache::DatasetCacheReader* dataset) {
  // Initializer the active nodes.
  std::vector<int> node_idx_to_active_node_idx(splits.size(), -1);
  struct ActiveNode {
    utils::bitmap::BitWriter writer;
    std::string elements_bitmap;
    // Total number of expected elements.
    size_t num_elements = 0;
  };

  const int num_possible_values =
      dataset->meta_data().columns(feature).categorical().num_values();

  std::vector<ActiveNode> active_nodes;
  active_nodes.reserve(active_node_idxs.size());
  for (const auto node_idx : active_node_idxs) {
    node_idx_to_active_node_idx[node_idx] = active_nodes.size();

    std::string elements_bitmap;
    const auto& condition = splits[node_idx].condition.condition();
    switch (condition.type_case()) {
      case decision_tree::proto::Condition::kContainsCondition:
        utils::bitmap::AllocateAndZeroBitMap(num_possible_values,
                                             &elements_bitmap);
        for (const auto attribute_value :
             condition.contains_condition().elements()) {
          DCHECK_GE(attribute_value, 0);
          DCHECK_LT(attribute_value, num_possible_values);
          utils::bitmap::SetValueBit(attribute_value, &elements_bitmap);
        }
        break;
      case decision_tree::proto::Condition::kContainsBitmapCondition:
        elements_bitmap =
            condition.contains_bitmap_condition().elements_bitmap();
        DCHECK_EQ((num_possible_values + 7) / 8, elements_bitmap.size());
        break;
      default:
        return absl::InternalError(
            "Unexpected condition type for categorical feature");
    }
    const auto num_elements = static_cast<uint64_t>(
        splits[node_idx].condition.num_training_examples_without_weight());
    ActiveNode active_node{
        /*writer=*/{num_elements, &(*split_evaluation)[node_idx]},
        /*elements_bitmap=*/elements_bitmap,
        /*num_elements=*/num_elements};
    active_node.writer.AllocateAndZeroBitMap();
    active_nodes.push_back(std::move(active_node));
  }

  // Scan the dataset and evaluate the split.
  ASSIGN_OR_RETURN(auto value_it,
                   dataset->InOrderCategoricalFeatureValueIterator(feature));
  ExampleIndex example_idx = 0;
  while (true) {
    RETURN_IF_ERROR(value_it->Next());
    const auto values = value_it->Values();
    if (values.empty()) {
      break;
    }

    for (auto value : values) {
      DCHECK_GE(value, 0);
      DCHECK_LT(value, num_possible_values);
      const auto node_idx = example_to_node[example_idx];
      if (node_idx != kClosedNode) {
        DCHECK_LT(node_idx, splits.size());
        const auto active_node_idx = node_idx_to_active_node_idx[node_idx];
        if (active_node_idx >= 0) {
          auto& active_node = active_nodes[active_node_idx];
          active_node.writer.Write(
              utils::bitmap::GetValueBit(active_node.elements_bitmap, value));
        }
      }
      example_idx++;
    }
  }
  RETURN_IF_ERROR(value_it->Close());

  // Finalize the writers.
  for (auto& active_node : active_nodes) {
    active_node.writer.Finish();
  }

  return absl::OkStatus();
}

absl::Status EvaluateSplitsPerBooleanFeature(
    const ExampleToNodeMap& example_to_node, const SplitPerOpenNode& splits,
    FeatureIndex feature, const std::vector<int>& active_node_idxs,
    SplitEvaluationPerOpenNode* split_evaluation,
    dataset_cache::DatasetCacheReader* dataset) {
  // Initializer the active nodes.
  std::vector<int> node_idx_to_active_node_idx(splits.size(), -1);
  struct ActiveNode {
    utils::bitmap::BitWriter writer;
    // Total number of expected elements.
    size_t num_elements = 0;
  };

  const int num_possible_values = 3;

  std::vector<ActiveNode> active_nodes;
  active_nodes.reserve(active_node_idxs.size());
  for (const auto node_idx : active_node_idxs) {
    node_idx_to_active_node_idx[node_idx] = active_nodes.size();

    std::string elements_bitmap;
    const auto& condition = splits[node_idx].condition.condition();
    switch (condition.type_case()) {
      case decision_tree::proto::Condition::kTrueValueCondition:
        // Nothing to do.
        break;
      default:
        return absl::InternalError(
            "Unexpected condition type for categorical feature");
    }
    const auto num_elements = static_cast<uint64_t>(
        splits[node_idx].condition.num_training_examples_without_weight());
    ActiveNode active_node{
        /*writer=*/{num_elements, &(*split_evaluation)[node_idx]},
        /*num_elements=*/num_elements};
    active_node.writer.AllocateAndZeroBitMap();
    active_nodes.push_back(std::move(active_node));
  }

  // Scan the dataset and evaluate the split.
  ASSIGN_OR_RETURN(auto value_it,
                   dataset->InOrderBooleanFeatureValueIterator(feature));
  ExampleIndex example_idx = 0;
  while (true) {
    RETURN_IF_ERROR(value_it->Next());
    const auto values = value_it->Values();
    if (values.empty()) {
      break;
    }

    for (auto value : values) {
      DCHECK_GE(value, 0);
      DCHECK_LT(value, num_possible_values);
      const auto node_idx = example_to_node[example_idx];
      if (node_idx != kClosedNode) {
        DCHECK_LT(node_idx, splits.size());
        const auto active_node_idx = node_idx_to_active_node_idx[node_idx];
        if (active_node_idx >= 0) {
          auto& active_node = active_nodes[active_node_idx];
          active_node.writer.Write(value == 1);
        }
      }
      example_idx++;
    }
  }
  RETURN_IF_ERROR(value_it->Close());

  // Finalize the writers.
  for (auto& active_node : active_nodes) {
    active_node.writer.Finish();
  }

  return absl::OkStatus();
}

absl::Status UpdateExampleNodeMap(
    const SplitPerOpenNode& splits,
    const SplitEvaluationPerOpenNode& split_evaluation,
    const NodeRemapping& node_remapping, ExampleToNodeMap* example_to_node,
    utils::concurrency::ThreadPool* thread_pool) {
  DCHECK_EQ(split_evaluation.size(), node_remapping.size());
  std::vector<utils::bitmap::BitReader> readers(split_evaluation.size());
  for (int node_idx = 0; node_idx < split_evaluation.size(); node_idx++) {
    const auto num_elements = static_cast<uint64_t>(
        splits[node_idx].condition.num_training_examples_without_weight());
    DCHECK_LE(num_elements, split_evaluation[node_idx].size() * 8);
    readers[node_idx].Open(split_evaluation[node_idx].data(), num_elements);
  }

  // TODO(gbm): In parallel.
  for (ExampleIndex example_idx = 0; example_idx < example_to_node->size();
       example_idx++) {
    auto& node_idx = (*example_to_node)[example_idx];
    if (node_idx == kClosedNode) {
      // The example is not is a closed node.
      continue;
    }
    DCHECK_GE(node_idx, 0);
    DCHECK_LT(node_idx, split_evaluation.size());

    if (node_remapping[node_idx].indices[0] == kClosedNode) {
      // The example is in a node that is closed during this iteration.
      node_idx = kClosedNode;
      continue;
    }
    const bool evaluation = readers[node_idx].Read();
    node_idx = node_remapping[node_idx].indices[evaluation];
  }

  for (auto& reader : readers) {
    reader.Finish();
  }

  return absl::OkStatus();
}

absl::Status UpdateLabelStatistics(const SplitPerOpenNode& splits,
                                   const NodeRemapping& node_remapping,
                                   LabelStatsPerNode* label_stats) {
  NodeIndex dst_num_nodes = 0;
  for (const auto& mapping : node_remapping) {
    for (const auto evaluation : {0, 1}) {
      const auto dst_node_idx = mapping.indices[evaluation];
      if (dst_node_idx != kClosedNode) {
        dst_num_nodes = std::max<NodeIndex>(
            dst_num_nodes, mapping.indices[evaluation] + NodeIndex{1});
      }
    }
  }
  label_stats->assign(dst_num_nodes, {});

  for (int src_node_idx = 0; src_node_idx < splits.size(); src_node_idx++) {
    for (const auto evaluation : {0, 1}) {
      const auto dst_node_idx =
          node_remapping[src_node_idx].indices[evaluation];
      if (dst_node_idx == kClosedNode) {
        continue;
      }
      DCHECK_GE(dst_node_idx, 0);
      DCHECK_LT(dst_node_idx, dst_num_nodes);
      (*label_stats)[dst_node_idx] =
          splits[src_node_idx].label_statistics[evaluation];
    }
  }
  return absl::OkStatus();
}

void ConvertFromProto(const proto::SplitPerOpenNode& src,
                      SplitPerOpenNode* dst) {
  dst->clear();
  dst->resize(src.splits_size());
  for (int split_idx = 0; split_idx < src.splits_size(); split_idx++) {
    const auto& src_split = src.splits(split_idx);
    auto& dst_split = (*dst)[split_idx];
    dst_split.condition = src_split.condition();
    dst_split.label_statistics[0] = src_split.label_statistics_neg();
    dst_split.label_statistics[1] = src_split.label_statistics_pos();
  }
}

void ConvertToProto(const SplitPerOpenNode& src, proto::SplitPerOpenNode* dst) {
  dst->mutable_splits()->Clear();
  dst->mutable_splits()->Reserve(src.size());
  for (int split_idx = 0; split_idx < src.size(); split_idx++) {
    const auto& src_split = src[split_idx];
    auto& dst_split = *dst->add_splits();
    *dst_split.mutable_condition() = src_split.condition;
    *dst_split.mutable_label_statistics_neg() = src_split.label_statistics[0];
    *dst_split.mutable_label_statistics_pos() = src_split.label_statistics[1];
  }
}

void ConvertToProto(const SplitPerOpenNode& src,
                    const std::vector<int>& split_idxs,
                    proto::SplitPerOpenNode* dst) {
  dst->mutable_splits()->Clear();
  dst->mutable_splits()->Reserve(src.size());
  for (int split_idx = 0; split_idx < src.size(); split_idx++) {
    dst->add_splits();
  }
  for (const auto split_idx : split_idxs) {
    const auto& src_split = src[split_idx];
    auto& dst_split = *dst->mutable_splits(split_idx);
    *dst_split.mutable_condition() = src_split.condition;
    *dst_split.mutable_label_statistics_neg() = src_split.label_statistics[0];
    *dst_split.mutable_label_statistics_pos() = src_split.label_statistics[1];
  }
}

}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
