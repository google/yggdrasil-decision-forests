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

#include "yggdrasil_decision_forests/learner/distributed_decision_tree/load_balancer/load_balancer.h"

#include "gmock/gmock.h"
#include "yggdrasil_decision_forests/utils/logging.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace distributed_decision_tree {
namespace {
using testing::ElementsAre;

TEST(LoadBalancer, StaticBalancing) {
  distributed_decision_tree::dataset_cache::proto::CacheMetadata
      cache_metadata = PARSE_TEST_PROTO(R"pb(
        columns { categorical { num_values: 15 } }
        columns { numerical { num_unique_values: 20 } }
        columns { numerical { discretized: true num_discretized_values: 10 } }
        columns { boolean {} }
      )pb");
  auto balancer = LoadBalancer::Create(/*features=*/{0, 1, 2, 3},
                                       /*num_workers=*/2, cache_metadata, {})
                      .value();
  EXPECT_THAT(balancer.WorkersPerFeatures(), ElementsAre(1, 0, 0, 1));
}

TEST(LoadBalancer, DynamicBalancing) {
  const dataset_cache::proto::CacheMetadata cache_metadata = PARSE_TEST_PROTO(
      R"pb(
        columns {
          available: true
          numerical { discretized: true num_discretized_values: 6 }
        }
        columns {}
        columns {
          available: true
          categorical { num_values: 9 }
        }
        columns {}
        columns {
          available: true
          categorical { num_values: 8 }
        }
        columns {}
        columns {
          available: true
          categorical { num_values: 7 }
        }
      )pb");

  const proto::LoadBalancerOptions options = PARSE_TEST_PROTO(
      R"pb(
        median_margin_ratio: 1.5
        max_unbalance_ratio: 2
        estimation_window_length: 3
        dynamic_balancing_frequency_iteration: 1
      )pb");

  auto balancer =
      LoadBalancer::Create(/*features=*/{0, 2, 4, 6}, /*num_workers=*/2,
                           cache_metadata, options)
          .value();

  balancer.AddFeatureLoadingDurationMeasurement(1.0);
  balancer.AddFeatureLoadingDurationMeasurement(0.9);
  balancer.AddFeatureLoadingDurationMeasurement(1.1);
  LOG(INFO) << balancer.Info();
  EXPECT_THAT(balancer.WorkersPerFeatures(),
              ElementsAre(1, -1, 0, -1, 1, -1, 0));  // 2 in each workers.

  // Worker #0 is faster than worker #1.
  LOG(INFO) << "Step 1";
  CHECK_OK(balancer.AddWorkDurationMeasurement({{1, 2}, {5, 2}}).status());
  CHECK_OK(balancer.ApplyPendingOrder());
  LOG(INFO) << balancer.Info();
  EXPECT_THAT(balancer.WorkersPerFeatures(),
              ElementsAre(1, -1, 0, -1, 1, -1, 0));  // 2 in each workers.

  LOG(INFO) << "Step 2";
  CHECK_OK(balancer.AddWorkDurationMeasurement({{1, 2}, {5, 2}}).status());
  CHECK_OK(balancer.ApplyPendingOrder());
  LOG(INFO) << balancer.Info();
  EXPECT_THAT(balancer.WorkersPerFeatures(),
              ElementsAre(1, -1, 0, -1, 1, -1, 0));  // 2 in each workers.

  LOG(INFO) << "Step 3";
  CHECK_OK(balancer.AddWorkDurationMeasurement({{1, 2}, {5, 2}}).status());
  CHECK_OK(balancer.ApplyPendingOrder());
  LOG(INFO) << balancer.Info();
  // Transfer of feature expected.
  EXPECT_THAT(balancer.WorkersPerFeatures(),
              ElementsAre(0, -1, 0, -1, 1, -1, 0));  // 3 in worker 0.

  // Now, worker #1 is faster than worker #0.
  LOG(INFO) << "Step 4";
  CHECK_OK(balancer.AddWorkDurationMeasurement({{8, 3}, {1, 1}}).status());
  CHECK_OK(balancer.ApplyPendingOrder());
  LOG(INFO) << balancer.Info();
  EXPECT_THAT(balancer.WorkersPerFeatures(),
              ElementsAre(0, -1, 0, -1, 1, -1, 0));  // 3 in worker 0.

  LOG(INFO) << "Step 5";
  CHECK_OK(balancer.AddWorkDurationMeasurement({{8, 3}, {1, 1}}).status());
  CHECK_OK(balancer.ApplyPendingOrder());
  LOG(INFO) << balancer.Info();
  EXPECT_THAT(balancer.WorkersPerFeatures(),
              ElementsAre(1, -1, 0, -1, 1, -1, 0));  // 2 in worker 0.

  LOG(INFO) << "Step 6";
  CHECK_OK(balancer.AddWorkDurationMeasurement({{4, 2}, {4, 2}}).status());
  CHECK_OK(balancer.ApplyPendingOrder());
  LOG(INFO) << balancer.Info();
  // Two transfers of feature expected.
  EXPECT_THAT(balancer.WorkersPerFeatures(),
              ElementsAre(1, -1, 0, -1, 1, -1, 0));  // 2 in worker 0.
}

}  // namespace
}  // namespace distributed_decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
