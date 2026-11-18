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

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/early_stopping/early_stopping.h"

#include <limits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/early_stopping/early_stopping_snapshot.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace learner {
namespace gradient_boosted_trees {
namespace {

using ::testing::ElementsAre;
using ::testing::Not;
using ::yggdrasil_decision_forests::test::EqualsProto;
using ::yggdrasil_decision_forests::test::StatusIs;

TEST(EarlyStopping, Interrupion) {
  EarlyStopping manager(/*early_stopping_num_trees_look_ahead=*/2,
                        /*initial_iteration=*/0);
  int iter_idx = 0;
  manager.set_trees_per_iterations(1);
  EXPECT_FALSE(manager.ShouldStop(iter_idx));

  ++iter_idx;
  CHECK_OK(manager.Update(/*validation_loss=*/10,
                          /*validation_secondary_metrics=*/{},
                          /*num_trees=*/0, /*current_iter_idx=*/iter_idx));
  EXPECT_FALSE(manager.ShouldStop(iter_idx));

  ++iter_idx;
  CHECK_OK(manager.Update(9, {}, 1, iter_idx));
  EXPECT_FALSE(manager.ShouldStop(iter_idx));

  ++iter_idx;
  CHECK_OK(manager.Update(8, {}, 2, iter_idx));
  EXPECT_FALSE(manager.ShouldStop(iter_idx));

  ++iter_idx;
  CHECK_OK(manager.Update(7, {}, 3, iter_idx));
  EXPECT_FALSE(manager.ShouldStop(iter_idx));

  ++iter_idx;
  CHECK_OK(manager.Update(8, {}, 4, iter_idx));
  EXPECT_FALSE(manager.ShouldStop(iter_idx));

  ++iter_idx;
  // This is the lowest (i.e. best) loss.
  CHECK_OK(manager.Update(6, {}, 5, iter_idx));
  EXPECT_FALSE(manager.ShouldStop(iter_idx));

  ++iter_idx;
  CHECK_OK(manager.Update(7, {}, 6, iter_idx));
  EXPECT_FALSE(manager.ShouldStop(iter_idx));

  ++iter_idx;
  CHECK_OK(manager.Update(8, {}, 7, iter_idx));
  EXPECT_TRUE(manager.ShouldStop(iter_idx));

  EXPECT_EQ(manager.best_num_trees(), 5);
  EXPECT_EQ(manager.best_loss(), 6);
}

TEST(EarlyStopping, Serialize) {
  EarlyStopping a(/*early_stopping_num_trees_look_ahead=*/2,
                  /*initial_iteration=*/0);
  EarlyStopping b(2, 1);
  a.set_trees_per_iterations(1);

  // Make some updates.
  CHECK_OK(a.Update(/*validation_loss=*/10,
                    /*validation_secondary_metrics=*/{},
                    /*num_trees=*/0,
                    /*iter_idx=*/0));
  CHECK_OK(a.Update(9, {}, 1, 1));

  // Check the internal representation of "a".
  const proto::EarlyStoppingSnapshot expected = PARSE_TEST_PROTO(
      R"pb(
        best_loss: 9
        last_loss: 9
        best_num_trees: 1
        last_num_trees: 1
        num_trees_look_ahead: 2
        trees_per_iterations: 1
        initial_iteration: 0
      )pb");
  EXPECT_THAT(a.Save(), EqualsProto(expected));

  // At this point "a" and "b" should be different.
  EXPECT_THAT(a.Save(), Not(EqualsProto(b.Save())));
  // Synchronize "a" and "b".
  EXPECT_OK(b.Load(a.Save()));

  // At this point "a" and "b" should be equal.
  EXPECT_THAT(a.Save(), EqualsProto(b.Save()));

  // Makes the same updates to "a" and "b".
  CHECK_OK(a.Update(8, {}, 2, 2));
  CHECK_OK(a.Update(7, {}, 3, 3));

  CHECK_OK(b.Update(8, {}, 2, 2));
  CHECK_OK(b.Update(7, {}, 3, 3));

  // At this point "a" and "b" should still be equal.
  EXPECT_THAT(a.Save(), EqualsProto(b.Save()));
}

TEST(EarlyStopping, TrackedMetricMinimization) {
  EarlyStopping manager(/*early_stopping_num_trees_look_ahead=*/2,
                        /*initial_iteration=*/1);
  manager.set_trees_per_iterations(1);
  manager.set_best_metric(/*best_metric_index=*/1,
                          /*best_metric_higher_is_better=*/false);

  // Ignored before initial_iteration.
  EXPECT_OK(manager.Update(/*validation_loss=*/10.f,
                           /*validation_secondary_metrics=*/{0.5f, 1.0f},
                           /*num_trees=*/1, /*current_iter_idx=*/0));
  EXPECT_EQ(manager.best_num_trees(), -1);
  EXPECT_FALSE(manager.ShouldStop(0));

  // Primary validation_loss keeps decreasing while the tracked metric reaches
  // its minimum (3.0f) at num_trees=3 and then increases.
  EXPECT_OK(manager.Update(9.f, {0.6f, 4.0f}, 2, 1));
  EXPECT_FALSE(manager.ShouldStop(1));

  EXPECT_OK(manager.Update(8.f, {0.7f, 3.0f}, 3, 2));
  EXPECT_FALSE(manager.ShouldStop(2));

  EXPECT_OK(manager.Update(7.f, {0.8f, 3.5f}, 4, 3));
  EXPECT_FALSE(manager.ShouldStop(3));

  EXPECT_OK(manager.Update(6.f, {0.9f, 3.8f}, 5, 4));
  EXPECT_TRUE(manager.ShouldStop(4));

  EXPECT_EQ(manager.best_num_trees(), 3);
  EXPECT_EQ(manager.best_loss(), 8.f);
  EXPECT_THAT(manager.best_metrics(), ElementsAre(0.7f, 3.0f));
}

TEST(EarlyStopping, TrackedMetricMaximization) {
  EarlyStopping manager(/*early_stopping_num_trees_look_ahead=*/2,
                        /*initial_iteration=*/0);
  manager.set_trees_per_iterations(1);
  manager.set_best_metric(/*best_metric_index=*/0,
                          /*best_metric_higher_is_better=*/true);

  // Primary validation_loss keeps decreasing while the tracked metric peaks at
  // 0.85f on num_trees=2 and ties on num_trees=4 without strictly improving.
  EXPECT_OK(manager.Update(10.f, {0.70f}, 1, 0));
  EXPECT_FALSE(manager.ShouldStop(0));

  EXPECT_OK(manager.Update(9.f, {0.85f}, 2, 1));
  EXPECT_FALSE(manager.ShouldStop(1));

  EXPECT_OK(manager.Update(8.f, {0.80f}, 3, 2));
  EXPECT_FALSE(manager.ShouldStop(2));

  EXPECT_OK(manager.Update(7.f, {0.85f}, 4, 3));
  EXPECT_TRUE(manager.ShouldStop(3));

  EXPECT_EQ(manager.best_num_trees(), 2);
  EXPECT_EQ(manager.best_loss(), 9.f);
}

TEST(EarlyStopping, FailsOnInvalidTrackedMetricIndex) {
  EarlyStopping negative_index(/*early_stopping_num_trees_look_ahead=*/2,
                               /*initial_iteration=*/0);
  negative_index.set_trees_per_iterations(1);
  negative_index.set_best_metric(/*best_metric_index=*/-1,
                                 /*best_metric_higher_is_better=*/false);
  EXPECT_THAT(negative_index.Update(1.f, {0.5f}, 1, 0),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EarlyStopping out_of_bounds(/*early_stopping_num_trees_look_ahead=*/2,
                              /*initial_iteration=*/1);
  out_of_bounds.set_trees_per_iterations(1);
  out_of_bounds.set_best_metric(/*best_metric_index=*/2,
                                /*best_metric_higher_is_better=*/true);
  EXPECT_OK(out_of_bounds.Update(1.f, {0.5f, 0.6f}, 1, 0));
  EXPECT_THAT(out_of_bounds.Update(1.f, {0.5f, 0.6f}, 2, 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EarlyStopping, SerializeTrackedMetric) {
  EarlyStopping a(/*early_stopping_num_trees_look_ahead=*/2,
                  /*initial_iteration=*/0);
  a.set_trees_per_iterations(1);
  a.set_best_metric(/*best_metric_index=*/1,
                    /*best_metric_higher_is_better=*/true);
  EXPECT_OK(a.Update(10.f, {0.8f, 0.6f}, 1, 0));
  EXPECT_OK(a.Update(9.f, {0.85f, 0.75f}, 2, 1));

  const proto::EarlyStoppingSnapshot expected = PARSE_TEST_PROTO(
      R"pb(
        best_loss: 9
        last_loss: 9
        best_metrics: 0.85
        best_metrics: 0.75
        last_metrics: 0.85
        last_metrics: 0.75
        best_num_trees: 2
        last_num_trees: 2
        num_trees_look_ahead: 2
        trees_per_iterations: 1
        initial_iteration: 0
        best_metric_index: 1
        best_metric_higher_is_better: true
      )pb");
  EXPECT_THAT(a.Save(), EqualsProto(expected));

  EarlyStopping b(5, 3);
  EXPECT_OK(b.Load(a.Save()));
  EXPECT_EQ(b.best_metric_index(), 1);
  EXPECT_TRUE(b.best_metric_higher_is_better());
  EXPECT_THAT(b.Save(), EqualsProto(a.Save()));

  // Loading a snapshot without best_metric_index resets both fields.
  EarlyStopping primary_only(2, 0);
  primary_only.set_trees_per_iterations(1);
  EXPECT_OK(primary_only.Update(5.f, {}, 1, 0));
  EXPECT_OK(b.Load(primary_only.Save()));
  EXPECT_FALSE(b.best_metric_index().has_value());
  EXPECT_FALSE(b.best_metric_higher_is_better());

  // Loading a snapshot whose best_metric_index exceeds best_metrics fails.
  proto::EarlyStoppingSnapshot invalid_snapshot = a.Save();
  invalid_snapshot.set_best_metric_index(5);
  EXPECT_THAT(b.Load(invalid_snapshot),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EarlyStopping, TrackedMetricIgnoresNaN) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  EarlyStopping manager(/*early_stopping_num_trees_look_ahead=*/2,
                        /*initial_iteration=*/0);
  manager.set_trees_per_iterations(1);
  manager.set_best_metric(/*best_metric_index=*/0,
                          /*best_metric_higher_is_better=*/true);

  // Initial NaN metrics do not lock in the first iteration as best.
  EXPECT_OK(manager.Update(10.f, {kNaN}, 1, 0));
  EXPECT_EQ(manager.best_num_trees(), -1);
  EXPECT_FALSE(manager.ShouldStop(0));

  EXPECT_OK(manager.Update(9.f, {kNaN}, 2, 1));
  EXPECT_EQ(manager.best_num_trees(), -1);
  EXPECT_FALSE(manager.ShouldStop(1));

  // First finite metric becomes the best, and later NaN does not overwrite it.
  EXPECT_OK(manager.Update(8.f, {0.75f}, 3, 2));
  EXPECT_EQ(manager.best_num_trees(), 3);
  EXPECT_FALSE(manager.ShouldStop(2));

  EXPECT_OK(manager.Update(7.f, {0.85f}, 4, 3));
  EXPECT_EQ(manager.best_num_trees(), 4);
  EXPECT_FALSE(manager.ShouldStop(3));

  EXPECT_OK(manager.Update(6.f, {kNaN}, 5, 4));
  EXPECT_EQ(manager.best_num_trees(), 4);
  EXPECT_FALSE(manager.ShouldStop(4));

  EXPECT_OK(manager.Update(5.f, {0.80f}, 6, 5));
  EXPECT_TRUE(manager.ShouldStop(5));
  EXPECT_EQ(manager.best_num_trees(), 4);
  EXPECT_THAT(manager.best_metrics(), ElementsAre(0.85f));
}

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace learner
}  // namespace yggdrasil_decision_forests
