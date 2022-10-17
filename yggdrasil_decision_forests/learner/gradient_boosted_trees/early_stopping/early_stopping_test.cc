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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/early_stopping/early_stopping_snapshot.pb.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace learner {
namespace gradient_boosted_trees {
namespace {

using ::testing::Not;
using ::yggdrasil_decision_forests::test::EqualsProto;

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

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace learner
}  // namespace yggdrasil_decision_forests
