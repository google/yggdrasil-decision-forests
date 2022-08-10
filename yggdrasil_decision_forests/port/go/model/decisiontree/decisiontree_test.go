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

package decisiontree

import (
	"testing"

	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/utils/test"
)

func TestLoadForestBlobSequence(t *testing.T) {
	path := "../../../../test_data/model/adult_binary_class_gbdt/nodes"

	forest, err := LoadForest(path, 1, "BLOB_SEQUENCE", 68)
	if err != nil {
		t.Fatal(err)
	}

	// Validated with :show_model
	test.CheckEq(t, len(forest.Trees), 68, "")
	test.CheckEq(t, forest.Trees[0].Root.RawNode.GetCondition().GetAttribute(), int32(5), "")
	test.CheckEq(t, forest.NumLeafs(), 4352, "")
	test.CheckEq(t, forest.NumNonLeafs(), 4284, "")
}
