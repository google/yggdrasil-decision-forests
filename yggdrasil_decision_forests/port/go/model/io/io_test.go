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

package io

import (
	"testing"

	_ "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/canonical"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/utils/test"
)

func TestLoadModelBlobSequence(t *testing.T) {
	modelPath := "../../../../test_data/model/adult_binary_class_gbdt"
	model, err := LoadModel(modelPath)
	if err != nil {
		t.Fatalf("Cannot load model. %v", err)
	}

	test.CheckEq(t, model.Name(), "GRADIENT_BOOSTED_TREES", "")
}
