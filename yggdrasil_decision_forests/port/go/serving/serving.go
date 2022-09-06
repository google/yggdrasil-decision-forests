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

// Package serving is the entry point for model inference (serving). Models can have more than
// one inference engine, this will pick the correct (fastest) one according to the model.
package serving

import (
	"fmt"

	dataspec_pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/dataset/proto"
	model_pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/proto"
	gbt "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/gradientboostedtrees"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model"
	rf "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/randomforest"
	df_engine "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/serving/decisionforest"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/serving/engine"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/serving/example"
)

// NewEngine creates the best available engine for the model. It fails if no engine is available
// for the model.
//
func NewEngine(model model.Model) (engine.Engine, error) {
	return NewEngineWithCompatibility(model, example.CompatibilityAutomatic)
}

// NewEngineWithCompatibility creates the best available engine for the model. It fails if no
// engine is available for the model.
//
// The "compatibility" argument facilitates cross API modeling. For example, to use a model trained
// with TensorFlow Decision Forests, the compatibility can be set to
// compatibilityTensorFlowDecisionForests. Not setting a compatibility corresponding to how the
// model was trained does not prevent the model to be used. However, in this case, the user need to
// make sure to understand the difference between the APIs (if any) and adapt the code accordingly.
// See the definition of the compatibilities (e.g. "compatibilityTensorFlowDecisionForests") for a
// description of the effects.
func NewEngineWithCompatibility(model model.Model, compatibility example.CompatibilityType) (engine.Engine, error) {

	if compatibility == example.CompatibilityAutomatic {
		compatibility = DetectCompatibility(model)
	}

	if gbtModel, match := model.(*gbt.Model); match {
		return newEngineGbt(gbtModel, compatibility)
	}

	if rfModel, match := model.(*rf.Model); match {
		return newEngineRf(rfModel, compatibility)
	}

	return nil, fmt.Errorf("No engine compatible to the model")
}

func newEngineGbt(model *gbt.Model, compatibility example.CompatibilityType) (engine.Engine, error) {
	switch model.Header().GetTask() {
	case model_pb.Task_CLASSIFICATION:
		numClasses := model.Dataspec().GetColumns()[model.Header().GetLabelColIdx()].GetCategorical().GetNumberOfUniqueValues()
		if numClasses == 3 {
			return df_engine.NewBinaryClassificationGBDTGenericEngine(model, compatibility)
		}
	case model_pb.Task_REGRESSION:
		return df_engine.NewRegressionGBDTGenericEngine(model, compatibility)
	case model_pb.Task_RANKING:
		return df_engine.NewRankingGBDTGenericEngine(model, compatibility)
	}
	return nil, fmt.Errorf("No engine compatible to the model")
}

func newEngineRf(model *rf.Model, compatibility example.CompatibilityType) (engine.Engine, error) {
	switch model.Header().GetTask() {
	case model_pb.Task_CLASSIFICATION:
		numClasses := model.Dataspec().GetColumns()[model.Header().GetLabelColIdx()].GetCategorical().GetNumberOfUniqueValues()
		if numClasses == 3 {
			return df_engine.NewBinaryClassificationRFGenericEngine(model, compatibility)
		}
	case model_pb.Task_REGRESSION:
		return df_engine.NewRegressionRFGenericEngine(model, compatibility)
	}
	return nil, fmt.Errorf("No engine compatible to the model")
}

// Different frameworks on which the model may have been trained.
const (
	TFEstimatorFramework = "TF Estimator" // Generally used in TF 1.0
	TFKerasFramework     = "TF Keras"     // Generally used in TF 2.0
)

// DetectCompatibility detects the most likely compatibility of the model.
// If unsure, returns the Yggdrasil (i.e. native) compatibility.
func DetectCompatibility(model model.Model) example.CompatibilityType {
	framework := model.Header().GetMetadata().GetFramework()

	if framework == TFEstimatorFramework || framework == TFKerasFramework {
		// Test if all the categorical feature feed as integer are marked
		// as "offset_value_by_one_during_training".
		allSet := true
		for _, column := range model.Dataspec().GetColumns() {
			if column.GetType() == dataspec_pb.ColumnType_CATEGORICAL {
				if column.GetCategorical().GetIsAlreadyIntegerized() {
					allSet = allSet && column.GetCategorical().GetOffsetValueByOneDuringTraining()
				}
			}
		}
		if allSet {
			return example.CompatibilityTensorFlowDecisionForests
		}
	}

	// Native YDF model.
	return example.CompatibilityYggdrasil
}
