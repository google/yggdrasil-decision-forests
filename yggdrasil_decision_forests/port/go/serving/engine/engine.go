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

// Package engine defines the Engine interface.
package engine

import "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/serving/example"

// Engine generates predictions for a model, given an example.
/*
Usage example:

	import (".../ports/go/model/io")

	// Load a model
	model, err := io.Load("/path/to/model")

	// Compile the model for fast inference.
	// At this point, the "model object can be discarded.
	engine, err := NewEngine(model)

	// b/235436797: When running a model trained with the TensorFlow Decision
	// Forests API, use the `NewEngineWithCompatibility` method instead:
	engine, err := serving.NewEngineWithCompatibility(model, example.CompatibilityTensorFlowDecisionForests)

	// Obtain the ID of the model input feature (should be only done once).
	feature_age, age_used_by_model := engine.Features().NumericalFeatures["age"]
	feature_country, country_used_by_model := engine.Features().CategoricalFeatures["country"]

	// Allocate a batch of examples and predictions
	examples := engine.AllocateExamples(10)
	predictions := engine.AllocatePredictions(10)

	// Clear the content of examples.
	// This is only useful if "examples" is re-used multiple times.
	examples.Clear()

	// Set all the example values as missing. This operation is only necessary if
	// not all the feature values will be set.
	examples.FillMissing()

	// Set the feature "age" to be 32 for the first examples.
	examples.SetNumerical(, feature_age, 32.0)

	// Set the feature "age" to be missing for the second examples.
	examples.SetMissingNumerical(, feature_age)

	// Set the feature "country" to be "UK" for the first examples.
	examples.SetCategorical(, feature_country, "UK")

	// Generates the predictions for the first two examples.
	engine.Predict(examples, 2, predictions)

	// Print the predictions.
	fmt.Println("Each prediction is of dimension %v", engine.OutputDim())
	fmt.Println("The predictions are %v and %v",predictions[0], predictions[1])
*/
type Engine interface {

	// Number of dimensions in the predictions. Predictions (allocated by
	// "AllocatePredictions" and populated by "Predict") contains "OutputDim"
	// elements for each example (example major; output dim minor).
	OutputDim() int

	// Populates "predictions" with the predictions of the engine.
	// Is "numExamples" is less than the number of examples allocated in
	// "examples", "Predict" computes the predictions for the first "numExamples"
	// examples is computed.
	Predict(examples *example.Batch, numExamples int, predictions []float32)

	// Allocates a set of examples. A same batch can be re-used multiple time.
	AllocateExamples(maxNumExamples int) *example.Batch

	// Allocates a set of predictions.
	AllocatePredictions(maxNumExamples int) []float32

	// Input features of the model. "Features()" is used to generated the feature
	// ids (e.g. "Features().GetNumericalFeatureID("age")") of the model during
	// the initialization phase.
	Features() *example.Features
}
