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

package serving

// Check the predictions of the Go engines with golden predictions.
// See "learning/lib/ami/simple_ml/test_data/prediction/README" for the generation of golden
// predictions.

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"

	model_pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/proto"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/io"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/serving/engine"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/serving/example"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/utils/test"
)

// readCsvFile returns the fields of a csv file.
func readCsvFile(path string) ([][]string, error) {
	fileHandle, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { fileHandle.Close() }()
	return csv.NewReader(fileHandle).ReadAll()
}

// Parse a 2d array of string into a 2d array to float32.
func parseFloat32(src [][]string) ([][]float32, error) {
	dst := make([][]float32, len(src))
	for rowIdx, srcRow := range src {
		dstRow := make([]float32, len(srcRow))
		dst[rowIdx] = dstRow
		for colIdx, srcValue := range srcRow {
			dstValue, err := strconv.ParseFloat(srcValue, 32)
			if err != nil {
				return nil, err
			}
			dstRow[colIdx] = float32(dstValue)
		}
	}
	return dst, nil
}

// testEngine tests the predictions of an engine against golden predictions.
func testEngine(t *testing.T, model model.Model, engine engine.Engine, datasetPath string, goldenPredictionPath string,
	doCheckPredictions bool) {

	dataset, err := readCsvFile(datasetPath)
	if err != nil {
		t.Fatal(err)
	}

	goldenPredictions, err := readCsvFile(goldenPredictionPath)
	if err != nil {
		t.Fatal(err)
	}

	if len(dataset) != len(goldenPredictions) {
		t.Fatalf("Non matching dataset and golden predictions")
	}

	datasetHeader := dataset[0]
	dataset = dataset[1:]
	// We assume the prediction are generated in the order of Yggdrasil Decision Forests.
	goldenPredictionsHeader := goldenPredictions[0]
	goldenPredictionsFloat32, err := parseFloat32(goldenPredictions[1:])
	if err != nil {
		t.Fatal(err)
	}

	const batchSize = 32
	numExamples := len(dataset)
	numBatches := (numExamples + batchSize - 1) / batchSize

	examples := engine.AllocateExamples(batchSize)
	predictions := engine.AllocatePredictions(batchSize)

	for batchIdx := 0; batchIdx < numBatches; batchIdx++ {
		beginIdx := batchIdx * batchSize
		endIdx := minInt((batchIdx+1)*batchSize, numExamples)
		numExamplesInBatch := endIdx - beginIdx

		// Set the example values.
		examples.Clear()
		for exampleIdx := beginIdx; exampleIdx < endIdx; exampleIdx++ {
			err := examples.SetFromFields(exampleIdx-beginIdx, datasetHeader, dataset[exampleIdx])
			if err != nil {
				t.Fatal(err)
			}
		}

		// Print the examples
		if batchIdx == 0 {
			//fmt.Println(examples.ToStringDebug())
		}

		// Generate the predictions.
		engine.Predict(examples, numExamplesInBatch, predictions)

		// Check the prediction values
		if doCheckPredictions {
			checkPredictions(t, model, goldenPredictionsHeader, goldenPredictionsFloat32,
				beginIdx, endIdx, predictions)
		}
	}
}

func checkPredictions(t *testing.T,
	model model.Model,
	goldenPredictionsHeader []string,
	goldenPredictionsFloat32 [][]float32,
	beginIdx int, endIdx int, predictions []float32) {

	switch model.Header().GetTask() {

	case model_pb.Task_CLASSIFICATION:
		numClasses := model.Dataspec().GetColumns()[model.Header().
			GetLabelColIdx()].GetCategorical().GetNumberOfUniqueValues()
		if numClasses == 3 {
			// Binary classification.
			test.CheckEq(t, len(goldenPredictionsHeader), 2, "Unexpected gold prediction shape")
			for exampleIdx := beginIdx; exampleIdx < endIdx; exampleIdx++ {
				exampleIdxInBatch := exampleIdx - beginIdx
				test.CheckNearFloat32(t, predictions[exampleIdxInBatch],
					goldenPredictionsFloat32[exampleIdx][1], 0.0001,
					fmt.Sprintf("non matching predictions for example %v", exampleIdx))
			}
		} else {
			// Multi-class classification.
			t.Fatal("Multi class classification not supported")
		}

	case model_pb.Task_REGRESSION:
	case model_pb.Task_RANKING:
		test.CheckEq(t, len(goldenPredictionsHeader), 1, "Unexpected gold prediction shape")
		for exampleIdx := beginIdx; exampleIdx < endIdx; exampleIdx++ {
			exampleIdxInBatch := exampleIdx - beginIdx
			test.CheckNearFloat32(t, predictions[exampleIdxInBatch],
				goldenPredictionsFloat32[exampleIdx][0], 0.0001,
				fmt.Sprintf("non matching predictions for example %v", exampleIdx))
		}

	default:
		t.Fatal("Task not supported")
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func canonicalPath(p string) string {
	p = strings.TrimPrefix(p, "yggdrasil_decision_forests/")
	return "../../../" + p
}

// testEngine tests the predictions of a model against golden predictions.
func testModel(t *testing.T, modelPath string, datasetPath string, goldenPredictionPath string, altMissingValue bool) engine.Engine {
	modelPath = canonicalPath(modelPath)
	datasetPath = canonicalPath(datasetPath)
	goldenPredictionPath = canonicalPath(goldenPredictionPath)
	model, err := io.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("Cannot load model. %v", err)
	}
	engine, err := NewEngine(model)
	if err != nil {
		t.Fatalf("Cannot create engine. %v", err)
	}

	if altMissingValue {
		// Change the representation of the missing values in the engine.
		// In this case, the predictions should not be compared to the golden predictions.
		engine.Features().OverrideMissingValuePlaceholders(-1, "")
	}

	testEngine(t, model, engine, datasetPath, goldenPredictionPath, !altMissingValue)
	return engine
}

func TestAdultBinaryClassGBTYdfFormat(t *testing.T) {
	testModel(t,
		"yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt",
		"yggdrasil_decision_forests/test_data/dataset/adult_test.csv",
		"yggdrasil_decision_forests/test_data/prediction/adult_test_binary_class_gbdt.csv",
		false)
}

func TestAbaloneRegressionGBTYdfFormat(t *testing.T) {
	testModel(t,
		"yggdrasil_decision_forests/test_data/model/abalone_regression_gbdt",
		"yggdrasil_decision_forests/test_data/dataset/abalone.csv",
		"yggdrasil_decision_forests/test_data/prediction/abalone_regression_gbdt.csv",
		false)
}

func TestSyntheticRankingGBTYdfFormat(t *testing.T) {
	testModel(t,
		"yggdrasil_decision_forests/test_data/model/synthetic_ranking_gbdt",
		"yggdrasil_decision_forests/test_data/dataset/synthetic_ranking_test.csv",
		"yggdrasil_decision_forests/test_data/prediction/synthetic_ranking_gbdt_test.csv",
		false)
}

func TestAdultBinaryClassRFYdfFormat(t *testing.T) {
	testModel(t,
		"yggdrasil_decision_forests/test_data/model/adult_binary_class_rf",
		"yggdrasil_decision_forests/test_data/dataset/adult_test.csv",
		"yggdrasil_decision_forests/test_data/prediction/adult_test_binary_class_rf.csv",
		false)
}

func TestAdultBinaryClassObliqueRFYdfFormat(t *testing.T) {
	testModel(t,
		"yggdrasil_decision_forests/test_data/model/adult_binary_class_oblique_rf",
		"yggdrasil_decision_forests/test_data/dataset/adult_test.csv",
		"yggdrasil_decision_forests/test_data/prediction/adult_test_binary_class_oblique_rf.csv",
		false)
}

func TestAbaloneRegressionClassClassRFYdfFormat(t *testing.T) {
	testModel(t,
		"yggdrasil_decision_forests/test_data/model/abalone_regression_rf",
		"yggdrasil_decision_forests/test_data/dataset/abalone.csv",
		"yggdrasil_decision_forests/test_data/prediction/abalone_regression_rf.csv",
		false)
}

func TestReadCsvFile(t *testing.T) {
	p := canonicalPath("yggdrasil_decision_forests/test_data/dataset/toy.csv")
	data, err := readCsvFile(p)
	expectedFirstTwoLines := [][]string{
		[]string{"Num_1", "Num_2", "Cat_1", "Cat_2", "Cat_set_1", "Cat_set_2", "Bool_1", "Bool_2", "Cat_3"},
		[]string{"1", "NA", "A", "A", "X", "", "0", "0", "1"}}
	if err != nil {
		t.Fatal(err)
	}
	test.CheckEq(t, len(data), 5, "Wrong shape")
	data = data[:2]
	test.CheckEq(t, data, expectedFirstTwoLines, "")
}

func TestParseFloat32(t *testing.T) {
	strValues := [][]string{[]string{"1", "2"}, []string{"3", "4"}}
	expectedFloatValues := [][]float32{[]float32{1, 2}, []float32{3, 4}}
	floatValues, err := parseFloat32(strValues)
	if err != nil {
		t.Fatal(err)
	}
	test.CheckEq(t, floatValues, expectedFloatValues, "")
}

func TestCompatibilityCategoricalInteger(t *testing.T) {
	tests := []struct {
		compatibility         example.CompatibilityType
		inputValue            uint32
		expectedInternalValue uint32
	}{
		{example.CompatibilityYggdrasil, 0, 0},
		{example.CompatibilityYggdrasil, 1, 1},
		{example.CompatibilityTensorFlowDecisionForests, 0, 1},
		{example.CompatibilityTensorFlowDecisionForests, 1, 2},
	}

	for _, currentTest := range tests {
		featureID := example.CategoricalFeatureID(0)
		features := &example.Features{
			CategoricalFeatures: map[string]example.CategoricalFeatureID{
				"a": featureID},
			CategoricalSpec: []example.CategoricalSpec{example.CategoricalSpec{NumUniqueValues: 3}},
			Compatibility:   currentTest.compatibility,
		}
		batch := example.NewBatch(1, features)
		batch.SetCategorical(0, featureID, currentTest.inputValue)
		test.CheckEq(t, batch.CategoricalValues[0], currentTest.expectedInternalValue, "")
	}
}
