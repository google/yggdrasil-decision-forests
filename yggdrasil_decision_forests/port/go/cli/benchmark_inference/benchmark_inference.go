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

/*
Benchmark the inference speed of a model.

Usage example:

	# Disable CPU power scaling
	sudo apt install linux-cpupower
	sudo cpupower frequency-set --governor performance

	# Configure benchmark
	MODEL=$(pwd)/learning/lib/ami/simple_ml/test_data/model/adult_gdt_classifier
	DATASET=csv:$(pwd)/learning/lib/ami/simple_ml/test_data/dataset/adult.csv
	BUILD_FLAGS="-c opt --copt=-mfma --copt=-mavx2"
	RUN_OPTIONS="--model=${MODEL} \
		--dataset=${DATASET} \
		--batch_size=100 \
		--warmup_runs=10 \
		--num_runs=100"

	# Benchmark
	bazel run ${BUILD_FLAGS} \
		//third_party/yggdrasil_decision_forests/port/go/cli:benchmarkinference -- \
		${RUN_OPTIONS}

Naming convention:
  - A (benchmark) "run" evaluates the speed of a model on a dataset.
  - A "run" is composed of one of more "unit runs".
  - A "unit run" measure the speed of a specific inference implementation (called "engine") with
    specific parameters (e.g. batchSize=10).
*/
package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"strings"
	"time"

	"flag"
	model_io "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/io/canonical"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/serving/engine"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/serving/example"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/serving"

	// External dependencies, pls keep in this position in file.
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/utils/file"
	// End of external dependencies.//
	//
)

var flagModel = flag.String("model", "", "Path to the model")
var flagDataset = flag.String("dataset", "", "Type path to the dataset e.g. csv:/tmp/my_file.csv")
var flagNumRuns = flag.Int("num_runs", 20, "Number of times the dataset is run. Higher values increase the precision of the timings, but increase the duration of the benchmark.")
var flagBatchSize = flag.Int("batch_size", 100, "Number of examples per batch. Note that some engine are not impacted by the batch size.")
var flagWarmupRuns = flag.Int("warmup_runs", 2, "Number of runs through the dataset before the benchmark.")

// Options are the options to run the benchmark.
type Options struct {
	// Number of times the entire dataset is run.
	numRuns int

	// Number of runs to "warmup" the engine i.e. running the engine before the benchmark.
	warmupRuns int

	// Number of examples in each batch. Some engine speed can be impacted by the batch size.
	batchSize int
}

// Run runs the benchmark. The results are printed on the standard output.
func Run(modelPath string, datasetPath string, options *Options) error {
	fmt.Printf("Run benchmark with\n  model: %v\n  dataset: %v\n  options: %v\n",
		modelPath, datasetPath, options)

	// Check the validity of the options
	if options.numRuns <= 0 {
		return fmt.Errorf("options.runs should be greater or equal to 1")
	}
	if options.batchSize <= 0 {
		return fmt.Errorf("options.batchSize should be greater or equal to 1")
	}
	if options.warmupRuns <= 0 {
		return fmt.Errorf("options.warmupRuns should be greater or equal to 1")
	}

	// Load the model
	fmt.Println("Load model")
	model, err := model_io.LoadModel(modelPath)
	if err != nil {
		return err
	}
	fmt.Println("\tFound model:", model.Name())

	// Compile the model
	fmt.Println("Compile model")
	engine, err := serving.NewEngine(model)
	if err != nil {
		return err
	}
	fmt.Printf("\tBuilt engine \"%T\" with %d input features\n",
		engine, engine.Features().NumFeatures())

	// Loads the dataset.
	fmt.Println("Load dataset")
	dataset, err := loadDataset(engine, datasetPath)
	if err != nil {
		return err
	}
	fmt.Printf("\t%d examples\n", dataset.NumAllocatedExamples())

	// Run the benchmark
	fmt.Println("Run benchmark")
	// For now, the benchmark is composed of a single evaluation.
	result, err := UnitRun(engine, dataset, options)
	if err != nil {
		return err
	}

	// Print the result
	fmt.Println("Results")
	fmt.Print(result)

	return nil
}

func (result *UnitRunResult) String() string {
	return fmt.Sprintf(
		`Avg. time per dataset:  %v
Avg. time per batch:    %v
Avg. time per examples: %v
`,
		// Note: In Go, duration * duration gives a duration, where the result is effectively
		// numNanoseconds * numNanoseconds -> numNanoseconds.
		result.durationPerExample*time.Duration(result.numExamples),
		result.durationPerExample*time.Duration(result.batchSize),
		result.durationPerExample)
}

func loadDataset(engine engine.Engine, typedPath string) (*example.Batch, error) {
	format, path, err := parseTypedPath(typedPath)
	if err != nil {
		return nil, err
	}
	switch format {
	case "csv":
		return loadDatasetCsv(engine, path)
	default:
		return nil, fmt.Errorf("Non supported dataset format %v", format)
	}
}

// parseTypedPath parses a typed path into its constituents.
//
// For example:
//
//	Input: "csv:/path/to/sharded/csv/file@10"
//	Results:
//	  1. "csv"
//	  2. "/path/to/sharded/csv/file@10"
//	  3. nil (i.e. no error)
func parseTypedPath(typedPath string) (pathType string, path string, err error) {
	i := strings.Index(typedPath, ":")
	if i == -1 {
		err = fmt.Errorf("Malformed typed dataset path. Expecting [format]:[path]. Instead, got %v", typedPath)
		return
	}
	pathType = typedPath[:i]
	path = typedPath[i+1:]
	err = nil
	return
}

func loadDatasetCsv(engine engine.Engine, path string) (*example.Batch, error) {
	// Read the csv content.
	fileHandle, err := file.OpenRead(context.Background(), path)
	if err != nil {
		return nil, err
	}
	fileIO := fileHandle.IO(context.Background())
	defer fileIO.Close()
	csvData, err := csv.NewReader(fileIO).ReadAll()
	if err != nil {
		return nil, err
	}

	// Skip the header file
	csvHeader := csvData[0]
	csvData = csvData[1:]
	numExamples := len(csvData)

	// Convert the csv into a "Batch".
	examples := engine.AllocateExamples(numExamples)
	for exampleIdx := 0; exampleIdx < numExamples; exampleIdx++ {
		if err := examples.SetFromFields(exampleIdx, csvHeader, csvData[exampleIdx]); err != nil {
			return nil, err
		}
	}
	return examples, nil
}

// UnitRunResult contains the benchmark result for a single run.
type UnitRunResult struct {
	durationPerExample time.Duration
	numExamples        int
	batchSize          int
}

// UnitRun benchmark a single engine on a give dataset.
func UnitRun(engine engine.Engine, dataset *example.Batch, options *Options) (*UnitRunResult, error) {

	batchSize := options.batchSize
	numExamples := dataset.NumAllocatedExamples()
	numBatches := (numExamples + batchSize - 1) / options.batchSize

	batch := engine.AllocateExamples(batchSize)
	predictions := engine.AllocatePredictions(batchSize)

	run := func(numRuns int) {
		for runIdx := 0; runIdx < numRuns; runIdx++ {
			for batchIdx := 0; batchIdx < numBatches; batchIdx++ {
				beginIdx := batchIdx * batchSize
				endIdx := (batchIdx + 1) * batchSize
				if endIdx > numExamples {
					endIdx = numExamples
				}
				numExamplesInBatch := endIdx - beginIdx

				// Set the example values.
				// The benchmark time account for a single copy of the feature values.
				batch.CopyFrom(dataset, beginIdx, endIdx)

				// Generate the predictions.
				engine.Predict(batch, numExamplesInBatch, predictions)
			}
		}
	}

	// Warmup
	_ = time.Now()
	run(options.warmupRuns)
	_ = time.Now()

	// Benchmark
	start := time.Now()
	run(options.numRuns)
	end := time.Now()

	result := &UnitRunResult{
		numExamples: numExamples,
		batchSize:   batchSize,
	}
	result.durationPerExample = end.Sub(start) / time.Duration(options.numRuns*numExamples)
	return result, nil
}

func main() {
	flag.Parse()

	options := Options{
		numRuns:    *flagNumRuns,
		batchSize:  *flagBatchSize,
		warmupRuns: *flagWarmupRuns}
	Run(*flagModel, *flagDataset, &options)
}
