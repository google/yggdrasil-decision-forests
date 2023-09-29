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

// Package example defines "Batch": a batch of examples; and "Features": the
// specification of the input features of a model.
package example

import (
	"fmt"
	"strconv"

	dataspec_pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/dataset/proto"
	model_pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/proto"
)

// OutOfVocabulary (OOV) is the special values of unknown or too-rare categorical values.
const OutOfVocabulary = uint32(0)
const OutOfVabulary = uint32(0) // Old typo -- not to break any compatibilities for now.

// NumericalFeatureID is the unique identifier of a numerical feature.
type NumericalFeatureID int

// CategoricalFeatureID is the unique identifier of a categorical feature.
type CategoricalFeatureID int

// Features contains the definition of the input features of a model.
type Features struct {
	// NumericalFeatures is the mapping between numerical feature names and numerical feature ids.
	// Indexed by "NumericalFeatureID".
	NumericalFeatures map[string]NumericalFeatureID
	// CategoricalFeatures is the mapping between categorical feature names and categorical feature
	// ids. Indexed by "CategoricalFeatureID".
	CategoricalFeatures map[string]CategoricalFeatureID

	// MissingNumericalValues is the representation of a "missing value" for each of the numerical
	// features.
	// Note: Currently, serving only support global imputation of missing values
	// during inference.
	MissingNumericalValues []float32
	// MissingCategoricalValues is the representation of a "missing value" for each of the categorical
	// features.NumericalFeatureID
	MissingCategoricalValues []uint32

	// CategoricalSpec is the meta-data about the categorical features. Indexed by
	// "CategoricalFeatureID".
	CategoricalSpec []CategoricalSpec

	// Compatibility indicates how the model is served.
	Compatibility CompatibilityType
}

// CategoricalSpec is the meta-data about a categorical feature.
type CategoricalSpec struct {
	// NumUniqueValues of this feature. The feature value should be in [0, NumUniqueValues).
	NumUniqueValues uint32

	// Dictionary of string values to integer values for this feature.
	// It is set to nil if the feature does not have a Dictionary i.e. the
	// feature is fed with `uint32` values directly.
	Dictionary map[string]uint32
}

// FeatureConstructionMap contains the mapping between the column index and the
// feature id. FeatureConstructionMap is only used during the model to engine
// compilation, and it is then discarded.
type FeatureConstructionMap struct {
	// Mapping between a column index (i.e. the index of the column in the
	// dataspec) and a NumericalFeatureID.
	NumericalFeatures map[int]NumericalFeatureID

	// Mapping between a column index (in the dataspec) and a
	// CategoricalFeatureID.
	CategoricalFeatures map[int]CategoricalFeatureID
}

// NewFeatures converts a dataspec into a feature definition used by an engine.
func NewFeatures(dataspec *dataspec_pb.DataSpecification,
	header *model_pb.AbstractModel, compatibility CompatibilityType) (*Features, *FeatureConstructionMap, error) {

	// Initialize the feature fields.
	features := &Features{Compatibility: compatibility}
	features.NumericalFeatures = map[string]NumericalFeatureID{}
	features.CategoricalFeatures = map[string]CategoricalFeatureID{}
	features.MissingNumericalValues = make([]float32, 0)
	features.MissingCategoricalValues = make([]uint32, 0)
	features.CategoricalSpec = make([]CategoricalSpec, 0)

	buildMap := &FeatureConstructionMap{}
	buildMap.NumericalFeatures = map[int]NumericalFeatureID{}
	buildMap.CategoricalFeatures = map[int]CategoricalFeatureID{}

	// Index the input features.
	for _, columnIdx := range header.GetInputFeatures() {
		column := dataspec.GetColumns()[columnIdx]

		switch column.GetType() {

		case dataspec_pb.ColumnType_NUMERICAL:
			featureID := NumericalFeatureID(len(features.NumericalFeatures))
			buildMap.NumericalFeatures[int(columnIdx)] = featureID
			features.NumericalFeatures[column.GetName()] = featureID
			missingValue := float32(column.GetNumerical().GetMean())
			features.MissingNumericalValues = append(features.MissingNumericalValues, missingValue)

		case dataspec_pb.ColumnType_CATEGORICAL:
			featureID := CategoricalFeatureID(len(features.CategoricalFeatures))
			buildMap.CategoricalFeatures[int(columnIdx)] = featureID
			features.CategoricalFeatures[column.GetName()] = featureID
			features.CategoricalSpec = append(features.CategoricalSpec, CategoricalSpec{})
			missingValue := uint32(column.GetCategorical().GetMostFrequentValue())
			features.MissingCategoricalValues = append(features.MissingCategoricalValues, missingValue)

			spec := &features.CategoricalSpec[len(features.CategoricalSpec)-1]
			spec.NumUniqueValues = uint32(column.GetCategorical().GetNumberOfUniqueValues())

			if !column.GetCategorical().GetIsAlreadyIntegerized() {
				// Copy the Dictionary
				spec.Dictionary = map[string]uint32{}
				for itemKey, itemValue := range column.GetCategorical().GetItems() {
					spec.Dictionary[itemKey] = uint32(itemValue.GetIndex())
				}
			}

		default:
			return nil, nil, fmt.Errorf("Non supported feature %v with type %v",
				column.GetName(), column.GetType())
		}
	}

	if compatibility == CompatibilityAutoTFX {
		features.OverrideMissingValuePlaceholders(-1, "")
	}

	return features, buildMap, nil
}

// NumFeatures is the number of features.
func (f *Features) NumFeatures() int {
	return len(f.NumericalFeatures) + len(f.CategoricalFeatures)
}

// OverrideMissingValuePlaceholders specifies the values that will replace the missing numerical
// and categorical values when calling SetMissing* during inference.
//
// Models are natively able to handle missing values. Overriding the missing values is a form of
// data pre-processing that should only be applied if such pre-processing is also applied during
// training.
//
// This overrides all missing values from all features, both numerical and categorical.
func (f *Features) OverrideMissingValuePlaceholders(numerical float32, categorical string) {
	// Numerical values.
	for i := 0; i < len(f.MissingNumericalValues); i++ {
		f.MissingNumericalValues[i] = numerical
	}

	// Categorical values
	for i := 0; i < len(f.MissingCategoricalValues); i++ {
		spec := &f.CategoricalSpec[i]
		if spec.Dictionary != nil {
			value, exists := spec.Dictionary[categorical]
			if !exists {
				value = OutOfVocabulary
			}
			f.MissingCategoricalValues[i] = value
		}
	}
}

// Batch is a set of examples.
type Batch struct {
	features    *Features
	numExamples int

	// {Example major, feature minor} values for the unary feature values.
	NumericalValues   []float32
	CategoricalValues []uint32
}

// NewBatch creates a batch of examples. The example values are in a
// non-defined state: Because being used, the features values should be set
// ether with "FillMissing" or "Set*".
func NewBatch(numExamples int, features *Features) *Batch {
	batch := &Batch{numExamples: numExamples, features: features}
	batch.NumericalValues = make([]float32, len(features.NumericalFeatures)*numExamples)
	batch.CategoricalValues = make([]uint32, len(features.CategoricalFeatures)*numExamples)
	return batch
}

// NumAllocatedExamples is the number of allocated examples.
func (batch *Batch) NumAllocatedExamples() int {
	return batch.numExamples
}

// Clear clears the content of a batch. After a clear call, the feature values
// are in a non defined state i.e. in the same state as after "NewBatch".
func (batch *Batch) Clear() {
	// Note: Nothing to clear for now.
}

// FillMissing sets all the feature values of all the examples as missing.
//
// This method is equivalent to, but more efficient than, calling the
// "SetMissing*" methods for all the features and all the examples.
func (batch *Batch) FillMissing() {
	for exampleIdx := 0; exampleIdx < batch.numExamples; exampleIdx++ {
		// Numerical features
		beginIdx := exampleIdx * len(batch.features.NumericalFeatures)
		endIdx := (exampleIdx + 1) * len(batch.features.NumericalFeatures)
		copy(batch.NumericalValues[beginIdx:endIdx], batch.features.MissingNumericalValues)

		// Categorical features
		beginIdx = exampleIdx * len(batch.features.CategoricalFeatures)
		endIdx = (exampleIdx + 1) * len(batch.features.CategoricalFeatures)
		copy(batch.CategoricalValues[beginIdx:endIdx], batch.features.MissingCategoricalValues)
	}
}

// SetNumerical sets the value of a numerical feature.
func (batch *Batch) SetNumerical(exampleIdx int, feature NumericalFeatureID, value float32) {
	batch.NumericalValues[int(feature)+exampleIdx*len(batch.features.NumericalFeatures)] = value
}

// SetMissingNumerical sets a numerical feature value as missing.
func (batch *Batch) SetMissingNumerical(exampleIdx int, feature NumericalFeatureID) {
	batch.NumericalValues[int(feature)+exampleIdx*len(batch.features.NumericalFeatures)] =
		batch.features.MissingNumericalValues[feature]
}

// SetCategorical sets the value of a categorical feature as an integer.
func (batch *Batch) SetCategorical(exampleIdx int, feature CategoricalFeatureID, value uint32) {
	if batch.features.Compatibility == CompatibilityTensorFlowDecisionForests ||
		batch.features.Compatibility == CompatibilityAutoTFX {
		value++
	}
	batch.CategoricalValues[int(feature)+exampleIdx*len(batch.features.CategoricalFeatures)] = value
}

// SetCategoricalFromString sets the value of a categorical feature.
func (batch *Batch) SetCategoricalFromString(exampleIdx int, feature CategoricalFeatureID, rawValue string) error {
	spec := &batch.features.CategoricalSpec[feature]
	// TODO: Report the feature name.
	if spec.Dictionary == nil {
		return fmt.Errorf("failed to set Feature %d to %q because feature does not have Dictionary and only accepts numerical values (uint32)", feature, rawValue)
	}
	value, exists := spec.Dictionary[rawValue]
	if !exists {
		value = OutOfVocabulary
	}

	batch.CategoricalValues[int(feature)+exampleIdx*len(batch.features.CategoricalFeatures)] = value
	return nil
}

// SetMissingCategorical sets a categorical feature value as missing.
func (batch *Batch) SetMissingCategorical(exampleIdx int, feature CategoricalFeatureID) {
	batch.CategoricalValues[int(feature)+exampleIdx*len(batch.features.CategoricalFeatures)] =
		batch.features.MissingCategoricalValues[feature]
}

// SetFromFields sets all the fields of an example from a csv-like field and
// header. This method is slow and should not be used for speed-sensitive code.
//
// Empty field and fields with the value "NA" are considered "missing values".
//
// Example:
//
//	examples.SetFromFields(0, ["a","b","c"], ["0.5","UK","NA"])
func (batch *Batch) SetFromFields(exampleIdx int, header []string, values []string) error {

	// Representation of missing values.
	const missingSymbol1 = ""
	const missingSymbol2 = "NA"

	for fieldIdx, key := range header {
		rawValue := values[fieldIdx]
		isMissing := rawValue == missingSymbol1 || rawValue == missingSymbol2

		// Numerical feature
		numericalFeatureID, found := batch.features.NumericalFeatures[key]
		if found {
			if isMissing {
				batch.SetMissingNumerical(exampleIdx, numericalFeatureID)
				continue
			}
			value, err := strconv.ParseFloat(rawValue, 32)
			if err != nil {
				return err
			}
			batch.SetNumerical(exampleIdx, numericalFeatureID, float32(value))
			continue
		}

		// Categorical feature
		categoricalFeatureID, found := batch.features.CategoricalFeatures[key]
		if found {
			if isMissing {
				batch.SetMissingCategorical(exampleIdx, categoricalFeatureID)
				continue
			}
			spec := &batch.features.CategoricalSpec[categoricalFeatureID]
			if spec.Dictionary == nil {
				// The feature value is an integer represented as an ASCII string.
				value, err := strconv.ParseInt(rawValue, 10, 32)
				if err != nil {
					return err
				}
				batch.SetCategorical(exampleIdx, categoricalFeatureID, uint32(value))
			} else {
				err := batch.SetCategoricalFromString(exampleIdx, categoricalFeatureID, rawValue)
				if err != nil {
					return err
				}
			}
			continue
		}
		// This column is not used by the model. We ignore it.
	}
	return nil
}

// CopyFrom copies the content of a batch from another batch.
// Assumes both source batch has the exact same features (e.g. it is created by the same engine).
func (batch *Batch) CopyFrom(src *Batch, beginIdx int, endIdx int) {
	batch.Clear()

	copy(
		batch.NumericalValues[:(endIdx-beginIdx)*len(batch.features.NumericalFeatures)],
		src.NumericalValues[(beginIdx*len(batch.features.NumericalFeatures)):(endIdx*len(batch.features.NumericalFeatures))])

	copy(
		batch.CategoricalValues[:(endIdx-beginIdx)*len(batch.features.CategoricalFeatures)],
		src.CategoricalValues[(beginIdx*len(batch.features.CategoricalFeatures)):(endIdx*len(batch.features.CategoricalFeatures))])
}

// ToStringDebug exports the content of the set of examples into a text-debug representation.
func (batch *Batch) ToStringDebug() string {

	repr := fmt.Sprintf("batch with %v example(s)\n", batch.NumAllocatedExamples())

	for exampleIdx := 0; exampleIdx < batch.NumAllocatedExamples(); exampleIdx++ {
		repr += fmt.Sprintf("exampleIdx: %v\n", exampleIdx)

		// Numerical features.
		for name, featureID := range batch.features.NumericalFeatures {
			value := batch.NumericalValues[int(featureID)+exampleIdx*len(batch.features.NumericalFeatures)]
			repr += fmt.Sprintf("\"%v\" (NUMERICAL id:%v): \"%v\"\n", name, featureID, value)
		}

		// Categorical features.
		for name, featureID := range batch.features.CategoricalFeatures {
			value := batch.CategoricalValues[int(featureID)+exampleIdx*len(batch.features.CategoricalFeatures)]
			spec := &batch.features.CategoricalSpec[featureID]
			if spec.Dictionary == nil {
				repr += fmt.Sprintf("\"%v\" (CATEGORICAL INTEGER id:%v): \"%v\"\n", name, featureID, value)
			} else {
				strValue := "<UNKNOWN>"
				for itemKey, itemIdx := range spec.Dictionary {
					if itemIdx == value {
						strValue = itemKey
						break
					}
				}
				repr += fmt.Sprintf("\"%v\" (CATEGORICAL STRING id:%v): \"%v\"\n", name, featureID, strValue)
			}
		}
	}
	return repr
}

// GetColumn gets the column spec from its name.
func GetColumn(name string, dataspec *dataspec_pb.DataSpecification) *dataspec_pb.Column {
	for _, column := range dataspec.GetColumns() {
		if column.GetName() == name {
			return column
		}
	}
	return nil
}

// CompatibilityType indicates how the model was trained, and it affects how features are consumed.
type CompatibilityType int32

const (
	// CompatibilityYggdrasil is the native way to consume examples and models model with Yggdrasil
	// Decision Forests.
	CompatibilityYggdrasil CompatibilityType = 0

	// CompatibilityTensorFlowDecisionForests consumes models trained with TensorFlow Decision
	// Forests.
	//
	// Compatibility impact: Categorical and categorical-set columns feed as integer are offset by
	// 1. See "CATEGORICAL_INTEGER_OFFSET" in TensorFlow Decision Forests.
	CompatibilityTensorFlowDecisionForests CompatibilityType = 1

	// CompatibilityAutoTFX consumes models trained with TensorFlow Decision
	// Forests.
	//
	// Compatibility impact: Categorical and categorical-set columns feed as integer are offset by
	// 1. See "CATEGORICAL_INTEGER_OFFSET" in TensorFlow Decision Forests. Missing numerical and
	// categorical string values are replaced respectively by -1 and "" (empty string).
	CompatibilityAutoTFX CompatibilityType = 2

	// CompatibilityAutomatic detects automatically the compatibility of the model.
	CompatibilityAutomatic = 3
)
