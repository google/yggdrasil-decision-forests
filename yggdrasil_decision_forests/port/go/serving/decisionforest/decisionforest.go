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

// Package decisionforest contains the engine inference code for decision forest
// models.
package decisionforest

import (
	"fmt"
	"math"

	dt "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/decisiontree"
	gbt "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/gradientboostedtrees"
	rf "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/randomforest"
	"github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/serving/example"

	dataspec_pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/dataset/proto"
	model_pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/proto"
	gbt_pb "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/gradientboostedtrees/proto"
)

// SetNodeSignature is the signature of a method that set a leaf value during
// engine compilation.
type SetNodeSignature func(srcNode *dt.Node, dstNode *genericNode) error

// ActivationSignature is an activation function. Activation functions are
// applied on the output of Gradient Boosted Trees models.
type ActivationSignature func(value float32) float32

// Type of a condition.
type genericConditionType uint8

const (
	// When the node is a leaf i.e. does not contain a condition.
	leafConditionType genericConditionType = 0

	// Condition: feature >= threshold.
	numericalIsHigherConditionType genericConditionType = 1

	// Condition: feature \in mask with mask a 32 bits bitmap.
	categoricalContainsMaskConditionType genericConditionType = 2

	// Condition: feature \in mask with mask containing any number of bits.
	categoricalContainsBufferConditionType genericConditionType = 3

	// Condition: features \dotproduct \weights >= threshold
	numericalObliqueProjectionIsHigherType genericConditionType = 4
)

// genericNode is the most generic decision tree node. It is expected that
// models build with genericNode have the higher coverage of features while
// being the least efficient.
type genericNode struct {

	// Offset to the positive child node.
	//
	// Note: The negative node is always next to its parent node. In other words,
	// the offset of the negative child node is always 1.
	rightIdx uint16

	// Index of the feature being tested. The exact interpretation of "featureIdx"
	// depends on "condition".
	//
	// Special case: In the case of oblique conditions, the feature indices are
	// defined in the oblique feature buffer, and "featureIdx" (this field) contains
	// the *number of features** tested in the condition.
	featureIdx uint16

	// The main parameter of the condition:
	//   numericalIsHigher*: Numerical condition as "value >=
	//     interpret_cast_float32(condition)".
	//
	//   categoricalContainsMask*: Categorical condition as "condition[value]",
	//     where "condition" is a bitmap. Only when the maximum value of the
	//     attribute is < 32.
	//
	//   categoricalContainsBuffer*: Categorical condition as
	//     "categoricalBitmap[condition * 8 + value]", where "categoricalBitmap"
	//     is a bitmap.
	//
	//   numericalObliqueProjectionIsHigher*: Index of the oblique weight in
	//     "oblique{Weights,Features}".
	//
	//     The oblique conditions is:
	//       condition := \sum_{i \in [0, featureIdx)} obliqueWeights[condition + i] *
	//                    feature[obliqueFeatures[condition + i]] >=
	//                    obliqueWeights[condition + featureIdx]
	//
	//     Note: "featureIdx" is the number of conditions. See comment above.
	condition uint32

	// Type of the condition. See the definition of "genericConditionType" for the
	// supported condition types.
	conditionType genericConditionType // 8bits
}

// genericEngine for all types of decision forest models.
type genericEngine struct {
	// features used as input.
	features *example.Features

	// The list of nodes, tree by tree, in a depth first (node, negative,
	// positive) order.
	nodes []genericNode

	// Index in "nodes" of the root nodes.
	rootOffsets []uint32

	// Bitmap used in categorical conditions. Used with the following conditions:
	// [categoricalContainsBufferConditionType].
	categoricalBitmap []byte

	// List of weights used in the oblique conditions.
	obliqueWeights []float32

	// List of feature indices used in the oblique conditions.
	obliqueFeatures []uint16
}

// Initialize the content of a generic engine.
// Note: Generic engines are constructed by value (!= by pointers).
func (e *genericEngine) initialize(forest *dt.Forest,
	header *model_pb.AbstractModel,
	dataspec *dataspec_pb.DataSpecification, setLeaf SetNodeSignature,
	compatibility example.CompatibilityType) error {

	// Create the input features.
	features, buildMap, err := example.NewFeatures(dataspec, header, compatibility)
	if err != nil {
		return err
	}
	e.features = features

	// Create the nodes
	// TODO: Allocate array with the number of nodes when available.
	e.nodes = make([]genericNode, 0)
	e.rootOffsets = make([]uint32, 0, len(forest.Trees))
	for _, tree := range forest.Trees {
		if len(e.nodes) > math.MaxUint32 {
			return fmt.Errorf("To many nodes in the forest")
		}
		e.rootOffsets = append(e.rootOffsets, uint32(len(e.nodes)))
		if err = e.addNode(tree.Root, buildMap, setLeaf); err != nil {
			return err
		}
	}

	return nil
}

// Gets the active leaf of a given example. This method is used during
// inference.
func (e *genericEngine) getLeaf(examples *example.Batch, exampleIdx int, nodeIdx int) *genericNode {
	var node *genericNode

	// This for-loop navigates down the tree from the root to the leaf.
	for {
		node = &e.nodes[nodeIdx]
		var eval bool
		switch node.conditionType {

		case leafConditionType:
			// Leaf
			return node

		case numericalIsHigherConditionType:
			// feature >= threshold condition
			valueIdx := int(node.featureIdx) + exampleIdx*len(e.features.NumericalFeatures)
			eval = examples.NumericalValues[valueIdx] >= math.Float32frombits(node.condition)

		case categoricalContainsBufferConditionType:
			// feature \in mask condition
			valueIdx := int(node.featureIdx) + exampleIdx*len(e.features.CategoricalFeatures)
			bitmapIdx := node.condition + uint32(examples.CategoricalValues[valueIdx])
			eval = GetBit(e.categoricalBitmap, bitmapIdx)

		case categoricalContainsMaskConditionType:
			// feature \in mask condition
			valueIdx := int(node.featureIdx) + exampleIdx*len(e.features.CategoricalFeatures)
			eval = (node.condition & (1 << examples.CategoricalValues[valueIdx])) != 0

		case numericalObliqueProjectionIsHigherType:
			numProjections := uint32(node.featureIdx)
			var accumulator float32 = 0
			for projIdx := uint32(0); projIdx < numProjections; projIdx++ {
				featureIdx := e.obliqueFeatures[node.condition+projIdx]
				valueIdx := int(featureIdx) + exampleIdx*len(e.features.NumericalFeatures)
				accumulator += e.obliqueWeights[node.condition+projIdx] * examples.NumericalValues[valueIdx]
			}
			eval = accumulator >= e.obliqueWeights[node.condition+numProjections]
		}

		// TODO: Optimize with boolean arithmetic when int(bool) is available
		// (https://github.com/golang/go/issues/9367).
		// Note: Looking at go assembly, I was not able to find a way to avoid a cmp
		// and a jmp or a conditional set.
		if eval {
			nodeIdx += int(node.rightIdx)
		} else {
			nodeIdx++
		}
	}
}

// addNode recursively adds a node and its descendants to the engine.
func (e *genericEngine) addNode(srcNode *dt.Node, buildMap *example.FeatureConstructionMap,
	setLeaf SetNodeSignature) error {
	// Allocate the node
	nodeIdx := len(e.nodes)
	e.nodes = append(e.nodes, genericNode{})
	dstNode := &e.nodes[nodeIdx]

	if srcNode.IsLeaf() {
		dstNode.conditionType = leafConditionType
		return setLeaf(srcNode, dstNode)
	}

	// Set the node's condition
	attributeIdx := int(srcNode.RawNode.GetCondition().GetAttribute())
	condition := srcNode.RawNode.GetCondition().GetCondition()

	switch {
	case condition.GetHigherCondition() != nil:
		featureID, found := buildMap.NumericalFeatures[attributeIdx]
		if !found {
			return fmt.Errorf("Cannot find column %v in the input features", attributeIdx)
		}
		if featureID > math.MaxUint16 {
			return fmt.Errorf("Too many features in the model")
		}
		dstNode.featureIdx = uint16(featureID)
		dstNode.condition = math.Float32bits(condition.GetHigherCondition().GetThreshold())
		dstNode.conditionType = numericalIsHigherConditionType

	case condition.GetContainsCondition() != nil:
		featureID, found := buildMap.CategoricalFeatures[attributeIdx]
		if !found {
			return fmt.Errorf("Cannot find column %v in the input features", attributeIdx)
		}

		numUniqueValues := e.features.CategoricalSpec[featureID].NumUniqueValues
		mask := make([]byte, (numUniqueValues+7)/8)
		elements := condition.GetContainsCondition().GetElements()
		for _, element := range elements {
			if element < 0 || uint32(element) >= numUniqueValues {
				return fmt.Errorf("Invalid element")
			}
			SetBit(mask, uint32(element))
		}

		err := e.setCategoricalContainsCondition(featureID, dstNode, numUniqueValues, mask)
		if err != nil {
			return err
		}

	case condition.GetContainsBitmapCondition() != nil:
		featureID, found := buildMap.CategoricalFeatures[attributeIdx]
		if !found {
			return fmt.Errorf("Cannot find column %v in the input features", attributeIdx)
		}

		numUniqueValues := e.features.CategoricalSpec[featureID].NumUniqueValues
		mask := condition.GetContainsBitmapCondition().GetElementsBitmap()

		err := e.setCategoricalContainsCondition(featureID, dstNode, numUniqueValues, mask)
		if err != nil {
			return err
		}

	case condition.GetObliqueCondition() != nil:
		if len(condition.GetObliqueCondition().GetAttributes()) !=
			len(condition.GetObliqueCondition().GetWeights()) {
			return fmt.Errorf("Invalid condition")
		}
		if len(condition.GetObliqueCondition().GetWeights()) >= math.MaxUint16 {
			return fmt.Errorf("Too many projections")
		}
		if len(e.obliqueWeights) != len(e.obliqueFeatures) {
			return fmt.Errorf("Inconsistent internal buffers")
		}

		numProjections := len(condition.GetObliqueCondition().GetWeights())
		dstNode.featureIdx = uint16(numProjections)
		dstNode.condition = uint32(len(e.obliqueWeights))
		dstNode.conditionType = numericalObliqueProjectionIsHigherType

		for projIdx := 0; projIdx < numProjections; projIdx++ {
			localAttributeIdx := int(condition.GetObliqueCondition().GetAttributes()[projIdx])
			localFeatureID, found := buildMap.NumericalFeatures[localAttributeIdx]
			if !found {
				return fmt.Errorf("Cannot find column %v in the input features", localAttributeIdx)
			}
			if localFeatureID > math.MaxUint16 {
				return fmt.Errorf("Too many features in the model")
			}

			e.obliqueWeights = append(e.obliqueWeights, condition.GetObliqueCondition().GetWeights()[projIdx])
			e.obliqueFeatures = append(e.obliqueFeatures, uint16(localFeatureID))
		}

		// Add threshold
		e.obliqueWeights = append(e.obliqueWeights, condition.GetObliqueCondition().GetThreshold())
		e.obliqueFeatures = append(e.obliqueFeatures, 0)

	default:
		return fmt.Errorf("Non supported condition type %v", condition)
	}

	// Build the negative branch.
	if err := e.addNode(srcNode.NegativeChild, buildMap, setLeaf); err != nil {
		return err
	}
	// Note: "dstNode" is now invalid.

	// Set the offset to the positive child
	rightIdx := len(e.nodes) - nodeIdx
	if rightIdx <= 0 {
		return fmt.Errorf("Invalid child")
	}
	if rightIdx > math.MaxUint16 {
		return fmt.Errorf("To many nodes in a single branch")
	}
	e.nodes[nodeIdx].rightIdx = uint16(rightIdx)

	// Build the positive branch.
	if err := e.addNode(srcNode.PositiveChild, buildMap, setLeaf); err != nil {
		return err
	}

	return nil
}

func (e *genericEngine) setCategoricalContainsCondition(featureID example.CategoricalFeatureID,
	dstNode *genericNode, numUniqueValues uint32, mask []byte) error {
	if len(mask) != (int(numUniqueValues)+7)/8 {
		return fmt.Errorf("Unexpected categorical mask size")
	}

	if len(mask) <= 4 {
		// Store the mask in an uint32.
		dstNode.featureIdx = uint16(featureID)
		dstNode.conditionType = categoricalContainsMaskConditionType

		// Converts the mask into a uint32.
		var value uint32
		for i := uint32(0); i < numUniqueValues; i++ {
			if GetBit(mask, i) {
				value |= 1 << i
			}
		}

		dstNode.condition = value
	} else {
		// Store the mask in the byte buffer.
		dstNode.featureIdx = uint16(featureID)
		dstNode.condition = uint32(len(e.categoricalBitmap)) * 8
		dstNode.conditionType = categoricalContainsBufferConditionType
		e.categoricalBitmap = append(e.categoricalBitmap, mask...)
	}

	return nil
}

// GetBit gets the i-th bit in a bitmap.
func GetBit(bitmap []byte, i uint32) bool {
	byteValue := bitmap[i/8]
	return (byteValue & (1 << (i & 7))) != 0
}

// SetBit sets the i-th bit in a bitmap.
func SetBit(bitmap []byte, i uint32) {
	byteIdx := i / 8
	byteValue := bitmap[byteIdx]
	bitmap[byteIdx] = byteValue | (1 << (i & 7))
}

// Identity activation function.
func activationIdentity(value float32) float32 {
	return value
}

// Sigmoid activation function used for binomial log-like losses.
func activationSigmoid(value float32) float32 {
	return 1.0 / (1.0 + expf(-value))
}

func expf(v float32) float32 {
	// TODO: Use a real math32.Exp when available.
	return float32(math.Exp(float64(v)))
}

// OneDimensionEngine is a specialization of the generic engine for models with a single output
// dimension.
type OneDimensionEngine struct {
	Activation        ActivationSignature
	initialPrediction float32
	base              genericEngine
}

// newOneDimensionEngine creates a OneDimensionEngine.
func newOneDimensionEngine(activation ActivationSignature,
	forest *dt.Forest,
	header *model_pb.AbstractModel,
	dataspec *dataspec_pb.DataSpecification,
	initialPrediction float32,
	setNode SetNodeSignature, compatibility example.CompatibilityType) (*OneDimensionEngine, error) {
	engine := &OneDimensionEngine{Activation: activation, initialPrediction: initialPrediction}
	err := engine.base.initialize(forest, header, dataspec, setNode, compatibility)
	if err != nil {
		return nil, err
	}
	return engine, nil
}

// setLeafOneDimensionRegressive sets the value of a single dimension regressive leaf.
func setLeafOneDimensionRegressive(srcNode *dt.Node, dstNode *genericNode) error {
	if srcNode.RawNode.GetRegressor() == nil {
		return fmt.Errorf("Invalid leaf")
	}
	// "condition" contains the float32 value of the leaf.
	dstNode.condition = math.Float32bits(srcNode.RawNode.GetRegressor().GetTopValue())
	return nil
}

// NewBinaryClassificationGBDTGenericEngine creates an engine for a binary
// classification GBT model.
func NewBinaryClassificationGBDTGenericEngine(model *gbt.Model, compatibility example.CompatibilityType) (*OneDimensionEngine, error) {
	if len(model.GbtHeader.GetInitialPredictions()) != 1 {
		return nil, fmt.Errorf("Invalid initial predictions")
	}
	if model.GbtHeader.GetLoss() != gbt_pb.Loss_BINOMIAL_LOG_LIKELIHOOD {
		return nil, fmt.Errorf("Incompatible loss. Expecting log likelihood")
	}

	engine, err := newOneDimensionEngine(activationSigmoid,
		model.Forest,
		model.Header(),
		model.Dataspec(),
		model.GbtHeader.GetInitialPredictions()[0], setLeafOneDimensionRegressive, compatibility)
	return engine, err
}

// NewRegressionGBDTGenericEngine creates an engine for a regression GBT model.
func NewRegressionGBDTGenericEngine(model *gbt.Model, compatibility example.CompatibilityType) (*OneDimensionEngine, error) {
	if len(model.GbtHeader.GetInitialPredictions()) != 1 {
		return nil, fmt.Errorf("Invalid initial predictions")
	}
	if model.GbtHeader.GetLoss() != gbt_pb.Loss_SQUARED_ERROR && model.GbtHeader.GetLoss() != gbt_pb.Loss_POISSON {
		return nil, fmt.Errorf("Incompatible loss. Expecting squared error")
	}

	engine, err := newOneDimensionEngine(activationIdentity,
		model.Forest,
		model.Header(),
		model.Dataspec(),
		model.GbtHeader.GetInitialPredictions()[0], setLeafOneDimensionRegressive, compatibility)
	return engine, err
}

// NewRankingGBDTGenericEngine creates an engine for a regression GBT model.
func NewRankingGBDTGenericEngine(model *gbt.Model, compatibility example.CompatibilityType) (*OneDimensionEngine, error) {
	if len(model.GbtHeader.GetInitialPredictions()) != 1 {
		return nil, fmt.Errorf("Invalid initial predictions")
	}
	if model.GbtHeader.GetLoss() != gbt_pb.Loss_LAMBDA_MART_NDCG5 &&
		model.GbtHeader.GetLoss() != gbt_pb.Loss_XE_NDCG_MART {
		return nil, fmt.Errorf("Incompatible loss. Expecting squared error")
	}

	engine, err := newOneDimensionEngine(activationIdentity,
		model.Forest,
		model.Header(),
		model.Dataspec(),
		model.GbtHeader.GetInitialPredictions()[0], setLeafOneDimensionRegressive, compatibility)
	return engine, err
}

// NewBinaryClassificationRFGenericEngine creates an engine for a binary
// classification RF model.
func NewBinaryClassificationRFGenericEngine(model *rf.Model, compatibility example.CompatibilityType) (*OneDimensionEngine, error) {
	setLeaf := func(srcNode *dt.Node, dstNode *genericNode) error {
		if srcNode.RawNode.GetClassifier() == nil {
			return fmt.Errorf("Invalid leaf")
		}
		// "condition" contains the float32 value of the leaf.

		var leafValue float32 = 0
		if model.RfHeader.GetWinnerTakeAllInference() {
			if srcNode.RawNode.GetClassifier().GetTopValue() == 2 {
				leafValue = 1.0 / float32(len(model.Forest.Trees))
			}
		} else {
			leafValue = float32(srcNode.RawNode.GetClassifier().GetDistribution().GetCounts()[2] /
				(srcNode.RawNode.GetClassifier().GetDistribution().GetSum() * float64(len(model.Forest.Trees))))
		}

		dstNode.condition = math.Float32bits(leafValue)
		return nil
	}
	engine, err := newOneDimensionEngine(activationIdentity,
		model.Forest,
		model.Header(),
		model.Dataspec(),
		0, setLeaf, compatibility)
	return engine, err
}

// NewRegressionRFGenericEngine creates an engine for a regression RF model.
func NewRegressionRFGenericEngine(model *rf.Model, compatibility example.CompatibilityType) (*OneDimensionEngine, error) {
	setLeaf := func(srcNode *dt.Node, dstNode *genericNode) error {
		if srcNode.RawNode.GetRegressor() == nil {
			return fmt.Errorf("Invalid leaf")
		}

		var leafValue = srcNode.RawNode.GetRegressor().GetTopValue() / float32(len(model.Forest.Trees))
		// "condition" contains the float32 value of the leaf.
		dstNode.condition = math.Float32bits(leafValue)
		return nil
	}
	engine, err := newOneDimensionEngine(activationIdentity,
		model.Forest,
		model.Header(),
		model.Dataspec(),
		0, setLeaf, compatibility)
	return engine, err
}

// AllocateExamples allocates a set of examples.
func (e *OneDimensionEngine) AllocateExamples(maxNumExamples int) *example.Batch {
	return example.NewBatch(maxNumExamples, e.Features())
}

// AllocatePredictions allocates a set of predictions.
func (e *OneDimensionEngine) AllocatePredictions(maxNumExamples int) []float32 {
	// Works before OutputDim() == 1.
	return make([]float32, maxNumExamples)
}

// Features of the engine.
func (e *OneDimensionEngine) Features() *example.Features {
	return e.base.features
}

// OutputDim is the output dimension of the engine.
func (e *OneDimensionEngine) OutputDim() int {
	return 1
}

// Predict generates predictions with the engine.
func (e *OneDimensionEngine) Predict(examples *example.Batch, numExamples int, predictions []float32) {
	for exampleIdx := 0; exampleIdx < numExamples; exampleIdx++ {
		prediction := e.initialPrediction
		for _, rootOffset := range e.base.rootOffsets {
			leaf := e.base.getLeaf(examples, exampleIdx, int(rootOffset))
			prediction += math.Float32frombits(leaf.condition)
		}
		predictions[exampleIdx] = e.Activation(prediction)
	}
}
