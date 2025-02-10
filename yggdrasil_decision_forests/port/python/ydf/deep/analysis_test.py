# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import numpy.testing as npt

from yggdrasil_decision_forests.dataset import data_spec_pb2
from yggdrasil_decision_forests.dataset import example_pb2
from ydf.dataset import dataspec as dataspec_lib
from ydf.deep import analysis as py_analysis_lib
from ydf.model import generic_model
from ydf.utils import test_utils
from yggdrasil_decision_forests.utils import distribution_pb2
from yggdrasil_decision_forests.utils import model_analysis_pb2
from yggdrasil_decision_forests.utils import partial_dependence_plot_pb2


Semantic = dataspec_lib.Semantic
InputFeature = generic_model.InputFeature
PartialDependencePlotSet = partial_dependence_plot_pb2.PartialDependencePlotSet
PartialDependencePlot = PartialDependencePlotSet.PartialDependencePlot
AttributeInfo = PartialDependencePlot.AttributeInfo
PDPBin = PartialDependencePlot.Bin
LabelAccumulator = PartialDependencePlot.LabelAccumulator
Attribute = example_pb2.Example.Attribute
VocabValue = data_spec_pb2.CategoricalSpec.VocabValue


class AnalysisTest(parameterized.TestCase):

  def test_get_pdp_features(self):
    input_features: List[InputFeature] = [
        InputFeature(name="f_cat", semantic=Semantic.CATEGORICAL, column_idx=0),
        InputFeature(name="f_num", semantic=Semantic.NUMERICAL, column_idx=1),
        InputFeature(name="f_hash", semantic=Semantic.HASH, column_idx=2),
        InputFeature(
            name="f_catset", semantic=Semantic.CATEGORICAL_SET, column_idx=3
        ),
        InputFeature(name="f_bool", semantic=Semantic.BOOLEAN, column_idx=4),
        InputFeature(
            name="f_discnum",
            semantic=Semantic.DISCRETIZED_NUMERICAL,
            column_idx=5,
        ),
        InputFeature(
            name="f_nvs",
            semantic=Semantic.NUMERICAL_VECTOR_SEQUENCE,
            column_idx=6,
        ),
    ]
    pdp_features = py_analysis_lib.get_pdp_features(input_features)
    expected_features = [
        InputFeature(name="f_cat", semantic=Semantic.CATEGORICAL, column_idx=0),
        InputFeature(name="f_num", semantic=Semantic.NUMERICAL, column_idx=1),
        InputFeature(name="f_bool", semantic=Semantic.BOOLEAN, column_idx=4),
    ]
    self.assertEqual(pdp_features, expected_features)

  @parameterized.parameters(-1, 0, 1)
  def test_get_numerical_bins_invalid(self, num_numerical_bins):
    spec_boundaries = np.arange(0, 6)
    with self.assertRaises(ValueError):
      py_analysis_lib.get_numerical_bins(num_numerical_bins, spec_boundaries)

  def test_get_numerical_bins_identity(self):
    spec_boundaries = np.array([0, 1, 2, 3, 4])
    num_numerical_bins = 6
    expected_centers = np.array([0, 0.5, 1.5, 2.5, 3.5, 4])
    expected_boundaries = np.array([0, 1, 2, 3, 4])
    centers, boundaries = py_analysis_lib.get_numerical_bins(
        num_numerical_bins, spec_boundaries
    )
    npt.assert_array_equal(centers, expected_centers)
    npt.assert_array_equal(boundaries, expected_boundaries)

  def test_get_numerical_bins_nonlinear_aligned(self):
    spec_boundaries = np.array([0, 1.5, 2, 2.5, 4])
    num_numerical_bins = 6
    expected_centers = np.array([0, 0.75, 1.75, 2.25, 3.25, 4])
    expected_boundaries = np.array([0, 1.5, 2, 2.5, 4])
    centers, boundaries = py_analysis_lib.get_numerical_bins(
        num_numerical_bins, spec_boundaries
    )
    npt.assert_array_equal(centers, expected_centers)
    npt.assert_array_equal(boundaries, expected_boundaries)

  def test_get_numerical_bins_nonlinear_unaligned(self):
    spec_boundaries = np.array([0, 1.5, 2, 2.5, 4])
    num_numerical_bins = 10
    expected_centers = np.array(
        [0, 0.375, 1.125, 1.625, 1.875, 2.125, 2.375, 2.875, 3.625, 4]
    )
    expected_boundaries = np.array([0, 0.75, 1.5, 1.75, 2, 2.25, 2.5, 3.25, 4])
    centers, boundaries = py_analysis_lib.get_numerical_bins(
        num_numerical_bins, spec_boundaries
    )
    npt.assert_array_equal(centers, expected_centers)
    npt.assert_array_equal(boundaries, expected_boundaries)

  def test_get_numerical_bins_nonlinear_negative(self):
    spec_boundaries = np.array([-4, -2.5, -2, -1.5, 0])
    num_numerical_bins = 10
    expected_centers = np.array([
        -4.0,
        -3.625,
        -2.875,
        -2.375,
        -2.125,
        -1.875,
        -1.625,
        -1.125,
        -0.375,
        0.0,
    ])
    expected_boundaries = np.array(
        [-4.0, -3.25, -2.5, -2.25, -2.0, -1.75, -1.5, -0.75, 0.0]
    )
    centers, boundaries = py_analysis_lib.get_numerical_bins(
        num_numerical_bins, spec_boundaries
    )
    npt.assert_array_equal(centers, expected_centers)
    npt.assert_array_equal(boundaries, expected_boundaries)

  @parameterized.parameters(
      {"val": -1, "expected_result": 0},
      {"val": 0.0, "expected_result": 1},
      {"val": 1.0, "expected_result": 2},
      {"val": 2.0, "expected_result": 2},
      {"val": 3.0, "expected_result": 3},
      {"val": 4.0, "expected_result": 3},
      {"val": 10.0, "expected_result": 4},
      {"val": 10.1, "expected_result": 4},
      {"val": float("nan"), "expected_result": 4},
  )
  def test_get_attribute_bin_idx_numerical(self, val, expected_result):
    data_spec = data_spec_pb2.DataSpecification(
        columns=[
            data_spec_pb2.Column(name="f_num1", type=data_spec_pb2.NUMERICAL),
            data_spec_pb2.Column(
                name="f_num2",
                type=data_spec_pb2.NUMERICAL,
                numerical=data_spec_pb2.NumericalSpec(
                    min_value=0.0, max_value=10.0
                ),
            ),
        ]
    )
    attr_info = AttributeInfo(
        attribute_idx=1, numerical_boundaries=[0.0, 1.0, 3.0, 10.0]
    )
    example = {
        "f_num1": np.array([1.0]),
        "f_num2": np.array([val]),
    }
    bin_idx = py_analysis_lib.get_attribute_bin_idx(
        example, data_spec, attr_info
    )
    self.assertEqual(bin_idx, expected_result)

  @parameterized.parameters(
      {"val": "a", "expected_result": 3},
      {"val": "b", "expected_result": 2},
      {"val": "c", "expected_result": 1},
      {"val": float("nan"), "expected_result": 0},
      {"val": "not_in_spec", "expected_result": 0},
  )
  def test_get_attribute_bin_idx_categorical(self, val, expected_result):
    data_spec = data_spec_pb2.DataSpecification(
        columns=[
            data_spec_pb2.Column(name="f_cat1", type=data_spec_pb2.CATEGORICAL),
            data_spec_pb2.Column(
                name="f_cat2",
                type=data_spec_pb2.CATEGORICAL,
                categorical=data_spec_pb2.CategoricalSpec(
                    number_of_unique_values=4,
                    items={
                        "<OOD>": VocabValue(index=0, count=2),
                        "a": VocabValue(index=3, count=2),
                        "b": VocabValue(index=2, count=1),
                        "c": VocabValue(index=1, count=2),
                    },
                ),
            ),
        ]
    )
    attr_info = AttributeInfo(attribute_idx=1)
    example = {
        "f_cat1": np.array(["foo"]),
        "f_cat2": np.array([val]),
    }
    bin_idx = py_analysis_lib.get_attribute_bin_idx(
        example, data_spec, attr_info
    )
    self.assertEqual(bin_idx, expected_result)

  def test_generate_modified_examples_numerical(self):
    example = {
        "f_num": np.array([1.1]),
        "f_cat": np.array(["a"]),
        "f_bool": np.array([True]),
    }
    pdp = PartialDependencePlot(
        pdp_bins=[
            PartialDependencePlot.Bin(
                center_input_feature_values=[Attribute(numerical=-1.0)]
            ),
            PartialDependencePlot.Bin(
                center_input_feature_values=[Attribute(numerical=0.0)]
            ),
            PartialDependencePlot.Bin(
                center_input_feature_values=[Attribute(numerical=1.23)]
            ),
            PartialDependencePlot.Bin(
                center_input_feature_values=[Attribute(numerical=1000)]
            ),
        ]
    )
    vocab_mappings = {"f_cat": ["<OOD>", "d", "c", "b", "a"]}
    modified_examples = py_analysis_lib.generate_modified_examples(
        example, pdp, "f_num", data_spec_pb2.NUMERICAL, vocab_mappings
    )
    self.assertSameElements(
        modified_examples.keys(), ["f_num", "f_cat", "f_bool"]
    )
    npt.assert_array_almost_equal(
        modified_examples["f_num"], np.array([-1.0, 0.0, 1.23, 1000.0])
    )
    npt.assert_array_equal(
        modified_examples["f_cat"], np.array(["a", "a", "a", "a"])
    )
    npt.assert_array_equal(
        modified_examples["f_bool"], np.array([True, True, True, True])
    )

  def test_generate_modified_examples_categorical(self):
    example = {
        "f_num": np.array([1.1]),
        "f_cat": np.array(["a"]),
        "f_bool": np.array([True]),
    }
    pdp = PartialDependencePlot(
        pdp_bins=[
            PartialDependencePlot.Bin(
                center_input_feature_values=[
                    Attribute(categorical=1),
                ]
            ),
            PartialDependencePlot.Bin(
                center_input_feature_values=[
                    Attribute(categorical=2),
                ]
            ),
            PartialDependencePlot.Bin(
                center_input_feature_values=[
                    Attribute(categorical=3),
                ]
            ),
            PartialDependencePlot.Bin(
                center_input_feature_values=[
                    Attribute(categorical=4),
                ]
            ),
        ]
    )
    vocab_mappings = {"f_cat": ["<OOD>", "d", "c", "b", "a"]}
    modified_examples = py_analysis_lib.generate_modified_examples(
        example, pdp, "f_cat", data_spec_pb2.CATEGORICAL, vocab_mappings
    )
    self.assertSameElements(
        modified_examples.keys(), ["f_num", "f_cat", "f_bool"]
    )
    npt.assert_array_almost_equal(
        modified_examples["f_num"], np.array([1.1, 1.1, 1.1, 1.1])
    )
    npt.assert_array_equal(
        modified_examples["f_cat"], np.array(["d", "c", "b", "a"])
    )
    npt.assert_array_equal(
        modified_examples["f_bool"], np.array([True, True, True, True])
    )

  def test_generate_modified_examples_boolean(self):
    example = {
        "f_num": np.array([1.1]),
        "f_cat": np.array(["a"]),
        "f_bool": np.array([True]),
    }
    pdp = PartialDependencePlot(
        pdp_bins=[
            PartialDependencePlot.Bin(
                center_input_feature_values=[
                    Attribute(boolean=False),
                ]
            ),
            PartialDependencePlot.Bin(
                center_input_feature_values=[
                    Attribute(boolean=True),
                ]
            ),
        ]
    )

    vocab_mappings = {"f_cat": ["<OOD>", "d", "c", "b", "a"]}
    modified_examples = py_analysis_lib.generate_modified_examples(
        example, pdp, "f_bool", data_spec_pb2.BOOLEAN, vocab_mappings
    )
    self.assertSameElements(
        modified_examples.keys(), ["f_num", "f_cat", "f_bool"]
    )
    npt.assert_array_almost_equal(
        modified_examples["f_num"], np.array([1.1, 1.1])
    )
    npt.assert_array_equal(modified_examples["f_cat"], np.array(["a", "a"]))
    npt.assert_array_equal(modified_examples["f_bool"], np.array([False, True]))

  def test_update_bin_with_prediction_binary_class(self):
    pdp_bin = PartialDependencePlot.Bin()
    label_spec = data_spec_pb2.Column(
        name="label",
        type=data_spec_pb2.CATEGORICAL,
        categorical=data_spec_pb2.CategoricalSpec(
            number_of_unique_values=3,
            items={
                "<OOD>": VocabValue(index=0, count=2),
                "neg": VocabValue(index=1, count=2),
                "pos": VocabValue(index=2, count=1),
            },
        ),
    )
    py_analysis_lib.initialize_bin_prediction_counters(
        [pdp_bin], label_spec, generic_model.Task.CLASSIFICATION
    )
    py_analysis_lib.update_bin_with_prediction(pdp_bin, 0.8)
    expected_bin_prediction = distribution_pb2.IntegerDistributionFloat(
        counts=[0.0, 0.2, 0.8], sum=1.0
    )
    test_utils.assertProto2Equal(
        self,
        pdp_bin.prediction.classification_class_distribution,
        expected_bin_prediction,
    )

  def test_update_bin_with_prediction_multi_class(self):
    pdp_bin = PartialDependencePlot.Bin()
    label_spec = data_spec_pb2.Column(
        name="label",
        type=data_spec_pb2.CATEGORICAL,
        categorical=data_spec_pb2.CategoricalSpec(
            number_of_unique_values=4,
            items={
                "<OOD>": VocabValue(index=0, count=2),
                "a": VocabValue(index=1, count=2),
                "b": VocabValue(index=2, count=1),
                "c": VocabValue(index=3, count=1),
            },
        ),
    )
    py_analysis_lib.initialize_bin_prediction_counters(
        [pdp_bin], label_spec, generic_model.Task.CLASSIFICATION
    )
    py_analysis_lib.update_bin_with_prediction(
        pdp_bin, np.array([0.1, 0.2, 0.7])
    )
    expected_bin_prediction = distribution_pb2.IntegerDistributionFloat(
        counts=[0.0, 0.1, 0.2, 0.7], sum=1.0
    )
    test_utils.assertProto2Equal(
        self,
        pdp_bin.prediction.classification_class_distribution,
        expected_bin_prediction,
    )
    py_analysis_lib.update_bin_with_prediction(
        pdp_bin, np.array([0.2, 0.3, 0.5])
    )
    expected_bin_prediction = distribution_pb2.IntegerDistributionFloat(
        counts=[0.0, 0.3, 0.5, 1.2], sum=2.0
    )
    test_utils.assertProto2Equal(
        self,
        pdp_bin.prediction.classification_class_distribution,
        expected_bin_prediction,
    )

  def test_update_bin_with_prediction_regression(self):
    pdp_bin = PartialDependencePlot.Bin()
    label_spec = data_spec_pb2.Column(
        name="label",
        type=data_spec_pb2.NUMERICAL,
    )
    py_analysis_lib.initialize_bin_prediction_counters(
        [pdp_bin], label_spec, generic_model.Task.REGRESSION
    )
    py_analysis_lib.update_bin_with_prediction(pdp_bin, 3.14)
    self.assertEqual(
        pdp_bin.prediction.sum_of_regression_predictions,
        3.14,
    )
    py_analysis_lib.update_bin_with_prediction(pdp_bin, 2.71)
    self.assertEqual(
        pdp_bin.prediction.sum_of_regression_predictions,
        5.85,
    )

  def test_initialize_pdp_binary_classification(self):
    mock_model = mock.MagicMock()
    mock_model.data_spec = lambda: data_spec_pb2.DataSpecification(
        columns=[
            data_spec_pb2.Column(
                name="f_num",
                type=data_spec_pb2.NUMERICAL,
                numerical=data_spec_pb2.NumericalSpec(min_value=0, max_value=1),
                discretized_numerical=data_spec_pb2.DiscretizedNumericalSpec(
                    boundaries=[0.0, 1.0, 2.0, 3.0]
                ),
            ),
            data_spec_pb2.Column(
                name="f_catset", type=data_spec_pb2.CATEGORICAL_SET
            ),
            data_spec_pb2.Column(
                name="f_cat_multi",
                type=data_spec_pb2.CATEGORICAL,
                categorical=data_spec_pb2.CategoricalSpec(
                    number_of_unique_values=4,
                    items={
                        "<OOD>": VocabValue(index=0, count=2),
                        "a": VocabValue(index=1, count=2),
                        "b": VocabValue(index=2, count=1),
                        "c": VocabValue(index=3, count=1),
                    },
                ),
            ),
            data_spec_pb2.Column(
                name="label",
                type=data_spec_pb2.CATEGORICAL,
                categorical=data_spec_pb2.CategoricalSpec(
                    number_of_unique_values=3,
                    items={
                        "<OOD>": VocabValue(index=0, count=2),
                        "neg": VocabValue(index=1, count=2),
                        "pos": VocabValue(index=2, count=1),
                    },
                ),
            ),
        ]
    )
    mock_model.label_col_idx = lambda: 2
    mock_model.task = lambda: generic_model.Task.CLASSIFICATION
    pdp = PartialDependencePlot()
    feature = InputFeature(
        name="f_num", semantic=Semantic.NUMERICAL, column_idx=0
    )
    py_analysis_lib.initialize_pdp(pdp, feature, mock_model, 5)
    self.assertEqual(pdp.type, PartialDependencePlot.Type.PDP)
    self.assertLen(pdp.pdp_bins, 5)
    expected_bin_centers = [0, 0.5, 1.5, 2.5, 3]
    for bin_idx, pdp_bin in enumerate(pdp.pdp_bins):
      self.assertLen(pdp_bin.center_input_feature_values, 1)
      self.assertEqual(
          pdp_bin.center_input_feature_values[0].numerical,
          expected_bin_centers[bin_idx],
      )
      self.assertTrue(
          pdp_bin.prediction.HasField("classification_class_distribution")
      )
    self.assertLen(pdp.attribute_info, 1)
    self.assertEqual(pdp.attribute_info[0].attribute_idx, 0)
    self.assertEqual(pdp.attribute_info[0].num_bins_per_input_feature, 5)
    self.assertSequenceEqual(
        pdp.attribute_info[0].numerical_boundaries, [0.0, 1.0, 2.0, 3.0]
    )
    self.assertEqual(pdp.attribute_info[0].scale, AttributeInfo.Scale.UNIFORM)

  def test_initialize_pdp_classification(self):
    mock_model = mock.MagicMock()
    mock_model.data_spec = lambda: data_spec_pb2.DataSpecification(
        columns=[
            data_spec_pb2.Column(
                name="f_num",
                type=data_spec_pb2.NUMERICAL,
                numerical=data_spec_pb2.NumericalSpec(min_value=0, max_value=3),
                discretized_numerical=data_spec_pb2.DiscretizedNumericalSpec(
                    boundaries=[0.0, 1.0, 2.0, 3.0]
                ),
            ),
            data_spec_pb2.Column(
                name="f_catset", type=data_spec_pb2.CATEGORICAL_SET
            ),
            data_spec_pb2.Column(
                name="f_cat_multi",
                type=data_spec_pb2.CATEGORICAL,
                categorical=data_spec_pb2.CategoricalSpec(
                    number_of_unique_values=4,
                    items={
                        "<OOD>": VocabValue(index=0, count=2),
                        "a": VocabValue(index=1, count=2),
                        "b": VocabValue(index=2, count=1),
                        "c": VocabValue(index=3, count=1),
                    },
                ),
            ),
            data_spec_pb2.Column(
                name="f_cat_binary",
                type=data_spec_pb2.CATEGORICAL,
                categorical=data_spec_pb2.CategoricalSpec(
                    number_of_unique_values=3,
                    items={
                        "<OOD>": VocabValue(index=0, count=2),
                        "neg": VocabValue(index=1, count=2),
                        "pos": VocabValue(index=2, count=1),
                    },
                ),
            ),
        ]
    )
    mock_model.label_col_idx = lambda: 0
    mock_model.task = lambda: generic_model.Task.REGRESSION
    pdp = PartialDependencePlot()
    feature = InputFeature(
        name="f_cat_binary", semantic=Semantic.CATEGORICAL, column_idx=3
    )
    py_analysis_lib.initialize_pdp(pdp, feature, mock_model, 5)
    self.assertEqual(pdp.type, PartialDependencePlot.Type.PDP)
    self.assertLen(pdp.pdp_bins, 3)
    for bin_idx, pdp_bin in enumerate(pdp.pdp_bins):
      self.assertLen(pdp_bin.center_input_feature_values, 1)
      self.assertEqual(
          pdp_bin.center_input_feature_values[0].categorical,
          bin_idx,
      )
      self.assertTrue(
          pdp_bin.prediction.HasField("sum_of_regression_predictions")
      )
    self.assertLen(pdp.attribute_info, 1)
    self.assertEqual(pdp.attribute_info[0].attribute_idx, 3)
    self.assertEqual(pdp.attribute_info[0].num_bins_per_input_feature, 3)

  def test_toy_pdp_regression(self):
    mock_model = mock.MagicMock()
    mock_model.data_spec = lambda: data_spec_pb2.DataSpecification(
        columns=[
            data_spec_pb2.Column(
                name="f_num",
                type=data_spec_pb2.NUMERICAL,
                numerical=data_spec_pb2.NumericalSpec(min_value=0, max_value=3),
                discretized_numerical=data_spec_pb2.DiscretizedNumericalSpec(
                    boundaries=[0.0, 1.0, 2.0, 3.0]
                ),
            ),
            data_spec_pb2.Column(name="f_bool", type=data_spec_pb2.BOOLEAN),
            data_spec_pb2.Column(
                name="f_cat_binary",
                type=data_spec_pb2.CATEGORICAL,
                categorical=data_spec_pb2.CategoricalSpec(
                    number_of_unique_values=3,
                    items={
                        "<OOD>": VocabValue(index=0, count=2),
                        "neg": VocabValue(index=1, count=2),
                        "pos": VocabValue(index=2, count=1),
                    },
                ),
            ),
            data_spec_pb2.Column(
                name="label_num",
                type=data_spec_pb2.NUMERICAL,
                numerical=data_spec_pb2.NumericalSpec(min_value=0, max_value=3),
                discretized_numerical=data_spec_pb2.DiscretizedNumericalSpec(
                    boundaries=[0.0, 1.0, 2.0, 3.0]
                ),
            ),
        ]
    )
    mock_model.label_col_idx = lambda: 3
    mock_model.task = lambda: generic_model.Task.REGRESSION

    mock_model.input_features = lambda: [
        InputFeature(name="f_num", semantic=Semantic.NUMERICAL, column_idx=0),
        InputFeature(
            name="f_bool", semantic=Semantic.CATEGORICAL, column_idx=1
        ),
        InputFeature(
            name="f_cat_binary", semantic=Semantic.BOOLEAN, column_idx=2
        ),
    ]
    mock_model.predict = lambda x: x["f_num"] * x["f_bool"]
    options = model_analysis_pb2.Options(
        pdp=model_analysis_pb2.Options.PlotConfig(num_numerical_bins=5)
    )
    data = {
        "f_num": np.array([1.5, 2.5]),
        "f_bool": np.array([True, False]),
        "f_cat_binary": np.array(["pos", "not_in_vocab"]),
        "label_num": np.array([1.5, 2.5]),
    }
    analysis = py_analysis_lib.model_analysis(
        model=mock_model, data=data, options=options
    )
    expected_pdp_set = PartialDependencePlotSet(
        pdps=[
            PartialDependencePlot(
                num_observations=2,
                pdp_bins=[
                    PDPBin(
                        prediction=LabelAccumulator(
                            sum_of_regression_predictions=0.0
                        ),
                        center_input_feature_values=[Attribute(numerical=0.0)],
                    ),
                    PDPBin(
                        prediction=LabelAccumulator(
                            sum_of_regression_predictions=0.5
                        ),
                        center_input_feature_values=[Attribute(numerical=0.5)],
                    ),
                    PDPBin(
                        prediction=LabelAccumulator(
                            sum_of_regression_predictions=1.5
                        ),
                        center_input_feature_values=[Attribute(numerical=1.5)],
                    ),
                    PDPBin(
                        prediction=LabelAccumulator(
                            sum_of_regression_predictions=2.5
                        ),
                        center_input_feature_values=[Attribute(numerical=2.5)],
                    ),
                    PDPBin(
                        prediction=LabelAccumulator(
                            sum_of_regression_predictions=3.0
                        ),
                        center_input_feature_values=[Attribute(numerical=3.0)],
                    ),
                ],
                attribute_info=[
                    AttributeInfo(
                        num_bins_per_input_feature=5,
                        attribute_idx=0,
                        num_observations_per_bins=[0, 0, 1, 1, 0],
                        numerical_boundaries=[0.0, 1.0, 2.0, 3.0],
                    )
                ],
                type=PartialDependencePlot.Type.PDP,
            ),
            PartialDependencePlot(
                num_observations=2,
                pdp_bins=[
                    PDPBin(
                        prediction=LabelAccumulator(
                            sum_of_regression_predictions=0.0
                        ),
                        center_input_feature_values=[Attribute(boolean=False)],
                    ),
                    PDPBin(
                        prediction=LabelAccumulator(
                            sum_of_regression_predictions=4.0
                        ),
                        center_input_feature_values=[Attribute(boolean=True)],
                    ),
                ],
                attribute_info=[
                    AttributeInfo(
                        num_bins_per_input_feature=2,
                        attribute_idx=1,
                        num_observations_per_bins=[1, 1],
                    )
                ],
                type=PartialDependencePlot.Type.PDP,
            ),
            PartialDependencePlot(
                num_observations=2,
                pdp_bins=[
                    PDPBin(
                        prediction=LabelAccumulator(
                            sum_of_regression_predictions=1.5
                        ),
                        center_input_feature_values=[Attribute(categorical=0)],
                    ),
                    PDPBin(
                        prediction=LabelAccumulator(
                            sum_of_regression_predictions=1.5
                        ),
                        center_input_feature_values=[Attribute(categorical=1)],
                    ),
                    PDPBin(
                        prediction=LabelAccumulator(
                            sum_of_regression_predictions=1.5
                        ),
                        center_input_feature_values=[Attribute(categorical=2)],
                    ),
                ],
                attribute_info=[
                    AttributeInfo(
                        num_bins_per_input_feature=3,
                        attribute_idx=2,
                        num_observations_per_bins=[1, 0, 1],
                    )
                ],
                type=PartialDependencePlot.Type.PDP,
            ),
        ]
    )
    test_utils.assertProto2Equal(
        self, analysis._analysis_proto.core_analysis.pdp_set, expected_pdp_set
    )


if __name__ == "__main__":
  absltest.main()
