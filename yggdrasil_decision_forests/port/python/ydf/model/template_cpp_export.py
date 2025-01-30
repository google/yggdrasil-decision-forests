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

"""Template to export a model to c++."""

import datetime
import re
from typing import List
from yggdrasil_decision_forests.dataset import data_spec_pb2
from ydf import version


def _feature_name_to_variable(name: str, prefix: str = "feature_") -> str:
  """Convert a feature into a variable name."""

  # Replace spaces and non-printable characters with "_".
  name = re.sub(r"[\W ]", "_", name)
  return f"{prefix}{name}"


def _cc_string(value: str) -> str:
  """Escapes a c++ string. Does not include the quotes."""
  return value.replace('"', '\\"')


def template(
    key: str,
    data_spec: data_spec_pb2.DataSpecification,
    input_features: List[int],
) -> str:
  """Returns the c++ template of exported model.

  Args:
    key: Identifier of the model. Used to create the namespace and define
      protection.
    data_spec: Dataspec of the model.
    input_features: Input features of the model.
  """

  feature_vars = []
  feature_index = []
  feature_sets_1 = []
  feature_sets_2 = []

  for feature_idx in input_features:
    col_spec = data_spec.columns[feature_idx]
    variable = _feature_name_to_variable(col_spec.name)

    if col_spec.type == data_spec_pb2.ColumnType.NUMERICAL:
      feature_vars.append(f"  serving_api::NumericalFeatureId {variable};")
      feature_index.append(
          f"  ASSIGN_OR_RETURN(m.{variable},"
          f' m.features->GetNumericalFeatureId("{_cc_string(col_spec.name)}"));'
      )
      feature_sets_1.append(
          f"  examples->SetNumerical(/*example_idx=*/0, {variable}, 1.f,"
          " *features);"
      )
      feature_sets_2.append(
          f"  examples->SetNumerical(/*example_idx=*/1, {variable}, 2.f,"
          " *features);"
      )
    elif col_spec.type == data_spec_pb2.ColumnType.CATEGORICAL:
      feature_vars.append(f"  serving_api::CategoricalFeatureId {variable};")
      feature_index.append(
          f"  ASSIGN_OR_RETURN(m.{variable},"
          f' m.features->GetCategoricalFeatureId("{_cc_string(col_spec.name)}"));'
      )
      feature_sets_1.append(
          f'  examples->SetCategorical(/*example_idx=*/0, {variable}, "A",'
          " *features);"
      )
      feature_sets_2.append(
          f'  examples->SetCategorical(/*example_idx=*/1, {variable}, "B",'
          " *features);"
      )
    elif col_spec.type == data_spec_pb2.ColumnType.BOOLEAN:
      feature_vars.append(f"  serving_api::BooleanFeatureId {variable};")
      feature_index.append(
          f"  ASSIGN_OR_RETURN(m.{variable},"
          f' m.features->GetBooleanFeatureId("{_cc_string(col_spec.name)}"));'
      )
      feature_sets_1.append(
          f"  examples->SetBoolean(/*example_idx=*/0, {variable}, true,"
          " *features);"
      )
      feature_sets_2.append(
          f"  examples->SetBoolean(/*example_idx=*/1, {variable}, false,"
          " *features);"
      )
    elif col_spec.type == data_spec_pb2.ColumnType.CATEGORICAL_SET:
      feature_vars.append(f"  serving_api::CategoricalSetFeatureId {variable};")
      feature_index.append(
          f"  ASSIGN_OR_RETURN(m.{variable},"
          f' m.features->GetCategoricalFeatureId("{_cc_string(col_spec.name)}"));'
      )
      feature_sets_1.append(
          f"  examples->SetCategoricalSet(/*example_idx=*/0, {variable},"
          ' std::vector<std::string>{"hello", "world"}, *features);'
      )
      feature_sets_2.append(
          f"  examples->SetCategoricalSet(/*example_idx=*/1, {variable},"
          ' std::vector<std::string> {"blue", "dog"}, *features);'
      )
    else:
      raise ValueError(
          "The automatic c++ exporter does not support yet feature"
          f" {col_spec.name!r} with semantic"
          f" {data_spec_pb2.ColumnType.Name(col_spec.type)}. Use the serving"
          " API manually instead."
      )

  str_feature_vars = "\n".join(feature_vars)
  str_feature_index = "\n".join(feature_index)
  str_feature_sets_1 = "\n".join(feature_sets_1)
  str_feature_sets_2 = "\n".join(feature_sets_2)

  return f"""\
// Automatically generated code running an Yggdrasil Decision Forests model in
// C++. This code was generated with "model.to_cpp()".
//
// Date of generation: {datetime.datetime.now()}
// YDF Version: {version.version}
//
// How to use this code:
//
// 1. Copy this code in a new .h file.
// 2. If you use Bazel/Blaze, use the following dependencies:
//      //third_party/absl/status:statusor
//      //third_party/absl/strings
//      //external/ydf_cc/yggdrasil_decision_forests/api:serving
// 3. In your existing code, include the .h file. Make predictions as follows:
//   // Load the model (to do only once).
//   namespace ydf = yggdrasil_decision_forests;
//   const auto model = ydf::exported_model_123::Load(<path to model>);
//   // Run the model
//   predictions = model.Predict();
// 4. By default, the "Predict" function takes no inputs and creates fake
//   examples. In practice, you want to add your input data as arguments to
//   "Predict" and call "examples->Set..." functions accordingly.
// 5. (Bonus)
//   Allocate one `examples` and `predictions` vector per thread and reuse them
//   to speed-up the inference.
//
#ifndef YGGDRASIL_DECISION_FORESTS_GENERATED_MODEL_{key}
#define YGGDRASIL_DECISION_FORESTS_GENERATED_MODEL_{key}

#include <memory>
#include <vector>

#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "external/ydf_cc/yggdrasil_decision_forests/api/serving.h"

namespace yggdrasil_decision_forests {{
namespace exported_model_{key} {{

struct ServingModel {{
  std::vector<float> Predict() const;

  // Compiled model.
  std::unique_ptr<serving_api::FastEngine> engine;

  // Index of the input features of the model.
  //
  // Non-owning pointer. The data is owned by the engine.
  const serving_api::FeaturesDefinition* features;

  // Number of output predictions for each example.
  // Equal to 1 for regression, ranking and binary classification with compact
  // format. Equal to the number of classes for classification.
  int NumPredictionDimension() const {{
    return engine->NumPredictionDimension();
  }}

  // Indexes of the input features.
{str_feature_vars}
}};

// TODO: Pass input feature values to "Predict".
inline std::vector<float> ServingModel::Predict() const {{
  // Allocate memory for 2 examples. Alternatively, for speed-sensitive code,
  // an "examples" object can be allocated for each thread and reused. It is
  // okay to allocate more examples than needed.
  const int num_examples = 2;
  auto examples = engine->AllocateExamples(num_examples);

  // Set all the values to be missing. The values may then be overridden by the
  // "Set*" methods. If all the values are set with "Set*" methods,
  // "FillMissing" can be skipped.
  examples->FillMissing(*features);

  // Example #0
{str_feature_sets_1}

  // Example #1
{str_feature_sets_2}

  // Run the model on the two examples.
  //
  // For speed-sensitive code, reuse the same predictions.
  std::vector<float> predictions;
  engine->Predict(*examples, num_examples, &predictions);
  return predictions;
}}

inline absl::StatusOr<ServingModel> Load(absl::string_view path) {{
  ServingModel m;

  // Load the model
  ASSIGN_OR_RETURN(auto model, serving_api::LoadModel(path));

  // Compile the model into an inference engine.
  ASSIGN_OR_RETURN(m.engine, model->BuildFastEngine());

  // Index the input features of the model.
  m.features = &m.engine->features();

  // Index the input features.
{str_feature_index}

  return m;
}}

}}  // namespace exported_model_{key}
}}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_GENERATED_MODEL_{key}
"""
