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

"""Utilities for loading YDF models."""

from absl import logging

from ydf.cc import ydf
from ydf.model import generic_model
from ydf.model import gradient_boosted_trees_model
from ydf.model import random_forest_model


def load_model(
    directory: str,
    advanced_options: generic_model.ModelIOOptions = generic_model.ModelIOOptions(),
) -> generic_model.ModelType:
  """Load a YDF model from disk.

  Usage example:

  ```python
  import pandas as pd
  import ydf

  model = ydf.load_model("/path/to/my/model/")

  df = pd.read_csv("my_dataset.csv")
  # Predict on the dataset
  model.predict(df)
  ```

  If a directory contains multiple YDF models, the models are uniquely
  identified by their prefix. The prefix to use can be specified in the advanced
  options. If the directory only contains a single model, the correct prefix is
  detected automatically.

  Args:
    directory: Directory containing the model.
    advanced_options: Advanced options for model loading.

  Returns:
    Model to use for inference, evaluation or inspection
  """
  logging.info("Loading model from %s", directory)
  # TODO: Add support for model prefixes
  cc_model: ydf.GenericCCModel = ydf.LoadModel(
      directory, advanced_options.file_prefix
  )
  return load_cc_model(cc_model)


def load_cc_model(cc_model: ydf.GenericCCModel) -> generic_model.ModelType:
  """Converts a C++ model into the correct Python-wrapped model.

  Args:
    cc_model: Generic C++ model.

  Returns:
    Python-wrapped model
  """
  model_name = cc_model.name()
  if model_name == ydf.RandomForestCCModel.kRegisteredName:
    return random_forest_model.RandomForestModel(cc_model)
  if model_name == ydf.GradientBoostedTreesCCModel.kRegisteredName:
    return gradient_boosted_trees_model.GradientBoostedTreesModel(cc_model)
  logging.info(
      "This model has type %s, which is not fully supported. Only generic model"
      " tasks (e.g. inference) are possible",
      model_name,
  )
  return generic_model.GenericModel(cc_model)
