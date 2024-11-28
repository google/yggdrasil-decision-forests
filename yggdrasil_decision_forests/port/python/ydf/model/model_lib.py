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

import os

from absl import logging

from yggdrasil_decision_forests.dataset import data_spec_pb2
from ydf.cc import ydf as ydf_cc
from ydf.dataset import dataspec
from ydf.model import generic_model
from ydf.model.gradient_boosted_trees_model import gradient_boosted_trees_model
from ydf.model.isolation_forest_model import isolation_forest_model
from ydf.model.random_forest_model import random_forest_model
from ydf.utils import log


TF_SAVED_MODEL_FILE_NAME = "saved_model.pb"
TF_ASSETS_DIRECTORY_NAME = "assets"


def is_saved_model(directory: str) -> bool:
  """Return True if the given directory is a TensorFlow SavedModel."""
  return os.path.isfile(os.path.join(directory, TF_SAVED_MODEL_FILE_NAME))


def transform_tfdf_categorical_columns(
    data_spec: data_spec_pb2.DataSpecification,
) -> None:
  """Add a vocabulary to integerized columns in the dataspec.

  Tensorflow Decision Forests consumes categorical columns a pre-integerized
  columns shifted by one for compliance with the Keras API. This is not
  supported by PYDF since it makes it very hard to feed data without using
  data converters. This function therefore adds an artificial dictionary to
  these columns in the dataspec.

  Args:
    data_spec: The dataspec of a TF-DF model, will be modified.

  Raises:
    ValueError: If an integerized category has a vocabulary or if an integerized
      column has a negative number of unique values in its column spec.
  """
  fixed = False
  for column in data_spec.columns:
    if (
        column.type == data_spec_pb2.ColumnType.CATEGORICAL
        and column.categorical.is_already_integerized
    ):
      if column.categorical.items:
        raise ValueError("Integerized categories should not have a vocabulary.")
      if column.categorical.number_of_unique_values < 0:
        raise ValueError(
            "The number of unique values in a dataspec must not be negative"
        )
      column.categorical.is_already_integerized = False
      # TF-DF columns are shifted by 1.
      column.categorical.items[dataspec.YDF_OOD].index = 0
      for i in range(1, column.categorical.number_of_unique_values):
        column.categorical.items[str(i - 1)].index = i
      fixed = True
  if fixed:
    log.info(
        "The model was created by Tensorflow Decision Forests and it contains"
        " integerized columns. A dictionary has automatically been added to the"
        " model's dataspec."
    )


def load_model(
    directory: str,
    advanced_options: generic_model.ModelIOOptions = generic_model.ModelIOOptions(),
) -> generic_model.ModelType:
  """Load a YDF model from disk.

  Usage example:

  ```python
  import pandas as pd
  import ydf

  # Create a model
  dataset = pd.DataFrame({"feature": [0, 1], "label": [0, 1]})
  learner = ydf.RandomForestLearner(label="label")
  model = learner.train(dataset)

  # Save model
  model.save("/tmp/my_model")

  # Load model
  loaded_model = ydf.load_model("/tmp/my_model")

  # Make predictions
  model.predict(dataset)
  loaded_model.predict(dataset)
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
  if advanced_options.file_prefix is not None:
    logging.info(
        "Loading model with prefix %s from %s",
        directory,
        advanced_options.file_prefix,
    )
  else:
    logging.info("Loading model from %s", directory)
  model_is_tfdf = is_saved_model(directory)

  if model_is_tfdf:
    raise ValueError(
        f"The model in directory {directory} is a TensorFlow Decision Forests"
        " model. For loading such models, use"
        " ydf.from_tensorflow_decision_forests()."
    )
  cc_model: ydf_cc.GenericCCModel = ydf_cc.LoadModel(
      directory, advanced_options.file_prefix
  )
  return load_cc_model(cc_model)


def load_cc_model(cc_model: ydf_cc.GenericCCModel) -> generic_model.ModelType:
  """Convert a C++ model into the correct Python-wrapped model.

  Args:
    cc_model: Generic C++ model.

  Returns:
    Python-wrapped model
  """
  model_name = cc_model.name()
  if model_name == ydf_cc.RandomForestCCModel.kRegisteredName:
    return random_forest_model.RandomForestModel(cc_model)
  if model_name == ydf_cc.GradientBoostedTreesCCModel.kRegisteredName:
    return gradient_boosted_trees_model.GradientBoostedTreesModel(cc_model)
  if model_name == ydf_cc.IsolationForestCCModel.kRegisteredName:
    return isolation_forest_model.IsolationForestModel(cc_model)
  logging.info(
      "This model has type %s, which is not fully supported. Only generic model"
      " tasks (e.g. inference) are possible",
      model_name,
  )
  return generic_model.GenericCCModel(cc_model)


def from_tensorflow_decision_forests(directory: str) -> generic_model.ModelType:
  """Load a TensorFlow Decision Forests model from disk.

  Usage example:

  ```python
  import pandas as pd
  import ydf

  # Import TF-DF model
  loaded_model = ydf.from_tensorflow_decision_forests("/tmp/my_tfdf_model")

  # Make predictions
  dataset = pd.read_csv("my_dataset.csv")
  model.predict(dataset)

  # Show details about the model
  model.describe()
  ```

  The imported model creates the same predictions as the original TF-DF model.

  Only TensorFlow Decision Forests models containing a single Decision Forest
  and nothing else are supported. That is, combined neural network / decision
  forest models cannot be imported. Unfortunately, importing such models may
  succeed but result in incorrect predictions, so check for prediction equality
  after importing.

  Args:
    directory: Directory containing the TF-DF model.

  Returns:
    Model to use for inference, evaluation or inspection
  """
  model_is_tfdf = is_saved_model(directory)

  if not model_is_tfdf:
    raise ValueError(
        "The given model is not a TensorFlow Decision Forests since it is"
        " missing a `saved_model.pb` file. Make sure the given directory"
        f" {directory} refers to the full TensorFlow Decision Forest model (not"
        " just one of its subdirectories)."
    )
  directory = os.path.join(directory, TF_ASSETS_DIRECTORY_NAME)
  cc_model: ydf_cc.GenericCCModel = ydf_cc.LoadModel(
      directory, file_prefix=None
  )

  data_spec = cc_model.data_spec()
  transform_tfdf_categorical_columns(data_spec)
  cc_model.set_data_spec(data_spec)
  return load_cc_model(cc_model)


def deserialize_model(
    data: bytes,
) -> generic_model.ModelType:
  """Loads a serialized YDF model.

  Usage example:

  ```python
  import pandas as pd
  import ydf

  # Create a model
  dataset = pd.DataFrame({"feature": [0, 1], "label": [0, 1]})
  learner = ydf.RandomForestLearner(label="label")
  model = learner.train(dataset)

  # Serialize model
  # Note: serialized_model is a bytes.
  serialized_model = model.serialize()

  # Deserialize model
  deserialized_model = ydf.deserialize_model(serialized_model)

  # Make predictions
  model.predict(dataset)
  deserialized_model.predict(dataset)
  ```

  Args:
    data: Serialized model.

  Returns:
    Model to use for inference, evaluation or inspection
  """
  cc_model: ydf_cc.GenericCCModel = ydf_cc.DeserializeModel(data)
  return load_cc_model(cc_model)
