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

r"""Minimal usage example of YDF.

This example trains, displays, evaluates and exports a Gradient Boosted Tree
model.

Usage example:

  pip install ydf pandas -U
  python minimal.py
"""

import sys
from absl import app
import pandas as pd
import ydf


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Download the Adult dataset, load in a Pandas dataframe.
  train_path = "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset/adult_train.csv"
  test_path = "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset/adult_train.csv"
  train_df = pd.read_csv(train_path)
  test_df = pd.read_csv(test_path)

  # Display full logs
  ydf.verbose(2)

  # Trains the model.
  model = ydf.GradientBoostedTreesLearner(label="income").train(train_df)

  # Some information about the model.
  print(model.describe())

  # Evaluates the model on the test dataset.
  evaluation = model.evaluate(test_df)
  print(evaluation)

  # Exports the model to disk.
  model.save("/tmp/ydf_model")

  # Reload the model from disk
  loaded_model = ydf.load_model("/tmp/ydf_model")

  # Make predictions with the model from disk.
  predictions = loaded_model.predict(test_df)
  print(predictions)

  # if not (sys.version_info < (3, 9)) and (sys.version_info < (3, 12)):
  #   # TensorFlow is not supported anymore for py3.8.
  #   # TensorFlow Decision Forests is not yet supported for py3.12.
  #   loaded_model.to_tensorflow_saved_model("/tmp/tf_saved_model")


if __name__ == "__main__":
  app.run(main)
