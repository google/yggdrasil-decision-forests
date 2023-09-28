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

"""Converts a arff dataset into a csv dataset."""

import numpy as np
import pandas as pd
from scipy.io import arff

# ===================
#    Configuration
# ===================

# Input and output paths.
# Update accordingly
input_ds_path = "~/hp_sweep/dataset/hls4ml_HLF.arff"
output_train_ds_path = "~/hp_sweep/dataset/hls4ml_HLF_train.csv"
output_test_ds_path = "~/hp_sweep/dataset/hls4ml_HLF_test.csv"

# If positive_class is set, the dataset is converted to a binary classification
# dataset where "positive_class" defines the positive class.
# If positive_class is None, the dataset is left as a multi-class classification
# dataset.
positive_class = b"t"
# positive_class = None

# ==========
#    Code
# ==========

print("Load dataset")
data_arff = arff.loadarff(input_ds_path)
data_pd = pd.DataFrame(data_arff[0])
print(data_pd.head())

print("Class distribution: ", data_pd["class"].value_counts())

if positive_class is not None:
  data_pd["class"] = data_pd["class"] != b"t"
  print(
      "Class distribution after conversion: ", data_pd["class"].value_counts()
  )

# Split dataset into a training and testing set.
np.random.seed(12345)
test_indices = np.random.rand(len(data_pd)) < 0.20
test_ds_pd = data_pd[test_indices]
train_ds_pd = data_pd[~test_indices]

print("Save test dataset")
test_ds_pd.to_csv(output_test_ds_path, index=False)

print("Save train dataset")
train_ds_pd.to_csv(output_train_ds_path, index=False)
