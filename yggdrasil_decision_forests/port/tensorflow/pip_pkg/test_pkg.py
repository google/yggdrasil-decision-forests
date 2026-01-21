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

"""Smoke test for the ydf_tf package. Requires ydf >= 0.15.0."""

import tempfile
import numpy as np
import ydf

dataset = {
    "f1": np.array([1, 2, 3, 4] * 100),
    "f2": np.array(["red", "blue", "blue", "red"] * 100),
    "label": np.array(["X", "Y", "X", "X"] * 100),
}

model = ydf.GradientBoostedTreesLearner(label="label").train(dataset)
model.to_tensorflow_saved_model(tempfile.gettempdir(), mode="tf")
