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

"""Definitions for generic decision forest models."""

from ydf.cc import ydf
from ydf.model import generic_model


class DecisionForestModel(generic_model.GenericModel):
  """A generic decision forest model for prediction and inspection."""
  _model: ydf.DecisionForestCCModel

  def num_trees(self):
    """Returns the number of trees in the decision forest."""
    return self._model.num_trees()
