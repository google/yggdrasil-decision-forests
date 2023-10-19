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

import numpy as np

from ydf.cc import ydf
from ydf.dataset import dataset
from ydf.model import generic_model


class DecisionForestModel(generic_model.GenericModel):
  """A generic decision forest model for prediction and inspection."""

  _model: ydf.DecisionForestCCModel

  def num_trees(self):
    """Returns the number of trees in the decision forest."""
    return self._model.num_trees()

  def predict_leaves(self, data: dataset.InputDataset) -> np.ndarray:
    """Gets the index of the active leaf in each tree.

    The active leaf is the leave that that receive the example during inference.

    The returned value "leaves[i,j]" is the index of the active leaf for the
    i-th example and the j-th tree. Leaves are indexed by depth first
    exploration with the negative child visited before the positive one.

    Args:
      data: Dataset.

    Returns:
      Index of the active leaf for each tree in the model.
    """

    ds = dataset.create_vertical_dataset(
        data, data_spec=self._model.data_spec()
    )
    return self._model.PredictLeaves(ds._dataset)  # pylint: disable=protected-access
