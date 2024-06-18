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

import sys
from typing import Iterator, Optional, Sequence

import numpy as np

from ydf.cc import ydf
from ydf.dataset import dataset
from ydf.model import generic_model
from ydf.model.tree import plot as plot_lib
from ydf.model.tree import tree as tree_lib


class DecisionForestModel(generic_model.GenericModel):
  """A generic decision forest model for prediction and inspection."""

  _model: ydf.DecisionForestCCModel

  def num_trees(self):
    """Returns the number of trees in the decision forest."""
    return self._model.num_trees()

  def get_tree(self, tree_idx: int) -> tree_lib.Tree:
    """Gets a single tree of the model.

    Args:
      tree_idx: Index of the tree. Should be in [0, num_trees()).

    Returns:
      The tree.
    """
    nodes = self._model.GetTree(tree_idx)
    return tree_lib.proto_nodes_to_tree(nodes, self.data_spec())

  def get_all_trees(self) -> Sequence[tree_lib.Tree]:
    """Returns all the trees in the model."""

    return list(self.iter_trees())

  def iter_trees(self) -> Iterator[tree_lib.Tree]:
    """Returns an iterator over all the trees in the model."""

    return (self.get_tree(tree_idx) for tree_idx in range(self.num_trees()))

  def print_tree(
      self, tree_idx: int = 0, max_depth: Optional[int] = 6, file=sys.stdout
  ) -> None:
    """Prints a tree in the terminal.

    Usage example:

    ```python
    # Create a dataset
    train_ds = pd.DataFrame({
        "c1": [1.0, 1.1, 2.0, 3.5, 4.2] + list(range(10)),
        "label": ["a", "b", "b", "a", "a"] * 3,
    })
    # Train a CART model
    model = ydf.CartLearner(label="label").train(train_ds)
    # Make sure the model is a CART
    assert isinstance(model, ydf.CARTModel)
    # Print the tree
    model.print_tree()
    ```

    Args:
      tree_idx: Index of the tree. Should be in [0, self.num_trees()).
      max_depth: Maximum tree depth of the plot. Set to None for full depth.
      file: Where to print the tree. By default, prints on the terminal standard
        output.
    """

    file.write(
        self.get_tree(tree_idx).pretty(self.data_spec(), max_depth=max_depth)
    )

  def plot_tree(
      self,
      tree_idx: int = 0,
      max_depth: Optional[int] = None,
      options: Optional[plot_lib.PlotOptions] = None,
      d3js_url: str = "https://d3js.org/d3.v6.min.js",
  ) -> plot_lib.TreePlot:
    """Plots an interactive HTML rendering of the tree.

    Usage example:

    ```python
    # Create a dataset
    train_ds = pd.DataFrame({
        "c1": [1.0, 1.1, 2.0, 3.5, 4.2] + list(range(10)),
        "label": ["a", "b", "b", "a", "a"] * 3,
    })
    # Train a CART model
    model = ydf.CartLearner(label="label").train(train_ds)
    # Make sure the model is a CART
    assert isinstance(model, ydf.CARTModel)
    # Plot the tree in Colab
    model.plot_tree()
    ```

    Args:
      tree_idx: Index of the tree. Should be in [0, self.num_trees()).
      max_depth: Maximum tree depth of the plot. Set to None for full depth.
      options: Advanced options for plotting. Set to None for default style.
      d3js_url: URL to load the d3.js library from.

    Returns:
      In interactive environments, an interactive plot. The HTML source can also
      be exported to file.
    """
    label_classes = None
    if self.task() == generic_model.Task.CLASSIFICATION:
      label_classes = self.label_classes()

    return self.get_tree(tree_idx).plot(
        self.data_spec(), max_depth, label_classes, options, d3js_url
    )

  def set_tree(self, tree_idx: int, tree: tree_lib.Tree) -> None:
    """Overrides a single tree of the model.

    Args:
      tree_idx: Index of the tree. Should be in [0, num_trees()).
      tree: New tree.
    """
    proto_nodes = tree_lib.tree_to_proto_nodes(tree, self.data_spec())
    self._model.SetTree(tree_idx, proto_nodes)

  def add_tree(self, tree: tree_lib.Tree) -> None:
    """Adds a single tree of the model.

    Args:
      tree: New tree.
    """
    proto_nodes = tree_lib.tree_to_proto_nodes(tree, self.data_spec())
    self._model.AddTree(proto_nodes)

  def remove_tree(self, tree_idx: int) -> None:
    """Removes a single tree of the model.

    Args:
      tree_idx: Index of the tree. Should be in [0, num_trees()).
    """
    self._model.RemoveTree(tree_idx)

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

  def distance(
      self,
      data1: dataset.InputDataset,
      data2: Optional[dataset.InputDataset] = None,
  ) -> np.ndarray:
    """Computes the pairwise distance between examples in "data1" and "data2".

    If "data2" is not provided, computes the pairwise distance between examples
    in "data1".

    Usage example:

    ```python
    import pandas as pd
    import ydf

    # Train model
    train_ds = pd.read_csv("train.csv")
    model = ydf.RandomForestLearner(label="label").Train(train_ds)

    test_ds = pd.read_csv("test.csv")
    distances = model.distance(test_ds, train_ds)
    # "distances[i,j]" is the distance between the i-th test example and the
    # j-th train example.
    ```

    Different models are free to implement different distances with different
    definitions. For this reasons, unless indicated by the model, distances
    from different models cannot be compared.

    The distance is not guaranteed to satisfy the triangular inequality
    property of metric distances.

    Not all models can compute distances. In this case, this function will raise
    an Exception.

    Args:
      data1: Dataset. Can be a dictionary of list or numpy array of values,
        Pandas DataFrame, or a VerticalDataset.
      data2: Dataset. Can be a dictionary of list or numpy array of values,
        Pandas DataFrame, or a VerticalDataset.

    Returns:
      Pairwise distance
    """

    ds1 = dataset.create_vertical_dataset(
        data1, data_spec=self._model.data_spec()
    )
    if data2 is None:
      ds2 = ds1
    else:
      ds2 = dataset.create_vertical_dataset(
          data2, data_spec=self._model.data_spec()
      )
    return self._model.Distance(ds1._dataset, ds2._dataset)  # pylint: disable=protected-access

  def set_node_format(self, node_format: generic_model.NodeFormat) -> None:
    """Set the serialization format for the nodes.

    Args:
      node_format: Node format to use when saving the model.
    """
    self._model.set_node_format(node_format.name)
