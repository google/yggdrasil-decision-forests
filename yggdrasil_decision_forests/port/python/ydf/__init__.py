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

"""(P)YDF - Yggdrasil Decision Forests in Python."""


def _check_install():
  from yggdrasil_decision_forests.dataset import data_spec_pb2

  if not hasattr(data_spec_pb2, "DType"):
    raise ValueError("""\
Collision between YDF and TensorFlow Decision Forests protobuf shared dependencies.
Please, reinstall YDF with the "--force" argument, restart the notebook runtime (if using a notebook), and try again:

!pip install ydf --force""")


_check_install()


# pylint: disable=g-importing-member,g-import-not-at-top,g-bad-import-order,reimported

# Version
from ydf.version import version as __version__

# Note: Keep this file organized in the same order as:
# public/docs/py_api/index.md

# Learner
from ydf.learner.generic_learner import GenericLearner
from ydf.learner.specialized_learners import CartLearner
from ydf.learner.specialized_learners import RandomForestLearner
from ydf.learner.specialized_learners import GradientBoostedTreesLearner
from ydf.learner.specialized_learners import DistributedGradientBoostedTreesLearner
from ydf.learner.specialized_learners import IsolationForestLearner

DecisionTreeLearner = CartLearner

# Model
from ydf.model.generic_model import GenericModel
from ydf.model.random_forest_model.random_forest_model import RandomForestModel
from ydf.model.gradient_boosted_trees_model.gradient_boosted_trees_model import GradientBoostedTreesModel
from ydf.model.isolation_forest_model.isolation_forest_model import IsolationForestModel
# A CART model is a Random Forest with a single tree
CARTModel = RandomForestModel

# Tuner
from ydf.learner.tuner import RandomSearchTuner
from ydf.learner.tuner import VizierTuner

# Utilities
from ydf.utils.log import verbose
from ydf.model.model_lib import load_model
from ydf.model.model_lib import deserialize_model

# A feature is a column used as input of a model. In practice, users generally
# use them interchangeably.
from ydf.dataset.dataspec import Column as Feature
from ydf.dataset.dataspec import Column
from ydf.model.generic_model import Task
from ydf.dataset.dataspec import Semantic
from ydf.learner.worker import start_worker
from ydf.utils.log import strict

# Advanced Utilities
from ydf.model.generic_model import ModelIOOptions
from ydf.dataset.dataset import create_vertical_dataset
from ydf.model.model_metadata import ModelMetadata
from ydf.model.model_lib import from_tensorflow_decision_forests
from ydf.model.generic_model import from_sklearn
from ydf.model.generic_model import NodeFormat

# Custom Loss
from ydf.learner.custom_loss import RegressionLoss
from ydf.learner.custom_loss import BinaryClassificationLoss
from ydf.learner.custom_loss import MultiClassificationLoss
from ydf.learner.custom_loss import Activation


# Tree
from ydf.model import tree

# pylint: enable=g-importing-member,g-import-not-at-top,g-bad-import-order,reimported
