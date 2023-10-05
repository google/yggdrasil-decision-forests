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

# WARNING: The API (TODO) reference documentation reads this file and expects a
# single import per line. Do not import several symbols from the same module in
# a single line.

# TIP: If you need to import something here that isn't part of the public API,
# and therefore shouldn't show up in the documentation, import it with a private
# name:
# from yggdrasil_decision_forests.port.python import submodule as _submodule

# Core
from ydf.dataset import dataset as _dataset
from ydf.learner import generic_learner as _generic_learner
from ydf.learner import specialized_learners as _specialized_learners
from ydf.model import generic_model as _generic_model
from ydf.model import model_lib as _model_lib

__version__ = "0.0.2"

# Dataset
create_vertical_dataset = _dataset.create_vertical_dataset
Column = _dataset.Column
# A feature is a column used as input of a model. In practice, users generally
# use them interchangeably.
Feature = Column
Task = _generic_learner.Task

# Model
load_model = _model_lib.load_model
ModelIOOptions = _generic_model.ModelIOOptions

# Learner
CartLearner = _specialized_learners.CartLearner
RandomForestLearner = _specialized_learners.RandomForestLearner
GradientBoostedTreesLearner = _specialized_learners.GradientBoostedTreesLearner
