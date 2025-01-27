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

"""ydf.experimental.* API."""

# pylint: disable=unused-import,g-importing-member,g-import-not-at-top,g-bad-import-order,reimported,disable=attribute-error

import sys
import logging
import importlib.util

if importlib.util.find_spec("jax") is not None:
  from ydf.deep.mlp import MultiLayerPerceptronModel
  from ydf.deep.mlp import MultiLayerPerceptronLearner

  from ydf.deep.tabular_transformer import TabularTransformerModel
  from ydf.deep.tabular_transformer import TabularTransformerLearner
else:
  logging.debug(
      "jax package is not available. Jax based models are not available"
  )
  from ydf.deep import jax_fallback

  MultiLayerPerceptronModel = jax_fallback.JaxFallBack
  MultiLayerPerceptronLearner = jax_fallback.JaxFallBack
  TabularTransformerModel = jax_fallback.JaxFallBack
  TabularTransformerLearner = jax_fallback.JaxFallBack
