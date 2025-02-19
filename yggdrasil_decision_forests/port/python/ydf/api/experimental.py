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

_has_experimental = False
_experimental_import_error = None

if sys.version_info < (3, 9):
  _experimental_import_error = "ydf.experimental requires Python>=3.9"
elif importlib.util.find_spec("jax") is not None:
  try:
    from ydf.deep.mlp import MultiLayerPerceptronModel
    from ydf.deep.mlp import MultiLayerPerceptronLearner

    from ydf.deep.tabular_transformer import TabularTransformerModel
    from ydf.deep.tabular_transformer import TabularTransformerLearner

    # TODO: Hook into ydf.load_model
    from ydf.deep.model_lib import load_model

    _has_experimental = True
  except Exception as e:
    _experimental_import_error = str(e)

if not _has_experimental:
  logging.debug(
      "ydf.experimental is not available: %s", _experimental_import_error
  )
  from ydf.deep import jax_fallback

  jax_fallback._jax_fallback_error = _experimental_import_error  # pylint:disable=protected-access

  MultiLayerPerceptronModel = jax_fallback.JaxFallBack
  MultiLayerPerceptronLearner = jax_fallback.JaxFallBack
  TabularTransformerModel = jax_fallback.JaxFallBack
  TabularTransformerLearner = jax_fallback.JaxFallBack
