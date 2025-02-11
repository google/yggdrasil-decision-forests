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

"""Empty fallback learner and model classes when jax is not available."""


_jax_fallback_error = None


def raise_error():
  if _jax_fallback_error is not None:
    detailed_error = f"The detailed error was: {_jax_fallback_error}"
  else:
    detailed_error = ""
  raise ValueError(
      '"jax" is needed to run this model. Make sure it installed and try'
      " again. See https://jax.readthedocs.io/en/latest/installation.html. The detailed error was: " + detailed_error
  )


class JaxFallBack:

  def __getattr__(self, name):
    def func(*args, **kwargs):
      raise_error()

    return func

  def __init__(self, *args, **kwargs):
    del args
    del kwargs
    raise_error()
