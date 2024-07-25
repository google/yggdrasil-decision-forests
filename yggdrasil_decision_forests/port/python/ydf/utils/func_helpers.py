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

"""Helper tools for manipulating Python functions."""

import functools
import inspect


def list_explicit_arguments(func):
  """Decorator that populates `explicit_args` with non-default args.

  This decorator allows to distinguish between an argument explicitly given to
  the function and others that only carry the default value.

  Args:
    func: A function whose `explicit_args` argument is populated.

  Returns:
    The decorated function.
  """
  arguments = inspect.getfullargspec(func)[0]

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    explicit_args = set(list(arguments[: len(args)]) + list(kwargs.keys()))
    if "explicit_args" in explicit_args:
      raise ValueError("`explicit_args` is for internal use only")
    kwargs["explicit_args"] = explicit_args
    return func(*args, **kwargs)

  return wrapper
