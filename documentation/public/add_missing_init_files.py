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

"""Adds the `__init__.py` files missing from the installed ydf package.

The ydf 0.17.0 wheel does not contain the `__init__.py` files of its
sub-packages. Without them, mkdocstrings (griffe) cannot load the `ydf` package.

TODO: Remove once the ydf wheel ships the `__init__.py` files.
"""

import pathlib

import ydf

for directory in pathlib.Path(ydf.__file__).parent.rglob("*"):
  if directory.is_dir() and directory.name != "__pycache__":
    (directory / "__init__.py").touch()
