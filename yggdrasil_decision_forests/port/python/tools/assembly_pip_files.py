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

"""Assembles the files to create a pip package from Bazel's output."""

import glob
import os
from pathlib import Path
import platform
import shutil as s


SRC_BIN = "bazel-bin/ydf"
DST_PK = "tmp_package"

if platform.system() == "Windows":
  DST_EXTENSION = "pyd"
else:
  DST_EXTENSION = "so"


def rec_glob_copy(src_dir: str, dst_dir: str, pattern: str):
  """Copies the files matching a pattern from the src to the dst directory."""

  # TODO: Use "root_dir=src_dir" argument when >=python3.10
  os.makedirs(dst_dir, exist_ok=True)
  for fall in glob.glob(f"{src_dir}/{pattern}", recursive=True):
    frel = os.path.relpath(fall, src_dir)
    dst = f"{dst_dir}/{frel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    s.copy(f"{src_dir}/{frel}", dst)


def replace_in_files(src_dir, extension, old_string, new_string):
  """Replaces a string in all files with a given extension within a directory."""

  for root, _, filenames in os.walk(src_dir):
    for filename in filenames:
      if filename.endswith(extension):
        filepath = os.path.join(root, filename)

        # Read file content
        with open(filepath, "r") as f:
          file_content = f.read()

        # Replace the string
        new_content = file_content.replace(old_string, new_string)

        # Overwrite the file with the modified content
        with open(filepath, "w") as f:
          f.write(new_content)


# Remove and recreate the package directory
if os.path.exists(DST_PK):
  try:
    s.rmtree(DST_PK)
  except Exception:
    print(
        "Fail to remove the existing dir with rmtree. Use rmdir instead (only"
        " for Windows)."
    )
    os.system(f"rmdir /S /Q {DST_PK}")
os.makedirs(DST_PK)

# Individual files
for f in [
    "config/setup.py",
    "config/MANIFEST.in",
    "README.md",
    "CHANGELOG.md",
    "bazel-python/external/ydf_cc/LICENSE",
]:
  s.copy(f, DST_PK)

# YDF compiled lib
os.makedirs(f"{DST_PK}/ydf/cc")
s.copy(f"{SRC_BIN}/cc/ydf.so", f"{DST_PK}/ydf/cc/ydf.{DST_EXTENSION}")

# Generated learner python code
os.makedirs(f"{DST_PK}/ydf/learner")
s.copy(f"{SRC_BIN}/learner/specialized_learners.py", f"{DST_PK}/ydf/learner")

# Copy the YDF c++ protos
rec_glob_copy(
    "bazel-bin/external/ydf_cc/yggdrasil_decision_forests",
    f"{DST_PK}/ydf/proto",
    "**/*.py",
)

# The PYDF source files
rec_glob_copy("ydf", f"{DST_PK}/ydf", "**/*.py")

# Create the missing __init__.py files
INIT_FILENAME = "__init__.py"
for path, _, files in os.walk(f"{DST_PK}/ydf/proto"):
  if INIT_FILENAME not in files:
    Path(f"{path}/{INIT_FILENAME}").touch()

# Change path to YDF proto files
replace_in_files(
    DST_PK, ".py", "from yggdrasil_decision_forests.", "from ydf.proto."
)
