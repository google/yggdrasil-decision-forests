#!/bin/bash
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


# Packaged already compiled (i.e. build) YDF into a pip package.
#
# Usage example:
#   # Generate the pip package with python3.9
#   ./tools/package_linux.sh python3.9

set -vex

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
ARCHITECTURE=$(uname -m)
function is_macos() {
  [[ "${PLATFORM}" == "darwin" ]]
}

# Temporary directory used to assemble the package.
SRCPK="$(pwd)/tmp_package"

# Pypi package version compatible with a given version of python.
# Example: Python3.8.2 => Package version: "38"
function python_to_package_version() {

  python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")'
}

# Installs dependency requirement for build the Pip package.
function install_dependencies() {
  python -m ensurepip -U || true
  python -m pip install pip -U
  python -m pip install setuptools -U
  python -m pip install build -U
  python -m pip install virtualenv -U
  python -m pip install auditwheel==6.0.0 --force-reinstall
}

function check_is_build() {
  # Check the correct location of the current directory.
  local cur_dir=${PWD##*/}
  if [ ! -d "bazel-bin" ] || [ $cur_dir != "python" ]; then
    echo "This script should be run from the root directory of the Python port of Yggdrasil Decision Forests (i.e. probably under port/python) of a compiled Bazel export (i.e. containing a bazel-bin directory)"
    exit 1
  fi
}

# Collects the library files into ${SRCPK}
function assemble_files() {
  check_is_build
  python tools/collect_pip_files.py
}

# Build a pip package.
function build_package() {

  pushd ${SRCPK}
  python -m build
  popd

  cp -R ${SRCPK}/dist .
}

# Tests a pip package.
function test_package() {
  PACKAGE="$1"

  PIP="python -m pip"

  if is_macos; then
    PACKAGEPATH="dist/ydf-*-cp${PACKAGE}-cp${PACKAGE}*-*.whl"
  else
    PACKAGEPATH="dist/ydf-*-cp${PACKAGE}-cp${PACKAGE}*.manylinux2014_${ARCHITECTURE}.whl"
  fi
  ${PIP} install ${PACKAGEPATH} --force-reinstall

  ${PIP} list
  ${PIP} show ydf -f

  # Run a small example (in different folder to avoid clashes)
  local current_folder=$(basename "$PWD")
  pushd ..
  ${PIP} install -r $current_folder/dev_requirements.txt
  ${PIP} install -r $current_folder/requirements.txt
  python $current_folder/examples/minimal.py
  popd

  if [ -d previous_package ]; then
    rm -r previous_package
  fi
  mkdir previous_package
  python -m pip download --no-deps -d previous_package ydf
  local old_file_size=`du -k "previous_package" | cut -f1`
  local new_file_size=`du -k $PACKAGEPATH | cut -f1`
  local scaled_old_file_size=$(($old_file_size * 12))
  local scaled_new_file_size=$(($new_file_size * 10))
  if [ "$scaled_new_file_size" -gt "$scaled_old_file_size" ]; then
    echo "New package is 20% larger than the previous one."
    echo "This may indicate an issue with the wheel, aborting."
    exit 1
  fi
  scaled_old_file_size=$(($old_file_size * 8))
  if [ "$scaled_new_file_size" -lt "$scaled_old_file_size" ]; then
    echo "New package is 20% smaller than the previous one."
    echo "This may indicate an issue with the wheel, aborting."
    exit 1
  fi
  rm -r previous_package
  echo "Testing $PACKAGEPATH successful"
}

# Builds and tests a pip package in a given version of python
function e2e_native() {
  PACKAGE=$(python_to_package_version python)

  install_dependencies
  build_package

  # Fix package.
  if is_macos; then
    PACKAGEPATH="dist/ydf-*-cp${PACKAGE}-cp${PACKAGE}*-*.whl"
  else
    PACKAGEPATH="dist/ydf-*-cp${PACKAGE}-cp${PACKAGE}*-linux_${ARCHITECTURE}.whl"
    python -m auditwheel repair --plat manylinux2014_${ARCHITECTURE} -w dist ${PACKAGEPATH}
  fi

  test_package ${PACKAGE}
}

assemble_files
e2e_native
