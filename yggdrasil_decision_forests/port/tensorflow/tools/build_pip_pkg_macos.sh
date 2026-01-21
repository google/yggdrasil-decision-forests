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



# Compile YDF_TF for macOS.
#
# Usage:
#   ./tools/build_pip_pkg_macos.sh
#
# For a local install location of ydf, provide YDF_LOCAL_DEPENDENCY_DIR.
# To build for TF < 2.20.0, use LEGACY=1.

set -e

function cleanup {
  echo "Cleaning up..."
  if [[ -f MODULE.bazel.bak ]]; then
    mv MODULE.bazel.bak MODULE.bazel
  fi
  if [[ -f ../../utils/registration.h.bak ]]; then
    mv ../../utils/registration.h.bak ../../utils/registration.h
  fi
  if [[ -f ../../utils/compile.bzl.bak ]]; then
    mv ../../utils/compile.bzl.bak ../../utils/compile.bzl
  fi
  if [[ -f ../../utils/BUILD.bak ]]; then
    mv ../../utils/BUILD.bak ../../utils/BUILD
  fi
  rm -rf test_env_build test_env test_run_dir
}
trap cleanup EXIT

# Backup original configuration files
cp MODULE.bazel MODULE.bazel.bak
cp ../../utils/registration.h ../../utils/registration.h.bak

# --- 2. Version Configuration ---
if [[ "$LEGACY" == "1" ]]; then
  echo "Running in LEGACY mode (TF < 2.20.0)"
  cp tools/MODULE.bazel.legacy MODULE.bazel
  # The old Abseil version has no NoDestructor, patch registration.h
  cp tools/registration.h.legacy ../../utils/registration.h
  sed -i '' 's|"@com_google_absl//absl/base:no_destructor",|# "@com_google_absl//absl/base:no_destructor",|g' ../../utils/BUILD 
  # Patching compile.bzl
  sed -i '' 's|load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")|load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")|g' ../../utils/compile.bzl
  sed -i '' 's|load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")|load("@rules_cc//cc:defs.bzl", "cc_proto_library")|g' ../../utils/compile.bzl

  # Note: TensorFlow <2.18.0 depends on Protobuf 4.25.3 or less, which is not
  # compatible with YDF versions using ydf-tf.
  # Technically, YDF users could still use the ydf-tf op manually, see
  # export_tf.py for details. However, this is neither tested nor supported.
  TF_VERSIONS=("2.19.1" "2.19.0" "2.18.1" "2.18.0")
  PY_VERSIONS=("3.9" "3.10" "3.11" "3.12")
else
  # Default mode uses the existing files (no cp needed, backups are already made)
  TF_VERSIONS=("2.20.0")
  PY_VERSIONS=("3.9" "3.10" "3.11" "3.12" "3.13")
fi

rm -rf pip_pkg/wheelhouse
mkdir -p pip_pkg/wheelhouse

for YDF_TF_VERSION in "${TF_VERSIONS[@]}"; do
  for YDF_PY_VERSION in "${PY_VERSIONS[@]}"; do
    echo "================================================="
    echo "Building for Python $YDF_PY_VERSION and TF $YDF_TF_VERSION"
    echo "================================================="

    # Install Python version via pyenv if missing
    pyenv install -s "$YDF_PY_VERSION"
    export PYENV_VERSION=$YDF_PY_VERSION

    # Create fresh build environment
    rm -rf test_env_build
    python -m venv test_env_build
    source test_env_build/bin/activate

    pip install --upgrade pip setuptools wheel build

    export YDF_PY_VERSION
    export YDF_TF_VERSION

    # Build C++
    bazel build -c opt \
      --action_env=YDF_PY_VERSION="$YDF_PY_VERSION" \
      --action_env=YDF_TF_VERSION="$YDF_TF_VERSION" \
      --define=tensorflow_with_header_lib=1 \
      //ydf_tf:inference.so

    rm -rf pip_pkg/ydf_tf
    mkdir -p pip_pkg/ydf_tf

    if [ -f bazel-bin/ydf_tf/inference.so ]; then
      cp bazel-bin/ydf_tf/inference.so pip_pkg/ydf_tf/
    elif [ -f bazel-bin/ydf_tf/inference.dylib ]; then
      cp bazel-bin/ydf_tf/inference.dylib pip_pkg/ydf_tf/inference.so
    else
      echo "Error: Shared library inference.so (or .dylib) not found in bazel-bin/ydf_tf/"
      exit 1
    fi

    cp ydf_tf/{__init__.py,api.py,op.py,op_dynamic.py} pip_pkg/ydf_tf/

    pushd pip_pkg
    rm -rf dist build *.egg-info
    python -m build --wheel
    popd

    # Clean up build venv
    deactivate
    rm -rf test_env_build

    echo "Testing wheel for Python $YDF_PY_VERSION..."
    rm -rf test_env
    python -m venv test_env
    source test_env/bin/activate

    PY_TAG="cp${YDF_PY_VERSION//.}"
    WHEEL_PATH=$(ls pip_pkg/dist/*$PY_TAG*.whl | head -n 1)

    if [ -z "$WHEEL_PATH" ]; then
      echo "Error: Could not find built wheel for $PY_TAG"
      exit 1
    fi

    pip install "$WHEEL_PATH"

    if [[ -n "${YDF_LOCAL_DEPENDENCY_DIR}" ]]; then
      echo "Installing YDF from ${YDF_LOCAL_DEPENDENCY_DIR}"
      pip install "ydf>=0.15.0" --no-index --find-links "${YDF_LOCAL_DEPENDENCY_DIR}"
    else
      echo "Installing YDF from PyPI"
      pip install "ydf>=0.15.0"
    fi

    rm -rf test_run_dir
    mkdir -p test_run_dir
    cp pip_pkg/test_pkg.py test_run_dir/

    pushd test_run_dir
    python test_pkg.py
    popd

    deactivate
    rm -rf test_env test_run_dir

    # Save the wheel
    cp "$WHEEL_PATH" pip_pkg/wheelhouse/
  done
done

echo "Success! Final wheels are in pip_pkg/wheelhouse/"