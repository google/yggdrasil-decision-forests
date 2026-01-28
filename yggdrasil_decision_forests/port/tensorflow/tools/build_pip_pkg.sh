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



# Compile YDF_TF for manylinux.
# Can be run on the manylinux_2_28_x86_64 docker.
# docker run --rm -it -v ~/dev/scratch/ydf-tf/yggdrasil-decision-forests/:/work_dir -w /work_dir quay.io/pypa/manylinux_2_28_x86_64:latest /bin/bash
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
  rm -rf test_env test_run_dir
}
trap cleanup EXIT

# Prepare build environment
mkdir -p /build_tools
cd /build_tools
if [ ! -f bazel ]; then
  curl -L -o bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64
  echo "e1508323f347ad1465a887bc5d2bfb91cffc232d11e8e997b623227c6b32fb76  bazel" | sha256sum --check
  chmod +x bazel
fi
export PATH="$PATH:/build_tools"

cd /work_dir/yggdrasil_decision_forests/port/tensorflow/

cp MODULE.bazel MODULE.bazel.bak
cp ../../utils/registration.h ../../utils/registration.h.bak

rm -rf pip_pkg/wheelhouse

if [[ "$LEGACY" == "1" ]]; then
  echo "Running in LEGACY mode (TF < 2.20.0)"
  # Prepare dependencies
  cp tools/MODULE.bazel.legacy MODULE.bazel
  # The old Abseil version has no NoDestructor, patch registration.h
  cp tools/registration.h.legacy ../../utils/registration.h
  sed -i 's|"@com_google_absl//absl/base:no_destructor",|# "@com_google_absl//absl/base:no_destructor",|g' ../../utils/BUILD
  # Patching compile.bzl
  sed -i 's|load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")|load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")|g' ../../utils/compile.bzl
  sed -i 's|load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")|load("@rules_cc//cc:defs.bzl", "cc_proto_library")|g' ../../utils/compile.bzl
  
  # Note: TensorFlow <2.18.0 depends on Protobuf 4.25.3 or less, which is not
  # compatible with YDF versions using ydf-tf.
  # Technically, YDF users could still use the ydf-tf op manually, see
  # export_tf.py for details. However, this is neither tested nor supported.
  TF_VERSIONS=("2.19.1" "2.19.0" "2.18.1" "2.18.0")
  PY_VERSIONS=("3.9" "3.10" "3.11" "3.12")
else
  TF_VERSIONS=("2.20.0")
  PY_VERSIONS=("3.9" "3.10" "3.11" "3.12" "3.13")
fi

for YDF_PY_VERSION in "${PY_VERSIONS[@]}"; do
  echo "Installing build dependencies for Python $YDF_PY_VERSION..."
  python$YDF_PY_VERSION -m pip install auditwheel setuptools build --quiet
done

for YDF_TF_VERSION in "${TF_VERSIONS[@]}"; do
  for YDF_PY_VERSION in "${PY_VERSIONS[@]}"; do
    echo "================================================="
    echo "Building for Python $YDF_PY_VERSION and TF $YDF_TF_VERSION"
    echo "================================================="

    # Set Tensorflow & Python version for TF Header download
    export YDF_TF_VERSION
    export YDF_PY_VERSION

    # Build C++
    # Added --action_env to ensure Bazel respects the changing env vars
    bazel build -c opt \
      --action_env=YDF_PY_VERSION="$YDF_PY_VERSION" \
      --action_env=YDF_TF_VERSION="$YDF_TF_VERSION" \
      --define=tensorflow_with_header_lib=1 \
      --copt=-mavx2 --copt=-mavx \
      //ydf_tf:inference.so

    # Create package structure
    rm -rf pip_pkg/ydf_tf
    mkdir -p pip_pkg/ydf_tf

    if [ ! -f bazel-bin/ydf_tf/inference.so ]; then
        echo "Error: bazel-bin/ydf_tf/inference.so not found!"
        exit 1
    fi

    cp bazel-bin/ydf_tf/inference.so pip_pkg/ydf_tf
    cp ydf_tf/{__init__.py,api.py,op.py,op_dynamic.py} pip_pkg/ydf_tf

    pushd pip_pkg
    rm -rf dist build *.egg-info
    python$YDF_PY_VERSION -m build --wheel

    # Repair package (Auditwheel)
    # Using 'head' to safely grab the generated wheel name
    RAW_WHEEL=$(ls dist/*.whl | head -n 1)
    if [ -z "$RAW_WHEEL" ]; then
        echo "Error: No wheel found in pip_pkg/dist"
        exit 1
    fi

    auditwheel repair "$RAW_WHEEL" --plat manylinux_2_28_x86_64 -w wheelhouse --exclude libtensorflow_framework.so.2
    popd

    echo "Testing wheel for Python $YDF_PY_VERSION..."

    # Clean up previous venv
    rm -rf test_env
    python$YDF_PY_VERSION -m venv test_env
    source test_env/bin/activate

    # Identify the repaired wheel in the wheelhouse
    PY_TAG="cp${YDF_PY_VERSION//.}"
    # Look for the wheel with the specific TF version and Python tag we just built
    REPAIRED_WHEEL=$(ls pip_pkg/wheelhouse/*$YDF_TF_VERSION*$PY_TAG*.whl | head -n 1)

    if [ -z "$REPAIRED_WHEEL" ]; then
      echo "Error: Could not find repaired wheel for $PY_TAG in pip_pkg/wheelhouse/"
      exit 1
    fi

    pip install "$REPAIRED_WHEEL"

    # Install YDF Dependency
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
  done
done

echo "Success! Final wheels are in pip_pkg/wheelhouse/"