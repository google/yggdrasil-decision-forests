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



# Can be run on the manylinux_2_28_x86_64 docker.
# docker run --rm -it -v ~/dev/scratch/ydf-tf/yggdrasil-decision-forests/:/work_dir -w /work_dir quay.io/pypa/manylinux_2_28_x86_64:latest /bin/bash

set -e

# Prepare build environment
mkdir /build_tools
cd /build_tools
curl -L -o bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64
chmod +x bazel
export PATH="$PATH:/build_tools"
cd /work_dir/yggdrasil_decision_forests/port/tensorflow/

# Prepare local Python runtime
export YDF_PY_VERSION=3.13
export YDF_TF_VERSION=2.20.0  # This won't work for versions < 2.20
python$YDF_PY_VERSION -m pip install auditwheel setuptools

# Build C++
bazel build -c opt --define=tensorflow_with_header_lib=1 //ydf_tf:inference.so

# Create package
rm -rf pip_pkg/ydf_tf
mkdir -p pip_pkg/ydf_tf
cp bazel-bin/ydf_tf/inference.so pip_pkg/ydf_tf
cp ydf_tf/{__init__.py,api.py,op.py,op_dynamic.py} pip_pkg/ydf_tf
cd pip_pkg
rm -rf dist build *.egg-info
python$YDF_PY_VERSION setup.py bdist_wheel

# Repair package
auditwheel repair dist/*.whl --plat manylinux_2_28_x86_64 -w wheelhouse --exclude libtensorflow_framework.so.2
echo "Success! Final wheel is in pip_pkg/wheelhouse/"
