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

import os
from setuptools import Distribution, setup


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True


tf_version = os.environ.get("YDF_TF_VERSION")

if tf_version is None:
  raise ValueError("No Value given in YDF_TF_VERSION")

setup(
    name="ydf_tf",
    author="Mathieu Guillame-Bert, Richard Stotz",
    author_email="decision-forests-contact@google.com",
    # YDF-TF versions are synced to the respective TF versions.
    version=tf_version,
    description="The custom op to export and serve YDF models in TensorFlow.",
    long_description="""# YDF for TensorFlow

This package contains a custom op for TensorFlow to perform inference on Yggdrasil Decision Forests (YDF) models.

*   This package needs to be installed so that the export from YDF to TensorFlow works.
*   This package needs to be installed to read YDF models with TensorFlow.
*   This package replaces the inference op formerly provided by TensorFlow Decision Forests (TF-DF).

See the [YDF documentation](https://ydf.readthedocs.io) for more information.""",
    long_description_content_type="text/markdown",
    url="https://github.com/google/yggdrasil-decision-forests",
    project_urls={
        "Bug Tracker": (
            "https://github.com/google/yggdrasil-decision-forests/issues"
        ),
    },
    packages=["ydf_tf"],
    package_data={
        "ydf_tf": ["*.so"],
    },
    include_package_data=True,
    distclass=BinaryDistribution,
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=[
        f"tensorflow=={tf_version}",
        "ydf",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
