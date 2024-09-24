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

"""Setup file for pip's build.

This file is used by tools/build_pip_package.sh.
"""
import platform
import sys
import setuptools
from setuptools.command.install import install
from setuptools.dist import Distribution

_VERSION = "0.8.0"

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

REQUIRED_PACKAGES = [
    "numpy",
    "absl_py",
    "protobuf>=3.14",
]

OPTIONAL_PACKAGES = {"pandas": ["pandas"]}

MAC_CROSS_COMPILED = False  # Change if cross-compiled


class InstallPlatlib(install):

  def finalize_options(self):
    install.finalize_options(self)
    if self.distribution.has_ext_modules():
      self.install_lib = self.install_platlib


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True

  def is_pure(self):
    return False


if "bdist_wheel" in sys.argv:
  if "--plat-name" not in sys.argv:
    if platform.system() == "Darwin":
      if MAC_CROSS_COMPILED:
        idx = sys.argv.index("bdist_wheel") + 1
        sys.argv.insert(idx, "--plat-name")
        if platform.processor() == "arm":
          sys.argv.insert(idx + 1, "macosx_10_15_x86_64")
        elif platform.processor() == "i386":
          sys.argv.insert(idx + 1, "macosx_12_0_arm64")
        else:
          raise ValueError(f"Unknown processor {platform.processor()}")
      else:
        idx = sys.argv.index("bdist_wheel") + 1
        sys.argv.insert(idx, "--plat-name")
        if platform.processor() == "arm":
          sys.argv.insert(idx + 1, "macosx_12_0_arm64")
        elif platform.processor() == "i386":
          sys.argv.insert(idx + 1, "macosx_10_15_x86_64")
        else:
          raise ValueError(f"Unknown processor {platform.processor()}")
    else:
      print("Not on MacOS")
  else:
    print("--plat-name supplied")
else:
  print("Not using bdist_wheel")

setuptools.setup(
    cmdclass={
        "install": InstallPlatlib,
    },
    name="ydf",
    version=_VERSION,
    author="Mathieu Guillame-Bert, Richard Stotz, Jan Pfeifer",
    author_email="decision-forests-contact@google.com",
    description=(
        "YDF (short for Yggdrasil Decision Forests) is a library for training,"
        " serving, evaluating and analyzing decision forest models such as"
        " Random Forest and Gradient Boosted Trees."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google/yggdrasil-decision-forests",
    project_urls={
        "Documentation": "https://ydf.readthedocs.io/",
        "Source": "https://github.com/google/yggdrasil-decision-forests.git",
        "Tracker": (
            "https://github.com/google/yggdrasil-decision-forests/issues"
        ),
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    distclass=BinaryDistribution,
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    license="Apache 2.0",
    keywords=(
        "machine learning decision forests random forest gradient boosted"
        " decision trees classification regression ranking uplift"
    ),
    install_requires=REQUIRED_PACKAGES,
    extras_require=OPTIONAL_PACKAGES,
    include_package_data=True,
    zip_safe=False,
)
