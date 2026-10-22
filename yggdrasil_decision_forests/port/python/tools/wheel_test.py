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

"""Tests the YDF wheel built by Bazel (see //tools:wheel).

The test does not depend on the YDF sources: The `ydf` package is imported from
the extracted wheel.
"""

import platform
import re
import subprocess
import sys
import zipfile

from absl import flags
from absl.testing import absltest
import numpy as np
from packaging import metadata as packaging_metadata

_WHEEL = flags.DEFINE_string("wheel", None, "Path to the wheel.", required=True)

# glibc version and architecture of a manylinux platform tag, e.g.
# "manylinux_2_27_x86_64".
_MANYLINUX_TAG = r"manylinux_(\d+)_(\d+)_(\w+)"

_PACKAGE_FILE_EXTENSIONS = (".py", ".so", ".js")


class WheelTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.wheel_path = _WHEEL.value
    with zipfile.ZipFile(cls.wheel_path) as z:
      cls.files = set(z.namelist())

  def test_content(self):
    expected = [
        "ydf/__init__.py",
        "ydf/version.py",
        "ydf/cc/ydf.so",
        "ydf/learner/specialized_learners.py",
        "ydf/model/tree/__init__.py",
        "ydf/model/tree/plotter.js",
        "yggdrasil_decision_forests/dataset/data_spec_pb2.py",
        "yggdrasil_decision_forests/model/abstract_model_pb2.py",
    ]
    for f in expected:
      self.assertIn(f, self.files)

    for f in self.files:
      top_level = f.split("/")[0]
      self.assertTrue(
          top_level in ("ydf", "yggdrasil_decision_forests")
          or top_level.endswith(".dist-info"),
          f"Unexpected file: {f}",
      )
      self.assertFalse(f.endswith("_test.py"), f"Unexpected test file: {f}")
      if not top_level.endswith(".dist-info"):
        self.assertTrue(
            f.endswith(_PACKAGE_FILE_EXTENSIONS), f"Unexpected file: {f}"
        )

  def test_metadata(self):
    dist_info = [f for f in self.files if f.endswith(".dist-info/METADATA")]
    self.assertLen(dist_info, 1)
    with zipfile.ZipFile(self.wheel_path) as z:
      raw_metadata = z.read(dist_info[0])
    # Raises an error if the metadata is not valid.
    meta = packaging_metadata.Metadata.from_email(raw_metadata, validate=True)
    self.assertIn(meta.name, ("ydf", "ydf-staging"))
    self.assertIsNotNone(meta.version)
    self.assertNotEmpty(meta.summary)
    self.assertIn("# YDF", meta.description)
    self.assertNotEmpty(meta.keywords)
    self.assertNotEmpty(meta.requires_dist)
    self.assertTrue(
        any(f.endswith(".dist-info/licenses/LICENSE") for f in self.files)
    )

  def test_import_and_train(self):
    wheel_dir = self.create_tempdir().full_path
    with zipfile.ZipFile(self.wheel_path) as z:
      z.extractall(wheel_dir)
    # The test does not depend on the YDF sources, so "ydf" can only be
    # imported from the extracted wheel.
    sys.path.insert(0, wheel_dir)
    import ydf  # pylint: disable=g-import-not-at-top

    self.assertStartsWith(ydf.__file__, wheel_dir)
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(200, 2))
    data = {"x1": x[:, 0], "x2": x[:, 1], "label": x[:, 0] + x[:, 1] > 1.0}
    model = ydf.GradientBoostedTreesLearner(label="label", num_trees=5).train(
        data
    )
    self.assertGreater(model.evaluate(data).accuracy, 0.8)

  def test_manylinux_compliance(self):
    if platform.system() != "Linux":
      self.skipTest("Linux only")

    # e.g. "ydf-0.16.1-cp312-cp312-manylinux_2_27_x86_64.whl".
    wheel_tag = re.search(_MANYLINUX_TAG + r"\.whl$", self.wheel_path)
    self.assertIsNotNone(wheel_tag, f"Not a manylinux wheel: {self.wheel_path}")

    output = subprocess.run(
        [sys.executable, "-m", "auditwheel", "show", self.wheel_path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    # e.g. '... is consistent with the following platform tag:
    # "manylinux_2_17_x86_64".' (auditwheel wraps the lines).
    audit_tag = re.search(r'platform\s+tag:\s*"' + _MANYLINUX_TAG, output)
    self.assertIsNotNone(audit_tag, output)
    self.assertEqual(audit_tag[3], wheel_tag[3], output)
    # The wheel must not require a more recent glibc than its tag claims.
    self.assertLessEqual(
        (int(audit_tag[1]), int(audit_tag[2])),
        (int(wheel_tag[1]), int(wheel_tag[2])),
        output,
    )


if __name__ == "__main__":
  absltest.main()
