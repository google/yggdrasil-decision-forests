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

"""Tests the YDF wheel built by Bazel (see the `wheel` target in //:BUILD).

The test does not depend on the YDF sources: The `ydf` package is imported from
the extracted wheel.
"""

import os
import platform
import re
import subprocess
import sys
import tempfile
import zipfile

from absl import flags
from absl.testing import absltest


_WHEEL = flags.DEFINE_string("wheel", None, "Path to the wheel.", required=True)


def _wheel_platform_tag(wheel_path: str) -> str:
  # Wheel filename: {distribution}-{version}-{python}-{abi}-{platform}.whl
  return os.path.basename(wheel_path)[: -len(".whl")].split("-")[-1]


def _parse_manylinux_tag(tag: str):
  """Returns (glibc major, glibc minor, arch) of a manylinux tag, or None."""
  legacy = {
      "manylinux1": (2, 5),
      "manylinux2010": (2, 12),
      "manylinux2014": (2, 17),
  }
  m = re.fullmatch(r"(manylinux1|manylinux2010|manylinux2014)_(\w+)", tag)
  if m:
    return (*legacy[m.group(1)], m.group(2))
  m = re.fullmatch(r"manylinux_(\d+)_(\d+)_(\w+)", tag)
  if m:
    return (int(m.group(1)), int(m.group(2)), m.group(3))
  return None


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
        "ydf/cc/__init__.py",
        "ydf/cc/ydf.so",
        "ydf/learner/__init__.py",
        "ydf/learner/specialized_learners.py",
        "ydf/model/tree/plotter.js",
        "yggdrasil_decision_forests/__init__.py",
        "yggdrasil_decision_forests/dataset/__init__.py",
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
      self.assertNotIn("wrapper", f, f"Unexpected wrapper generator file: {f}")

  def test_metadata(self):
    dist_info = [f for f in self.files if f.endswith(".dist-info/METADATA")]
    self.assertLen(dist_info, 1)
    with zipfile.ZipFile(self.wheel_path) as z:
      metadata = z.read(dist_info[0]).decode()
    self.assertIn("Requires-Dist: protobuf", metadata)
    self.assertIn("Description-Content-Type: text/markdown", metadata)
    self.assertTrue(
        any(f.endswith(".dist-info/licenses/LICENSE") for f in self.files)
    )

  def test_import_and_train(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      with zipfile.ZipFile(self.wheel_path) as z:
        z.extractall(tmp_dir)

      # Run in a sub-process to make sure the "ydf" package of the wheel is
      # used.
      script = """
import sys
import numpy as np
import ydf

assert ydf.__file__.startswith(sys.argv[1]), ydf.__file__
rng = np.random.default_rng(1)
x = rng.uniform(size=(200, 2))
data = {"x1": x[:, 0], "x2": x[:, 1], "label": x[:, 0] + x[:, 1] > 1.0}
model = ydf.GradientBoostedTreesLearner(label="label", num_trees=5).train(data)
accuracy = model.evaluate(data).accuracy
assert accuracy > 0.8, accuracy
print("ydf", ydf.__version__, "accuracy", accuracy)
"""
      env = dict(os.environ)
      env["PYTHONPATH"] = os.pathsep.join(
          [tmp_dir] + [p for p in sys.path if p]
      )
      subprocess.run(
          [sys.executable, "-c", script, tmp_dir],
          check=True,
          cwd=tmp_dir,
          env=env,
      )

  def test_manylinux_compliance(self):
    if platform.system() != "Linux":
      self.skipTest("Linux only")

    wheel_tag = _parse_manylinux_tag(_wheel_platform_tag(self.wheel_path))
    self.assertIsNotNone(wheel_tag, f"Not a manylinux wheel: {self.wheel_path}")

    result = subprocess.run(
        [sys.executable, "-m", "auditwheel", "show", self.wheel_path],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    # e.g. "... is consistent with the following platform tag:
    # "manylinux_2_27_x86_64"."
    m = re.search(r'following platform tag:\s*"([^"]+)"', output)
    self.assertIsNotNone(m, output)
    audit_tag = _parse_manylinux_tag(m.group(1))
    self.assertIsNotNone(
        audit_tag, f"The wheel is not manylinux compatible:\n{output}"
    )
    self.assertEqual(audit_tag[2], wheel_tag[2], output)
    # The wheel must not require a more recent glibc than its tag claims.
    self.assertLessEqual(audit_tag[:2], wheel_tag[:2], output)


if __name__ == "__main__":
  absltest.main()
