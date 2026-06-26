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

from absl.testing import absltest

from ydf.utils import pybind_test_helper


class PybindTest(absltest.TestCase):

  def test_safe_python_call_success(self):
    pybind_test_helper.call_safe_python_call_status(lambda: None)

  def test_safe_python_call_success_status_or(self):
    val = pybind_test_helper.call_safe_python_call_status_or(lambda: 42)
    self.assertEqual(val, 42)

  def test_safe_python_call_py_exception(self):
    def bad_func():
      raise ValueError("Py error")

    with self.assertRaisesRegex(
        RuntimeError,
        "Python function 'test_func' raised: ValueError: Py error",
    ):
      pybind_test_helper.call_safe_python_call_status(bad_func)

  def test_safe_python_call_throw_std(self):
    with self.assertRaisesRegex(
        RuntimeError, "Python function 'test_throw_std' raised: C\\+\\+ error"
    ):
      pybind_test_helper.call_safe_python_call_throw_std()

  def test_safe_python_call_throw_unknown(self):
    with self.assertRaisesRegex(
        RuntimeError,
        "Python function 'test_throw_unknown' raised an unknown C\\+\\+ "
        "exception",
    ):
      pybind_test_helper.call_safe_python_call_throw_unknown()

  def test_make_safe_gil_holder(self):
    called = False

    def cb():
      nonlocal called
      called = True

    notifier = pybind_test_helper.GilCheckNotifier(cb)
    notifier.destroy()
    self.assertFalse(called)


if __name__ == "__main__":
  absltest.main()
