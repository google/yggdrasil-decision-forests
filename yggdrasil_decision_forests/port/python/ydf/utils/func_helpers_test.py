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

from typing import Any, Union

from absl.testing import absltest

from ydf.utils import func_helpers


class FuncHelpersTest(absltest.TestCase):

  def test_list_explicit_arguments_single_argument(self):
    @func_helpers.list_explicit_arguments
    def identity_or_false(foo=42, explicit_args=None) -> Union[int, bool]:
      if "foo" in explicit_args:
        return foo
      else:
        assert foo == 42
        return False

    self.assertEqual(identity_or_false(), False)
    args = {}
    self.assertEqual(identity_or_false(**args), False)
    self.assertEqual(identity_or_false(42), 42)
    self.assertEqual(identity_or_false(53), 53)
    self.assertEqual(identity_or_false(foo=42), 42)
    args = {"foo": 42}
    self.assertEqual(identity_or_false(**args), 42)
    args["foo"] = 53
    self.assertEqual(identity_or_false(**args), 53)

  def test_list_explicit_arguments_multiple_arguments(self):
    @func_helpers.list_explicit_arguments
    def identity_or_false(foo=42, bar=3.14, explicit_args=None) -> Any:
      if "foo" in explicit_args and "bar" in explicit_args:
        return (foo, bar)
      if "foo" in explicit_args:
        assert bar == 3.14
        return (foo, False)
      if "bar" in explicit_args:
        assert foo == 42
        return (False, bar)
      assert bar == 3.14
      assert foo == 42
      return (False, False)

    self.assertEqual(identity_or_false(), (False, False))
    self.assertEqual(identity_or_false(), (False, False))
    self.assertEqual(identity_or_false(42, 3.14), (42, 3.14))
    self.assertEqual(identity_or_false(42), (42, False))
    self.assertEqual(identity_or_false(bar=2.71), (False, 2.71))

  def test_list_explicit_arguments_raises_when_explicit_explicit_args(self):
    @func_helpers.list_explicit_arguments
    def func(foo=42, explicit_args=None) -> Union[int, bool]:
      if "foo" in explicit_args:
        return foo
      else:
        assert foo == 42
        return False

    with self.assertRaises(ValueError):
      func(explicit_args=123)


if __name__ == "__main__":
  absltest.main()
