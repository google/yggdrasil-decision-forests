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

import threading
from absl import logging
from absl.testing import absltest
import pandas as pd
from pandas import testing as pd_testing
from ydf.util import log_book

LogBook = log_book.LogBook


class LogBookTest(absltest.TestCase):

  def test_basic(self):
    tmp_dir = self.create_tempdir().full_path
    e = LogBook(tmp_dir)
    self.assertFalse(e.exist({"a": 1, "b": 2}))
    self.assertFalse(e.exist({"a": 1, "b": 2, "c": [5, 6]}))
    e.add({"a": 1, "b": 2}, {"x": {1: 2.0}})

    self.assertTrue(e.exist({"a": 1, "b": 2}))
    self.assertFalse(e.exist({"a": 1, "b": 2, "c": [5, 6]}))

    e2 = LogBook(tmp_dir)
    self.assertTrue(e2.exist({"a": 1, "b": 2}))
    self.assertFalse(e2.exist({"a": 1, "b": 2, "c": [5, 6]}))

    logging.info("\n%s", e2.to_dataframe().to_dict(orient="list"))
    frame = e2.to_dataframe()
    frame["timestamp"] = "2025-01-25 13:15:57"
    pd_testing.assert_frame_equal(
        frame,
        pd.DataFrame({
            "id": [1],
            "timestamp": ["2025-01-25 13:15:57"],
            "a": [1],
            "b": [2],
            "x": [{"1": 2.0}],
        }),
    )

  def test_multi_thread(self):
    tmp_dir = self.create_tempdir().full_path
    e = LogBook(tmp_dir)

    def new_experiment(idx: int):
      e = LogBook(tmp_dir, print_num_experiments=False)
      key = {"idx": idx}
      assert not e.exist(key)
      result = {"something": idx * 2}
      e.add(key, result)

    threads = []
    for i in range(100):
      thread = threading.Thread(target=new_experiment, args=(i,))
      threads.append(thread)
      thread.start()

    for thread in threads:
      thread.join()

    logging.info("\n%s", e.to_dataframe())
    self.assertEqual(e.num_experiments(), 100)

  def test_wrong_inputs(self):
    tmp_dir = self.create_tempdir().full_path
    with self.assertRaisesRegex(ValueError, "`key` is not a dictionary"):
      LogBook(tmp_dir).add(1, {})  # pytype: disable=wrong-arg-types
    with self.assertRaisesRegex(ValueError, "`result` is not a dictionary"):
      LogBook(tmp_dir).add({}, 1)  # pytype: disable=wrong-arg-types
    with self.assertRaisesRegex(ValueError, "`key` contains a reserved key"):
      LogBook(tmp_dir).add({"id": 1}, {})
    with self.assertRaisesRegex(ValueError, "`result` contains a reserved key"):
      LogBook(tmp_dir).add({}, {"timestamp": 1})

  def test_with_default(self):
    tmp_dir = self.create_tempdir().full_path
    e1 = LogBook(tmp_dir)
    e1.add({"k1": 1}, {"x": 1})
    pd_testing.assert_frame_equal(
        e1.to_dataframe().drop("timestamp", axis=1),
        pd.DataFrame({
            "id": [1],
            "k1": [1],
            "x": [1],
        }),
    )

    e2 = LogBook(tmp_dir, default_keys={"k2": 1})
    pd_testing.assert_frame_equal(
        e2.to_dataframe().drop("timestamp", axis=1),
        pd.DataFrame({
            "id": [1],
            "k1": [1],
            "k2": [1],
            "x": [1],
        }),
    )
    self.assertTrue(e2.exist({"k1": 1}))
    self.assertTrue(e2.exist({"k1": 1, "k2": 1}))
    e2.add({"k1": 2}, {"x": 2})
    pd_testing.assert_frame_equal(
        e2.to_dataframe().drop("timestamp", axis=1),
        pd.DataFrame({
            "id": [1, 2],
            "k1": [1, 2],
            "k2": [1, 1],
            "x": [1, 2],
        }),
    )

    e3 = LogBook(tmp_dir, default_keys={"k2": 2})
    e3.add({"k1": 3, "k2": 3}, {"x": 3})
    pd_testing.assert_frame_equal(
        e3.to_dataframe().drop("timestamp", axis=1),
        pd.DataFrame({
            "id": [1, 2, 3],
            "k1": [1, 2, 3],
            "k2": [2, 2, 3],
            "x": [1, 2, 3],
        }),
    )
    self.assertTrue(e3.exist({"k1": 1}))
    self.assertTrue(e3.exist({"k1": 1, "k2": 2}))

  def test_filtering(self):
    tmp_dir = self.create_tempdir().full_path
    e = LogBook(tmp_dir)

    e.add({"a": 1, "b": 1}, {})
    e.add({"a": 2, "b": 2}, {})

    pd_testing.assert_frame_equal(
        e.to_dataframe({"a": 1}).drop("timestamp", axis=1),
        pd.DataFrame({
            "id": [1],
            "a": [1],
            "b": [1],
        }),
    )


if __name__ == "__main__":
  absltest.main()
