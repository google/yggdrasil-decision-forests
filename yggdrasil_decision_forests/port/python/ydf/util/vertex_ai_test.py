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
from ydf.util import vertex_ai


class TFExampleTest(absltest.TestCase):
  maxDiff = None

  def test_manager(self):
    self.assertEqual(
        vertex_ai.get_vertex_ai_cluster_spec("""\
{
    "cluster": {
        "workerpool0": [
            "cmle-training-workerpool0-669872baaf-0:2222"
        ],
        "workerpool1": [
            "cmle-training-workerpool1-669872baaf-0:2222",
            "cmle-training-workerpool1-669872baaf-1:2222"
        ]
        },
    "environment": "cloud",
    "task": {
        "type": "workerpool0",
        "index": 0
    }
}"""),
        vertex_ai.VertexAIClusterSpec(
            workers=[
                "cmle-training-workerpool1-669872baaf-0:2222",
                "cmle-training-workerpool1-669872baaf-1:2222",
            ],
            is_worker=False,
            port=2222,
        ),
    )

  def test_worker(self):
    self.assertEqual(
        vertex_ai.get_vertex_ai_cluster_spec("""\
{
    "cluster": {
        "workerpool0": [
            "cmle-training-workerpool0-669872baaf-0:2222"
        ],
        "workerpool1": [
            "cmle-training-workerpool1-669872baaf-0:2222",
            "cmle-training-workerpool1-669872baaf-1:2222"
        ]
        },
    "environment": "cloud",
    "task": {
        "type": "workerpool1",
        "index": 1
    }
}"""),
        vertex_ai.VertexAIClusterSpec(
            workers=[
                "cmle-training-workerpool1-669872baaf-0:2222",
                "cmle-training-workerpool1-669872baaf-1:2222",
            ],
            is_worker=True,
            port=2222,
        ),
    )

  def test_bad_pools(self):
    with self.assertRaisesRegex(ValueError, "Expecting two workpools"):
      vertex_ai.get_vertex_ai_cluster_spec("""\
{
    "cluster": {
        "workerpool0": [
            "cmle-training-workerpool0-669872baaf-0:2222"
        ]
        },
    "environment": "cloud",
    "task": {
        "type": "workerpool0",
        "index": 0
    }
}""")


if __name__ == "__main__":
  absltest.main()
