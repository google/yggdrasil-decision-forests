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

"""Testing Metrics."""

import logging
import os
import textwrap

from absl import flags
from absl.testing import absltest
import numpy as np
from numpy import testing as npt

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from yggdrasil_decision_forests.metric import metric_pb2
from ydf.metric import metric
from yggdrasil_decision_forests.utils import distribution_pb2


def data_root_path() -> str:
  return ""


def pydf_test_data_path() -> str:
  return os.path.join(
      data_root_path(),
      "ydf/test_data",
  )


class EvaluationTest(absltest.TestCase):

  def test_no_metrics(self):
    e = metric.Evaluation()
    self.assertEqual(str(e), "No metrics")
    self.assertEqual(e.to_dict(), {})
    self.assertIsNone(e.accuracy)

  def test_set_and_get(self):
    e = metric.Evaluation()
    e.accuracy = 0.6
    self.assertEqual(e.accuracy, 0.6)
    self.assertEqual(e.to_dict()["accuracy"], 0.6)

    e.num_examples = 50
    self.assertEqual(e.accuracy, 0.6)
    self.assertEqual(e.num_examples, 50)

  def test_str(self):
    e = metric.Evaluation(accuracy=0.6)
    self.assertEqual(str(e), "accuracy: 0.6\n")

    e.num_examples = 50
    self.assertEqual(
        str(e),
        """accuracy: 0.6
num examples: 50
""",
    )

    e.custom_metrics["my_complex_metric"] = "hello\nworld"
    self.assertEqual(
        str(e),
        """accuracy: 0.6
my_complex_metric:
    hello
    world
num examples: 50
""",
    )

  def test_all_metrics(self):
    e = metric.Evaluation()
    e.loss = 0.1
    e.num_examples = 10
    e.accuracy = 0.2
    e.confusion_matrix = metric.ConfusionMatrix(
        classes=("a",), matrix=np.array([[1]])
    )
    e.rmse = 0.3
    e.rmse_ci95_bootstrap = (0.1, 0.4)
    e.ndcg = 0.4
    e.qini = 0.5
    e.auuc = 0.6
    e.num_examples_weighted = 0.7

    print(str(e))

    self.assertEqual(
        str(e),
        textwrap.dedent("""\
        accuracy: 0.2
        confusion matrix:
            label (row) \\ prediction (col)
            +---+---+
            |   | a |
            +---+---+
            | a | 1 |
            +---+---+
        RMSE: 0.3
        RMSE 95% CI [B]: (0.1, 0.4)
        NDCG: 0.4
        QINI: 0.5
        AUUC: 0.6
        loss: 0.1
        num examples: 10
        num examples (weighted): 0.7
        """),
    )

    golden_path = os.path.join(
        pydf_test_data_path(), "golden", "display_metric_to_html.html.expected"
    )
    golden_data = open(golden_path).read()
    effective_data = e._repr_html_()
    if golden_data != effective_data:
      effective_path = "/tmp/golden_test_value.html"
      with open(effective_path, "w") as f:
        f.write(effective_data)
      logging.info("Saving effective data to %s", effective_path)
    self.assertEqual(e._repr_html_(), golden_data)


class ConfusionTest(absltest.TestCase):

  def test_str(self):
    c = metric.ConfusionMatrix(
        classes=(
            "a",
            "b",
        ),
        matrix=np.array([[1, 2], [3, 4]]),
    )
    self.assertEqual(
        str(c),
        textwrap.dedent("""\
        label (row) \\ prediction (col)
        +---+---+---+
        |   | a | b |
        +---+---+---+
        | a | 1 | 2 |
        +---+---+---+
        | b | 3 | 4 |
        +---+---+---+
        """),
    )


class SafeDivTest(absltest.TestCase):

  def test_base(self):
    self.assertEqual(metric.safe_div(4.0, 2.0), 2.0)

  def test_zero(self):
    self.assertEqual(metric.safe_div(0.0, 0.0), 0.0)

  def test_error(self):
    with self.assertRaisesRegex(ValueError, "Cannot divide"):
      metric.safe_div(1.0, 0.0)


class CharacteristicTest(absltest.TestCase):

  def test_base(self):
    c = metric.Characteristic(
        name="name",
        roc_auc=0.1,
        pr_auc=0.2,
        per_threshold=[
            metric.CharacteristicPerThreshold(
                true_positive=1,
                false_positive=2,
                true_negative=3,
                false_negative=4,
                threshold=5,
            ),
            metric.CharacteristicPerThreshold(
                true_positive=10,
                false_positive=20,
                true_negative=30,
                false_negative=40,
                threshold=50,
            ),
        ],
    )
    self.assertEqual(c.name, "name")

    npt.assert_array_almost_equal(c.recalls, [(1) / (1 + 4), (10) / (10 + 40)])
    npt.assert_array_almost_equal(
        c.specificities, [3 / (3 + 2), 30 / (30 + 20)]
    )
    npt.assert_array_almost_equal(
        c.false_positive_rates, [2 / (2 + 3), 20 / (20 + 30)]
    )
    npt.assert_array_almost_equal(c.precisions, [1 / (1 + 2), 10 / (10 + 20)])
    npt.assert_array_almost_equal(
        c.accuracies,
        [(1 + 3) / (1 + 2 + 3 + 4), (10 + 30) / (10 + 20 + 30 + 40)],
    )


class EvaluationProtoTest(absltest.TestCase):

  def test_convert_empty(self):
    proto_eval = metric_pb2.EvaluationResults()
    self.assertEqual(
        metric.evaluation_proto_to_evaluation(proto_eval).to_dict(), {}
    )

  def test_convert_classification(self):
    proto_eval = metric_pb2.EvaluationResults(
        count_predictions_no_weight=1,
        count_predictions=1,
        label_column=ds_pb.Column(
            name="my_label",
            categorical=ds_pb.CategoricalSpec(
                number_of_unique_values=3, is_already_integerized=True
            ),
        ),
        classification=metric_pb2.EvaluationResults.Classification(
            confusion=distribution_pb2.IntegersConfusionMatrixDouble(
                counts=list([0, 0, 0, 0, 1, 2, 0, 3, 4]),
                sum=1 + 2 + 3 + 4,
                nrow=3,
                ncol=3,
            ),
            sum_log_loss=2,
        ),
    )
    print(metric.evaluation_proto_to_evaluation(proto_eval))
    dict_eval = metric.evaluation_proto_to_evaluation(proto_eval).to_dict()
    self.assertDictContainsSubset(
        {"accuracy": (1 + 4) / (1 + 2 + 3 + 4), "loss": 2.0, "num_examples": 1},
        dict_eval,
    )

    self.assertEqual(dict_eval["confusion_matrix"].classes, ("1", "2"))
    npt.assert_array_almost_equal(
        dict_eval["confusion_matrix"].matrix, [[1, 2], [3, 4]]
    )

  def test_convert_regression(self):
    proto_eval = metric_pb2.EvaluationResults(
        count_predictions_no_weight=1,
        loss_value=2,
        count_predictions=2,
        label_column=ds_pb.Column(name="my_label"),
        regression=metric_pb2.EvaluationResults.Regression(
            sum_square_error=8,
            bootstrap_rmse_lower_bounds_95p=9,
            bootstrap_rmse_upper_bounds_95p=10,
        ),
    )
    self.assertDictEqual(
        metric.evaluation_proto_to_evaluation(proto_eval).to_dict(),
        {
            "loss": 2.0,
            "num_examples": 1,
            "rmse": 2.0,
            "rmse_ci95_bootstrap": (9.0, 10.0),
            "num_examples_weighted": 2,
        },
    )

  def test_convert_ranking(self):
    proto_eval = metric_pb2.EvaluationResults(
        count_predictions_no_weight=1,
        loss_value=2,
        count_predictions=3,
        label_column=ds_pb.Column(name="my_label"),
        ranking=metric_pb2.EvaluationResults.Ranking(
            ndcg=metric_pb2.MetricEstimate(value=5)
        ),
    )
    self.assertDictEqual(
        metric.evaluation_proto_to_evaluation(proto_eval).to_dict(),
        {
            "loss": 2.0,
            "ndcg": 5.0,
            "num_examples": 1,
            "num_examples_weighted": 3,
        },
    )

  def test_convert_uplift(self):
    proto_eval = metric_pb2.EvaluationResults(
        count_predictions_no_weight=1,
        loss_value=2,
        count_predictions=3,
        label_column=ds_pb.Column(name="my_label"),
        uplift=metric_pb2.EvaluationResults.Uplift(qini=6, auuc=7),
    )
    self.assertDictEqual(
        metric.evaluation_proto_to_evaluation(proto_eval).to_dict(),
        {
            "auuc": 7.0,
            "loss": 2.0,
            "num_examples": 1,
            "qini": 6.0,
            "num_examples_weighted": 3,
        },
    )


if __name__ == "__main__":
  absltest.main()
