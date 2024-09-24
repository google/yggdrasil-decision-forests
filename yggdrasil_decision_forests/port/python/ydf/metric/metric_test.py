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

import textwrap

from absl.testing import absltest
import numpy as np
from numpy import testing as npt

from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from yggdrasil_decision_forests.metric import metric_pb2
from ydf.metric import metric
from yggdrasil_decision_forests.utils import distribution_pb2


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

  def test_with_weights(self):
    c = metric.ConfusionMatrix(
        classes=(
            "a",
            "b",
        ),
        matrix=np.array([[1.123456, 12345.6789], [3.456789, 4.5678901]]),
    )
    self.assertEqual(
        str(c),
        """\
label (row) \\ prediction (col)
+---------+---------+---------+
|         |       a |       b |
+---------+---------+---------+
|       a | 1.12346 | 12345.7 |
+---------+---------+---------+
|       b | 3.45679 | 4.56789 |
+---------+---------+---------+
""",
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


class EvaluationTest(absltest.TestCase):

  def test_empty(self):
    proto_eval = metric_pb2.EvaluationResults()
    self.assertEqual(metric.Evaluation(proto_eval).to_dict(), {})

  def test_classification(self):
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
            rocs=[
                metric_pb2.Roc(),
                metric_pb2.Roc(),
                metric_pb2.Roc(
                    count_predictions=10,
                    auc=0.8,
                    pr_auc=0.7,
                    curve=[
                        metric_pb2.Roc.Point(
                            threshold=1, tp=2, fp=3, tn=4, fn=6
                        ),
                        metric_pb2.Roc.Point(
                            threshold=2, tp=1, fp=2, tn=3, fn=4
                        ),
                    ],
                ),
            ],
        ),
    )
    evaluation = metric.Evaluation(proto_eval)
    print(evaluation)
    dict_eval = evaluation.to_dict()
    self.assertLessEqual(
        {
            "accuracy": (1 + 4) / (1 + 2 + 3 + 4),
            "loss": 2.0,
            "num_examples": 1,
        }.items(),
        dict_eval.items(),
    )

    self.assertEqual(dict_eval["confusion_matrix"].classes, ("1", "2"))
    npt.assert_array_almost_equal(
        dict_eval["confusion_matrix"].matrix, [[1, 3], [2, 4]]
    )

    self.assertEqual(
        str(evaluation),
        textwrap.dedent("""\
        accuracy: 0.5
        confusion matrix:
            label (row) \\ prediction (col)
            +---+---+---+
            |   | 1 | 2 |
            +---+---+---+
            | 1 | 1 | 3 |
            +---+---+---+
            | 2 | 2 | 4 |
            +---+---+---+
        characteristics:
            name: '2' vs others
            ROC AUC: 0.8
            PR AUC: 0.7
            Num thresholds: 2
        loss: 2
        num examples: 1
        num examples (weighted): 1
        """),
    )

    _ = evaluation.html()

  def test_regression(self):
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
    evaluation = metric.Evaluation(proto_eval)
    print(evaluation)
    self.assertDictEqual(
        evaluation.to_dict(),
        {
            "loss": 2.0,
            "num_examples": 1,
            "rmse": 2.0,
            "rmse_ci95_bootstrap": (9.0, 10.0),
            "num_examples_weighted": 2,
        },
    )

    self.assertEqual(
        str(evaluation),
        textwrap.dedent("""\
        RMSE: 2
        RMSE 95% CI [B]: (9.0, 10.0)
        loss: 2
        num examples: 1
        num examples (weighted): 2
        """),
    )

    _ = evaluation.html()

  def test_ranking(self):
    proto_eval = metric_pb2.EvaluationResults(
        count_predictions_no_weight=1,
        loss_value=2,
        count_predictions=3,
        label_column=ds_pb.Column(name="my_label"),
        ranking=metric_pb2.EvaluationResults.Ranking(
            ndcg=metric_pb2.MetricEstimate(value=5),
            mrr=metric_pb2.MetricEstimate(value=9),
        ),
    )
    evaluation = metric.Evaluation(proto_eval)
    print(evaluation)
    self.assertDictEqual(
        evaluation.to_dict(),
        {
            "loss": 2.0,
            "ndcg": 5.0,
            "mrr": 9.0,
            "num_examples": 1,
            "num_examples_weighted": 3,
        },
    )

    self.assertEqual(
        str(evaluation),
        textwrap.dedent("""\
        NDCG: 5
        MRR: 9
        loss: 2
        num examples: 1
        num examples (weighted): 3
        """),
    )

    _ = evaluation.html()

  def test_uplift(self):
    proto_eval = metric_pb2.EvaluationResults(
        count_predictions_no_weight=1,
        loss_value=2,
        count_predictions=3,
        label_column=ds_pb.Column(name="my_label"),
        uplift=metric_pb2.EvaluationResults.Uplift(qini=6, auuc=7),
    )
    evaluation = metric.Evaluation(proto_eval)
    print(evaluation)
    self.assertDictEqual(
        evaluation.to_dict(),
        {
            "auuc": 7.0,
            "loss": 2.0,
            "num_examples": 1,
            "qini": 6.0,
            "num_examples_weighted": 3,
        },
    )

    self.assertEqual(
        str(evaluation),
        textwrap.dedent("""\
        QINI: 6
        AUUC: 7
        loss: 2
        num examples: 1
        num examples (weighted): 3
        """),
    )

    _ = evaluation.html()


if __name__ == "__main__":
  absltest.main()
