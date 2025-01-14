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

import logging
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from yggdrasil_decision_forests.dataset import data_spec_pb2 as ds_pb
from ydf.deep import dataset as deep_dataset_lib
from ydf.deep import layer as layer_lib
from ydf.deep import preprocessor as preprocessor_lib
from ydf.utils import test_utils


class PreprocessorTest(parameterized.TestCase):

  def test_base(self):
    dataspec = ds_pb.DataSpecification(
        columns=[
            ds_pb.Column(
                name="l",
                type=ds_pb.ColumnType.CATEGORICAL,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=3,
                    items={
                        "OOD": ds_pb.CategoricalSpec.VocabValue(index=0),
                        "x": ds_pb.CategoricalSpec.VocabValue(index=1),
                        "y": ds_pb.CategoricalSpec.VocabValue(index=2),
                    },
                ),
            ),
            ds_pb.Column(
                name="n1",
                type=ds_pb.ColumnType.NUMERICAL,
                numerical=ds_pb.NumericalSpec(
                    mean=1,
                    standard_deviation=2,
                ),
                discretized_numerical=ds_pb.DiscretizedNumericalSpec(
                    boundaries=[1, 2, 3],
                ),
            ),
            ds_pb.Column(
                name="c1",
                type=ds_pb.ColumnType.CATEGORICAL,
                categorical=ds_pb.CategoricalSpec(
                    number_of_unique_values=3,
                    items={
                        "OOD": ds_pb.CategoricalSpec.VocabValue(index=0),
                        "x": ds_pb.CategoricalSpec.VocabValue(index=1),
                        "y": ds_pb.CategoricalSpec.VocabValue(index=2),
                    },
                ),
            ),
        ]
    )
    p = preprocessor_lib.Preprocessor(
        dataspec, [1, 2], numerical_zscore=True, numerical_quantiles=True
    )
    x = {
        "l": np.array(["x", "x", "y", "y", "x", "x", "x"]),
        "n1": np.array([-1.0, 1.0, 1.5, 2.0, 3.0, 3.5, np.nan]),
        "c1": np.array(["x", "y", "z", "", "x", "y", "y"]),
    }

    result_premodel = p.apply_premodel(x, has_labels=True)
    jax_batch = deep_dataset_lib.batch_numpy_to_jax(result_premodel)
    result_inmodel = p.apply_inmodel(jax_batch, has_labels=True)
    logging.info("result_premodel:%s", result_premodel)
    logging.info("result_inmodel:%s", result_inmodel)

    test_utils.assert_almost_equal(
        result_premodel,
        {
            "l": np.array([0, 0, 1, 1, 0, 0, 0], dtype=np.int32),
            "n1": np.array([-1.0, 1.0, 1.5, 2.0, 3.0, 3.5, np.nan]),
            "c1": np.array([1, 2, 0, 0, 1, 2, 2], dtype=np.int32),
        },
    )
    test_utils.assert_almost_equal(
        result_inmodel,
        [
            (
                layer_lib.Feature(
                    "l",
                    type=layer_lib.FeatureType.CATEGORICAL,
                    num_categorical_values=2,
                ),
                jnp.array([0, 0, 1, 1, 0, 0, 0], dtype=np.int32),
            ),
            (
                layer_lib.Feature("n1_ZSCORE", layer_lib.FeatureType.NUMERICAL),
                jnp.array([-1.0, 0.0, 0.25, 0.5, 1.0, 1.25, 0.0]),
            ),
            (
                layer_lib.Feature(
                    "n1_QUANTILE", layer_lib.FeatureType.NUMERICAL
                ),
                jnp.array([-2.0, 0.0, 0.5, 1.0, 2.0, 2.5, 0.0]) / 2,
            ),
            (
                layer_lib.Feature(
                    "c1",
                    type=layer_lib.FeatureType.CATEGORICAL,
                    num_categorical_values=3,
                ),
                jnp.array([1, 2, 0, 0, 1, 2, 2], dtype=np.int32),
            ),
        ],
    )


if __name__ == "__main__":
  absltest.main()
