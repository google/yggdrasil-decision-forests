# TensorFlow Serving

[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) (TF Serving)
is a tool to run TensorFlow models online in large production settings using a
RPC or REST API. TensorFlow Decision Forests (TF-DF) is supported natively by TF
Serving >=2.11.

TF-DF models are directly compatible with TF Serving. Yggdrasil models can be
used with TF Serving after being
[converted](https://ydf.readthedocs.io/en/latest/convert_model.html#convert-a-yggdrasil-model-to-a-tensorflow-decision-forests-model)
first.

Check our
[TF Serving + TF-DF tutorial](https://www.tensorflow.org/decision_forests/tensorflow_serving)
on tensorflow.org for more details.

``` {note}
TensorFlow Serving 2.11 was not yet released (Oct. 2022). In the meantime,
*TensorFlow Serving 2.11 Nightly* with support with TF-DF is available
[here](https://github.com/tensorflow/decision-forests/releases/tag/serving-1.0.1).
Prior version of TF Serving (e.g. TF Serving 2.8-2.10) are compatible with
TF-DF. However, they requires to be *re-compiled* with TF-DF support
([instructions](https://github.com/tensorflow/decision-forests/blob/main/documentation/tensorflow_serving.md#compile-tf-seringtf-decision-forests-from-source)).
```
