# Usage example for Yggdrasil Decision Forests in JavaScript

See "../README.md" for details.

See
[here](https://achoum.github.io/yggdrasil_decision_forests_js_example/example.html)
for a live version of this example.

## Data

The `model.zip` was created as:

```shell
zip -r -j third_party/yggdrasil_decision_forests/port/javascript/model.zip \
  third_party/yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt
```

The 4 input examples are the first 4 examples in
`third_party/yggdrasil_decision_forests/test_data/dataset/adult_test.csv`.

The predictions on those examples are expected to be:

Those predictions can also be generated using the CLI interface:

```shell
bazel build -c opt //third_party/yggdrasil_decision_forests/cli:predict

./bazel-bin/third_party/yggdrasil_decision_forests/cli/predict \
  --alsologtostderr \
  --model=third_party/yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt \
  --dataset=csv:third_party/yggdrasil_decision_forests/test_data/dataset/adult_test.csv \
  --output=csv:/tmp/predictions.csv

head /tmp/predictions.csv
# <=50K,>50K
# 0.987869,0.0121307
# 0.668998,0.331002
# 0.219888,0.780112
# 0.88848,0.11152
```
