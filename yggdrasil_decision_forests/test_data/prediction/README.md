# Predictions

This directory contains predictions from YDF and other frameworks models.

In the case of YDF predictions, the files contain the golden predictions of the
corresponding models in the `model` directory. Those predictions have been
generated using the `:predict` CLI tool. For example:

```shell

./yggdrasil_decision_forests/cli/predict \
  --model=$yggdrasil_decision_forests/test_data/model/adult_binary_class_rf \
  --dataset=csv:yggdrasil_decision_forests/test_data/dataset/adult_test.csv \
  --output=csv:yggdrasil_decision_forests/test_data/prediction/adult_test_binary_class_rf.csv
```
