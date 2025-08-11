A default GBT model trained on the adult dataset.

```python
import pandas as pd
import ydf

ds_path = "third_party/yggdrasil_decision_forests/test_data/dataset"
model_path = "third_party/yggdrasil_decision_forests/test_data/model/abalone_regression_gbdt_v2"

train_ds = pd.read_csv(f"{ds_path}/abalone.csv")
model = ydf.GradientBoostedTreesLearner(label="Rings", task=ydf.Task.REGRESSION).train(train_ds)
model.set_node_format(ydf.NodeFormat.BLOB_SEQUENCE_GZIP)
model.save(model_path)
with open(f"{model_path}/describe.txt", "w") as f:
  f.write(model.describe(output_format="text"))
```
