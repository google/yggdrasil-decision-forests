A default GBT model trained on the iris dataset.

```python
import pandas as pd
import ydf

ds_path = "third_party/yggdrasil_decision_forests/test_data/dataset"
model_path = "third_party/yggdrasil_decision_forests/test_data/model/iris_multi_class_gbdt_v2"

train_ds = pd.read_csv(f"{ds_path}/iris.csv")
model = ydf.GradientBoostedTreesLearner(label="class").train(train_ds)
model.save(model_path)
with open(f"{model_path}/describe.txt", "w") as f:
  f.write(model.describe(output_format="text"))
```
