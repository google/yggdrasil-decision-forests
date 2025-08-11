A default GBT model trained on a multi-dim synthetic dataset.

```python
import numpy as np
import pandas as pd
import ydf

model_path = "third_party/yggdrasil_decision_forests/test_data/model/synthetic_multidim_gbdt"

f1 = np.random.random((100, 5)) / 5
f2 = np.random.random((100, 3)) / 3
train_ds = {
    "f1": f1,
    "f2": f2,
    "label": np.sum(f1, axis=1) >= np.sum(f2, axis=1),
}
model = ydf.GradientBoostedTreesLearner(
    label="label", task=ydf.Task.REGRESSION
).train(train_ds)
model.set_node_format(ydf.NodeFormat.BLOB_SEQUENCE_GZIP)
model.save(model_path)
with open(f"{model_path}/describe.txt", "w") as f:
  f.write(model.describe(output_format="text"))
```
