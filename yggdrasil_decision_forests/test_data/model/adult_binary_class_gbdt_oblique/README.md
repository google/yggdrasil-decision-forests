An oblique GBT model trained on the adult dataset.

```python
import pandas as pd
import ydf

ds_path = "third_party/yggdrasil_decision_forests/test_data/dataset"
model_path = "third_party/yggdrasil_decision_forests/test_data/model/adult_binary_class_gbdt_v2"

train_ds = pd.read_csv(f"{ds_path}/adult_train.csv")
model = ydf.GradientBoostedTreesLearner(
    label="income",
    split_axis="SPARSE_OBLIQUE",
    sparse_oblique_normalization="STANDARD_DEVIATION",
    sparse_oblique_weights="CONTINUOUS",
).train(train_ds)
model.set_node_format(ydf.NodeFormat.BLOB_SEQUENCE_GZIP)
model.save(model_path)
with open(f"{model_path}/describe.txt", "w") as f:
  f.write(model.describe(output_format="text"))
```
