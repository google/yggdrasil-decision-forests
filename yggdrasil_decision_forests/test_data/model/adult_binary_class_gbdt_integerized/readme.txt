Train and test dataset and model were generated with

```python
import pandas as pd
import ydf
import numpy as np

ds_path = "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset"

# Download and load the dataset into Pandas DataFrames
train_ds = pd.read_csv(f"{ds_path}/adult_train.csv")
test_ds = pd.read_csv(f"{ds_path}/adult_test.csv")

def map_col_to_int(df1, df2, name: str):
  c = df1[name].astype("category")
  d = dict(zip(c.cat.categories, range(1, len(c.cat.categories) + 1)))
  d[np.nan] = -1
  df1[name] = df1[name].map(d).astype(int)
  df2[name] = df2[name].map(d).astype(int)

categorical_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
for c in categorical_features:
  map_col_to_int(train_ds, test_ds, c)

column_defs = [
    ydf.Column(c, ydf.Semantic.CATEGORICAL, is_already_integerized=True)
    for c in categorical_features
]

model = ydf.GradientBoostedTreesLearner(label="income",
                                include_all_columns=True,
                                features=column_defs,
                                num_trees=50).train(train_ds)
```