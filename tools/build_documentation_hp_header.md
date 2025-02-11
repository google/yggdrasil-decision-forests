# Hyper-parameters

Learners and their respective hyperparameters are listed on this page.

## Usage

With the **Python** and **TensorFlow Decision Forests APIs**, hyperparameters
are provided as **constructor arguments**. For example:

```python
import ydf
model = ydf.RandomForestLearner(num_trees=1000).train(...)

import tensorflow_decision_forests as tfdf
model = tfdf.keras.RandomForestModel(num_trees=1000)
```

With the **C++** and **CLI APIs**, the hyper-parameters are passed in the
**training configuration protobuffer**. For example:

```c++
learner: "RANDOM_FOREST"
[yggdrasil_decision_forests.model.random_forest.proto.random_forest_config] {
  num_trees: 1000
}
```
