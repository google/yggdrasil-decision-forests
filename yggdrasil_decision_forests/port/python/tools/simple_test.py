import ydf
import numpy as np

ds = {
    "label": np.array([0, 1] * 10),
    "feature": np.array([0, 1, 2, 3] * 5),
}
learner = ydf.RandomForestLearner(label="label")
model = learner.train(ds)
_ = model.predict(ds)
_ = model.evaluate(ds)
