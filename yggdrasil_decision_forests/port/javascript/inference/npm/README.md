# YDF in JS

With this package, you can generate predictions of machine learning models
trained with [YDF](https://ydf.readthedocs.io) in the browser and with NodeJS.

## Usage example

First, let's train a machine learning model in python. For more details, read
[YDF's documentation](https://ydf.readthedocs.io).

In Python in a Colab or in a Jupyter Notebook, run:

```python
# Install YDF
!pip install ydf pandas

import ydf
import pandas as pd

# Download a training dataset
ds_path = "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset/"
train_ds = pd.read_csv(ds_path + "adult_train.csv")

# Train a Gradient Boosted Trees model
learner = ydf.GradientBoostedTreesLearner(label="income", pure_serving_model=True)
model = learner.train(train_ds)

# Save the model
model.save("/tmp/my_model")

# Zip the model
# Important: Use -j to not include the directory structure.
!zip -rj /tmp/my_model.zip /tmp/my_model
```

Then:

### Run the model with NodeJS and CommonJS

```js
(async function (){
    // Load the YDF library
    const ydf = await require("yggdrasil-decision-forests")();

    // Load the model
    const fs = require("node:fs");
    let model = await ydf.loadModelFromZipBlob(fs.readFileSync("./model.zip"));

    // Create a batch of examples.
    let examples = {
        "age": [39, 40, 40, 35],
        "workclass": ["State-gov", "Private", "Private", "Federal-gov"],
        "fnlwgt": [77516, 121772, 193524, 76845],
        "education": ["Bachelors", "Assoc-voc", "Doctorate", "9th"],
        "education_num": ["13", "11", "16", "5"],
        "marital_status": ["Never-married", "Married-civ-spouse", "Married-civ-spouse", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Craft-repair", "Prof-specialty", "Farming-fishing"],
        "relationship": ["Not-in-family", "Husband", "Husband", "Husband"],
        "race": ["White", "Asian-Pac-Islander", "White", "Black"],
        "sex": ["Male", "Male", "Male", "Male"],
        "capital_gain": [2174, 0, 0, 0],
        "capital_loss": [0, 0, 0, 0],
        "hours_per_week": [40, 40, 60, 40],
        "native_country": ["United-States", null, "United-States", "United-States"]
    };

    // Make predictions
    let predictions = model.predict(examples);
    console.log("predictions:", predictions);

    // Release model
    model.unload();
}())
```

### Run the model with NodeJS and ES6

```js
import * as fs from "node:fs";
import YggdrasilDecisionForests from 'yggdrasil-decision-forests';

// Load the YDF library
let ydf = await YggdrasilDecisionForests();

// Load the model
let model = await ydf.loadModelFromZipBlob(fs.readFileSync("./model.zip"));

// Create a batch of examples.
let examples = {
    "age": [39, 40, 40, 35],
    "workclass": ["State-gov", "Private", "Private", "Federal-gov"],
    "fnlwgt": [77516, 121772, 193524, 76845],
    "education": ["Bachelors", "Assoc-voc", "Doctorate", "9th"],
    "education_num": ["13", "11", "16", "5"],
    "marital_status": ["Never-married", "Married-civ-spouse", "Married-civ-spouse", "Married-civ-spouse"],
    "occupation": ["Adm-clerical", "Craft-repair", "Prof-specialty", "Farming-fishing"],
    "relationship": ["Not-in-family", "Husband", "Husband", "Husband"],
    "race": ["White", "Asian-Pac-Islander", "White", "Black"],
    "sex": ["Male", "Male", "Male", "Male"],
    "capital_gain": [2174, 0, 0, 0],
    "capital_loss": [0, 0, 0, 0],
    "hours_per_week": [40, 40, 60, 40],
    "native_country": ["United-States", null, "United-States", "United-States"]
};

// Make predictions
let predictions = model.predict(examples);
console.log("predictions:", predictions);

// Release model
model.unload();
```

### Run the model with in Browser

```html
<script src="./node_modules/yggdrasil-decision-forests/dist/inference.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.0/jszip.min.js"></script>
<script>
YggdrasilDecisionForests()
    .then(ydf => ydf.loadModelFromUrl("http://localhost:3000/model.zip"))
    .then(model => {
        let examples = {
            "age": [39, 40, 40, 35],
            "workclass": ["State-gov", "Private", "Private", "Federal-gov"],
            "fnlwgt": [77516, 121772, 193524, 76845],
            "education": ["Bachelors", "Assoc-voc", "Doctorate", "9th"],
            "education_num": ["13", "11", "16", "5"],
            "marital_status": ["Never-married", "Married-civ-spouse", "Married-civ-spouse", "Married-civ-spouse"],
            "occupation": ["Adm-clerical", "Craft-repair", "Prof-specialty", "Farming-fishing"],
            "relationship": ["Not-in-family", "Husband", "Husband", "Husband"],
            "race": ["White", "Asian-Pac-Islander", "White", "Black"],
            "sex": ["Male", "Male", "Male", "Male"],
            "capital_gain": [2174, 0, 0, 0],
            "capital_loss": [0, 0, 0, 0],
            "hours_per_week": [40, 40, 60, 40],
            "native_country": ["United-States", null, "United-States", "United-States"]
        };
        predictions = model.predict(examples);
        model.unload();
    });
</script>
```

## For developers

### Run unit tests

```sh
npm test
```

### Update the binary bundle

```sh
# Assume the shell is located in a clone of:
# https://github.com/google/yggdrasil-decision-forests.git

# Compile the YDF with WebAssembly
yggdrasil_decision_forests/port/javascript/tools/build_zipped_library.sh

# Extract the the content of `dist` in `yggdrasil_decision_forests/port/javascript/npm/dist`.
unzip dist/ydf.zip -d yggdrasil_decision_forests/port/javascript/npm/dist
```
