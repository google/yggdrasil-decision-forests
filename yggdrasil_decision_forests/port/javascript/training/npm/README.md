# YDF Training in JS

With this package, you can train machine learning models with
[YDF](https://ydf.readthedocs.io) in the browser and with Node.js.

## Usage example

This package supports multiple surfaces.

### Run the model with in Browser

```html
<script src="./node_modules/ydf-training/dist/training.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.0/jszip.min.js"></script>
<script>
YDFTraining()
    .then(ydf => fetch("http://localhost:3000/data.csv"))
    .then( async (response) => {
      const data = await response.text()
      const task = "CLASSIFICATION";
      const label = "label";
      const model = new ydf.GradientBoostedTreesLearner(label, task).train(data);
      const predictions = model.predict(data);
      console.log(model.describe());
      const modelAsZipBlob = await model.save();
      model.unload();
    });
</script>
```

### Run the model with NodeJS and CommonJS

```js
(async function (){
    // Load the YDF library.
    const ydf = await require("ydf-training")();

    // Load the model.
    const fs = require("node:fs");
    const data = fs.readFileSync("data.csv", 'utf-8');
    const task = "CLASSIFICATION";
    const label = "label";
    const model = new ydf.GradientBoostedTreesLearner(label, task).train(data);

    // Make predictions.
    const predictions = model.predict(data);
    console.log("predictions:", predictions);

    // Describe the model.
    const description = model.describe();
    console.log( predictions);

    // Save the model to disk.
    var fileReader = new FileReader();
    fileReader.onload = function() {
      fs.writeFileSync('model.zip', Buffer.from(new Uint8Array(this.result)));
    };
    const blob = await model.save();
    fileReader.readAsArrayBuffer(blob);

    // Release model
    model.unload();
}())
```

### Run the model with NodeJS and ES6

```js
import * as fs from "node:fs";
import YDFTraining from 'ydf-training';

// Load the YDF library
let ydf = await YDFTraining();

const data = fs.readFileSync("data.csv", 'utf-8');
const task = "CLASSIFICATION";
const label = "label";
const model = new ydf.GradientBoostedTreesLearner(label, task).train(data);

// Make predictions.
const predictions = model.predict(data);
console.log("predictions:", predictions);

// Describe the model.
const description = model.describe();
console.log( predictions);

// Save the model to disk.
var fileReader = new FileReader();
fileReader.onload = function() {
  fs.writeFileSync('model.zip', Buffer.from(new Uint8Array(this.result)));
};
const blob = await model.save();
fileReader.readAsArrayBuffer(blob);

// Release model
model.unload();
```

## For developers

### Run unit tests

```sh
npm test
```

### Update the binary bundle

Building the binary bundle requires Bazel and Node.js installed.

```sh
# Assume the shell is located in a clone of:
# https://github.com/google/yggdrasil-decision-forests.git

# Compile the YDF Training
yggdrasil_decision_forests/port/javascript/tools/build_zipped_library.sh
```

You can find the compiled bundle in
`third_party/yggdrasil_decision_forests/port/javascript/training/npm/`