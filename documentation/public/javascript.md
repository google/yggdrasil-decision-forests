# Javascript

YDF offers two different npm packages to run on the web:

*   [ydf-inference](https://www.npmjs.com/package/ydf-inference) Only for 
    generating predictions using an existing model. Models can be trained with
    ydf-training (see below), YDF python, or any other YDF API. If you only need
    model predictions, use this package instead of ydf-training to save on
    binary size.
*   [ydf-training](https://www.npmjs.com/package/ydf-training) for both training
    models and generating predictions.

Both packages are compatible with NodeJS+CommonJS, NodeJS+ES6 and Browser JS.

## ydf-inference

`ydf-inference` is YDF's interface for model inference on the Web.
See the [Readme on npmjs.com](https://www.npmjs.com/package/ydf-inference) for
information about downloading and testing the package.

The following example shows how to download a YDF model and make predictions
on a Javascript dictionary of arrays.

```html
<script src="./node_modules/ydf-inference/dist/inference.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.0/jszip.min.js"></script>
<script>
YDFInference()
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

## ydf-training

`ydf-training` is YDF's interface for training and inspecting models in
Javascript. It is implemented with Javascript and WebAssembly.
See the [Readme on npmjs.com](https://www.npmjs.com/package/ydf-training) for
information about downloading and testing the package.

The following example shows how to train a Gradient Boosted Trees model on a 
first csv dataset, and then use this model to make predictions on a second csv
dataset.

```html
<script src="./node_modules/ydf-training/dist/training.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.0/jszip.min.js"></script>
<script>
YDFTraining()
    .then( async (ydf) => {
      // Download the datasets.
      const rawTrain = await fetch("http://localhost:3000/train.csv");
      const train = await rawTrain.text();
      const rawTest = await fetch("http://localhost:3000/test.csv");
      const test = await rawTest.text();

      // Prepare the training configuration.
      const task = "CLASSIFICATION";
      const label = "label";
    
      // Train the model.
      const model = new ydf.GradientBoostedTreesLearner(label, task).train(data);
      
      // Make predictions.
      const predictions = model.predict(data);

      // Print the description of the model.
      console.log(model.describe());

      // Save the model to later. This model can also be run with ydf-inference
      // or Python YDF.
      const modelAsZipBlob = await model.save();
      model.unload();
    });
</script>
```

### Known limitations

`ydf-training` currently only supports a subset of the functionality of YDF's
Python surface, namely **supervised learning** with Random Forests and 
Gradient Boosted Trees. Hyperparameter configuration is not yet supported.
Additionally, model evaluation and model analysis are not yet supported.

For feature requests, please open an issue [on GitHub](https://github.com/google/yggdrasil-decision-forests).
