/**
 * @fileoverview JavaScript native inference for Yggdrasil Decision Forests
 * models.
 *
 * See "README.md" and "/example" for two usage examples.
 */

/**
 * Semantic of the input features. Those fields are equivalent to the
 * "ColumnType" proto enum defined in
 * "yggdrasil_decision_forests/dataset/data_spec.proto".
 * @enum {string}
 */
const ColumnType = {
  NUMERICAL: 'NUMERICAL',
  CATEGORICAL: 'CATEGORICAL',
  CATEGORICAL_SET: 'CATEGORICAL_SET',
};

/**
 * A list of input examples.
 *
 * Examples represents a list of examples. It is an object indexed by feature
 * names. For each feature, the object contains an array of values indexed by
 * example. In those arrays, the value null represent missing values.
 *
 * Example:
 *
 *  A list of two examples. The first example features are: f1=1 and
 *  f2=red. The second examples features are f1=2 and f2 is missing.
 *  const examples = {
 *     "f1": [1, 2],
 *     "f2": ["red", null],
 *    };
 *
 * @typedef {!Object<string, (!Array<number>|!Array<string>)>}
 */
let Examples;

/**
 * A machine learning model.
 */
class Model {
  /**
   * Creates a model.
   * @param {!InternalModel} internalModel The internal cc/wasm model.
   */
  constructor(internalModel) {
    /** @private {?InternalModel} */
    this.internalModel = internalModel;

    const rawInputFeatures = this.internalModel.getInputFeatures();
    /**
     * List of input features of the model.
     * @const @private @type {!Array<!InputFeature>}
     */
    this.inputFeatures =
        Array.from({length: rawInputFeatures.size()})
            .map((unused, index) => rawInputFeatures.get(index));
  }

  /**
   * Lists the input features of the model.
   * @return {!Array<!InputFeature>} List of input features of the model..
   */
  getInputFeatures() {
    return this.inputFeatures;
  }

  /**
   * Applies the model on a list of examples and returns the predictions.
   *
   * Usage example:
   *
   *  // A list of two examples. The first example features are: f1=1 and
   *  // f2=red. The second examples features are f1=2 and f2 is missing.
   *  const examples = {
   *     "f1": [1 ,2],
   *     "f2": ["red" ,null],
   *    };
   *
   *  const predictions = model.predict(examples);
   *  // If the model's output dimension is 1 (e.g. the model is a binary
   *  // classifier configured to return the probability of the "positive"
   *  // class), "predictions[0]" and "predictions[1]" are respectively the
   *  // probability predictions of the first and second examples.
   *
   * @param {!Examples} examples A list of examples represented by a single
   *     object containing one attribute for each of the input features of the
   *     model. Each field is an array containing a value for each of the
   *     the value null represent missing values.
   * @return {!Array<number>} The predictions of the model.
   */
  predict(examples) {
    if (typeof examples !== 'object') {
      throw Error('argument should be an array or an object');
    }

    // Detect the number of examples and ensure that all the fields (i.e.
    // features) are arrays with the same number of items.
    let numExamples = undefined;
    for (const values of Object.values(examples)) {
      if (!Array.isArray(values)) {
        throw Error('features should be arrays');
      }
      if (numExamples === undefined) {
        numExamples = values.length;
      } else if (numExamples !== values.length) {
        throw Error('features have a different number of values');
      }
    }
    if (numExamples === undefined) {
      // The example does not contain any features.
      throw Error('not features');
    }

    // Fill the example
    this.internalModel.newBatchOfExamples(numExamples);
    for (const featureDef of this.inputFeatures) {
      const values = examples[featureDef.name];
      if (featureDef.type === ColumnType.NUMERICAL) {
        for (const [exampleIdx, value] of values.entries()) {
          if (value === null) continue;
          this.internalModel.setNumerical(
              exampleIdx, featureDef.internalIdx, value);
        }
      } else if (featureDef.type === ColumnType.CATEGORICAL) {
        for (const [exampleIdx, value] of values.entries()) {
          if (value === null) continue;
          if (typeof value === 'string') {
            this.internalModel.setCategoricalString(
                exampleIdx, featureDef.internalIdx, value);
          } else {
            this.internalModel.setCategoricalInt(
                exampleIdx, featureDef.internalIdx, value);
          }
        }
      } else if (featureDef.type === ColumnType.CATEGORICAL_SET) {
        for (const [exampleIdx, value] of values.entries()) {
          if (value === null) continue;
          this.internalModel.setCategoricalSetString(
              exampleIdx, featureDef.internalIdx, value);
        }
      } else {
        throw Error(`Non supported feature type ${featureDef}`);
      }
    }

    // Extract predictions
    const internalPredictions = this.internalModel.predict();
    return Array.from({length: internalPredictions.size()})
        .map((unused, index) => internalPredictions.get(index));
  }

  /**
   * Unloads the model from memory.
   *
   * Usage example:
   *
   *    model.unload();
   *    model = null;
   *
   * Models (e.g. loaded with "loadModelFromUrl") should be released from
   * memory manually by calling "model.unload()". Not "unloading" the model
   * will result in a memory leak.
   *
   * TODO(gbm): Unload the model automatically using "finalizers" once available
   * in JS.
   *
   * See
   * https://emscripten.org/docs/porting/connecting_cpp_and_javascript/embind.html?highlight=pointer%20delete#memory-management
   * for details.
   */
  unload() {
    if (this.internalModel !== null) {
      this.internalModel.delete();
      this.internalModel = null;
    }
  }
}

/**
 * Loads a model from a URL.
 *
 * Usage example:
 *
 *    let model = null;
 *    ydf.loadModelFromUrl("model.zip").then((loadedModel) => {
 *        model = loadedModel;
 *    }
 *
 * @param {string} url Url to a model.
 * @return {!Promise<!Model>} The loaded model.
 */
Module['loadModelFromUrl'] = async function loadModelFromUrl(url) {
  // Download model
  const serializedModel = await fetch(url).then((r) => r.blob());

  // Create model directory in RAM.
  const modelPath = 'model_' + Math.floor(Math.random() * 0xFFFFFFFF);
  Module.FS.mkdirTree(modelPath);

  // Unzip Model
  const zippedModel = await JSZip.loadAsync(serializedModel);

  // Extract model
  const promiseUncompressed = [];

  zippedModel.forEach((filename, file) => {
    promiseUncompressed.push(
        file.async('blob')
            .then((data) => data.arrayBuffer())
            .then(
                (data) => Module.FS.writeFile(
                    modelPath + '/' + filename, new Uint8Array(data),
                    {'encoding': 'binary'})));
  });

  await Promise.all(promiseUncompressed);

  // Load model in Yggdrasil.
  const modelWasm = Module.InternalLoadModel(modelPath);

  // Delete the model on disk.
  for (const filename of Module.FS.readdir(modelPath)) {
    if (filename === '.' || filename === '..') {
      continue;
    }
    Module.FS.unlink(modelPath + '/' + filename);
  }
  Module.FS.rmdir(modelPath);

  if (modelWasm == null) {
    throw Error('Cannot parse model');
  }

  return new Model(modelWasm);
};
