/*
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @fileoverview JS Wrapper for the model class.
 */

import {
  InputFeature,
  InternalModel,
  MainModule,
  vectorFloat,
} from '../training_for_types';
import * as conversion from '../util/conversion';

declare interface ModelInterface {
  inputFeatures: InputFeature[];
  labelClasses: string[];
  predict: (examples: string | {[key: string]: unknown[]}) => number[];
  describe: () => string;
  save: () => Promise<Blob>;
  unload: () => void;
}

/**
 * Typings for the Emscripten file system.
 * TODO: Use Emscripten-provided typings once
 * https://github.com/emscripten-core/emscripten/issues/20296 is resolved.
 */
export declare interface EmscriptenFS {
  writeFile: (
    fileName: string,
    data: string | Uint8Array,
    opts?: {encoding: 'binary'; flags?: string},
  ) => void;
  readFile: (
    path: string,
    opts: {encoding: 'binary'; flags?: string},
  ) => Uint8Array;
  unlink: (fileName: string) => void;
  readdir: (path: string) => string[];
  stat: (path: string) => string[];
  mkdirTree: (path: string) => void;
  rmdir: (path: string) => void;
}

interface JSZipInputByType {
  base64: string;
  string: string;
  text: string;
  binarystring: string;
  array: number[];
  uint8array: Uint8Array;
  arraybuffer: ArrayBuffer;
  blob: Blob;
}

interface JSZipOutputByType {
  base64: string;
  text: string;
  binarystring: string;
  array: number[];
  uint8array: Uint8Array;
  arraybuffer: ArrayBuffer;
  blob: Blob;
}

type JSZipInputFileFormat = JSZipInputByType[keyof JSZipInputByType];
type JSZipInputType = keyof JSZipInputByType;
type JSZipOutputType = keyof JSZipOutputByType;

interface JSZipGeneratorOptions<T extends JSZipOutputType = JSZipOutputType> {
  compressionOptions?: null | {
    level: number;
  };
  type?: T;
  comment?: string;
  mimeType?: string;
  encodeFileName?(filename: string): string;
  streamFiles?: boolean;
  platform?: 'DOS' | 'UNIX';
}

declare interface JSZipInterface {
  /**
   * Create JSZip instance
   */
  new (data?: any, options?: any): this;
  file<T extends JSZipInputType>(
    path: string,
    data: JSZipInputByType[T] | Promise<JSZipInputByType[T]>,
  ): this;
  file<T extends JSZipInputType>(path: string, data: null): this;
  generateAsync<T extends JSZipOutputType>(
    options?: JSZipGeneratorOptions<T>,
  ): Promise<JSZipOutputByType[T]>;
  loadAsync(data: JSZipInputFileFormat): any;
}

declare var JSZip: JSZipInterface;
// copybara:insert_begin(No node support in G3)
// if ((Module as any)['ENVIRONMENT'] === 'NODE') {
//   // In Nodejs
//   // @ts-ignore
//   JSZip = require('jszip');
// }
// copybara:insert_end

declare var Module: MainModule;
// TODO: Use Emscripten-provided typings once
// https://github.com/emscripten-core/emscripten/issues/20296 is resolved.
const FS: EmscriptenFS = (Module as any)['FS'];

const ColumnType = {
  NUMERICAL: 'NUMERICAL',
  CATEGORICAL: 'CATEGORICAL',
  CATEGORICAL_SET: 'CATEGORICAL_SET',
  BOOLEAN: 'BOOLEAN',
};

/**
 * A YDF model.
 */
export class Model implements ModelInterface {
  inputFeatures: InputFeature[];
  labelClasses: string[];
  constructor(private internalModel: InternalModel | null) {
    if (this.internalModel === null) {
      throw new Error('The internal model cannot be null.');
    }
    const rawInputFeatures = this.internalModel.getInputFeatures();

    this.inputFeatures = [];
    for (let i = 0; i < rawInputFeatures.size(); i++) {
      const inputFeature = rawInputFeatures.get(i);
      if (inputFeature === undefined) {
        throw new Error(`Found undefined input feature at index ${i}`);
      }
      this.inputFeatures.push(inputFeature);
    }

    this.labelClasses = conversion.ccVectorStringToJSVector(
      this.internalModel.getLabelClasses(),
    );
  }

  /**
   * Lists the input features of the model.
   */
  getInputFeatures() {
    return this.inputFeatures;
  }

  /**
   * Lists the label classes for a classification model. Empty otherwise.
   * In case of multi-class classification, "getLabelClasses" maps to the second
   * dimension of the probablity array returned by "predict".
   */
  getLabelClasses() {
    return this.labelClasses;
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
   *
   * In case of a regression, binary-classification, ranking, anomaly detection
   * , or uplifting model, the prediction is an array of shape [num_examples].
   * For example, predictions[0] is the prediction for the first example, and
   * predictions[1] is the prediction for the second example.
   *
   * For a binary classification model, the prediction is the probability of
   * the positive class i.e. `getLabelClasses()[1]`.
   *
   * For a multi-class classification model, the prediction is an array of shape
   * [num_examples, num_classes] where the num_classes dimension is mapped to
   * the label classes "getLabelClasses()".
   */
  predict(examples: string | {[key: string]: unknown[]}): number[] {
    if (this.internalModel === null) {
      throw new Error('The model is null.');
    }
    let internalPredictions: vectorFloat;
    if (typeof examples === 'string') {
      const fileName = `/test_${Math.floor(Math.random() * 0xffffffff)}.csv`;
      FS.writeFile(fileName, examples);

      internalPredictions = this.internalModel.predictFromPath(
        `csv:${fileName}`,
      );
      FS.unlink(fileName);
    } else if (typeof examples === 'object') {
      const decoder = new TextDecoder('utf-8');

      // Detect the number of examples and ensure that all the fields (i.e.
      // features) are arrays with the same number of items.
      let numExamples = undefined;
      for (const values of Object.values(examples)) {
        if (!Array.isArray(values)) {
          throw new Error('Features should be arrays');
        }
        if (numExamples === undefined) {
          numExamples = values.length;
        } else if (numExamples !== values.length) {
          throw new Error('Features have a different number of values');
        }
      }
      if (numExamples === undefined) {
        // The example does not contain any features.
        throw new Error('No features defined for the model.');
      }

      // Fill the examples
      this.internalModel.newBatchOfExamples(numExamples);
      for (const featureDef of this.inputFeatures) {
        const featureName = conversion.embindStringToString(
          featureDef.name,
          decoder,
        );
        const values = examples[featureName];
        if (featureDef.type === ColumnType.NUMERICAL) {
          for (const [exampleIdx, value] of values.entries()) {
            if (value === null) continue;
            this.internalModel.setNumerical(
              exampleIdx,
              featureDef.internalIdx,
              value as number,
            );
          }
        } else if (featureDef.type === ColumnType.CATEGORICAL) {
          for (const [exampleIdx, value] of values.entries()) {
            if (value === null) continue;
            if (typeof value === 'string') {
              this.internalModel.setCategoricalString(
                exampleIdx,
                featureDef.internalIdx,
                value,
              );
            } else {
              this.internalModel.setCategoricalInt(
                exampleIdx,
                featureDef.internalIdx,
                value as number,
              );
            }
          }
        } else if (featureDef.type === ColumnType.BOOLEAN) {
          for (const [exampleIdx, value] of values.entries()) {
            if (value === null) continue;
            this.internalModel.setBoolean(
              exampleIdx,
              featureDef.internalIdx,
              value as boolean,
            );
          }
        } else if (featureDef.type === ColumnType.CATEGORICAL_SET) {
          for (const [exampleIdx, value] of values.entries()) {
            if (value === null) continue;
            const vectorString = conversion.stringArrayToVectorString(
              value as string[],
            );
            this.internalModel.setCategoricalSetString(
              exampleIdx,
              featureDef.internalIdx,
              vectorString,
            );
            vectorString.delete();
          }
        } else {
          throw new Error(`Non supported feature type ${featureDef}`);
        }
      }

      // Extract predictions
      internalPredictions = this.internalModel.predict();
    } else {
      throw new Error(
        `Unknown data type ${typeof examples}. Data must be either a string containing CSV data or a named dictionary of values.`,
      );
    }
    const jsPredictions =
      conversion.ccVectorFloatToJSVector(internalPredictions);
    internalPredictions.delete();
    return jsPredictions;
  }

  describe(): string {
    if (this.internalModel === null) {
      throw new Error('The model has been unloaded');
    }
    return this.internalModel.describe();
  }

  async save(): Promise<Blob> {
    console.log('Model save');
    if (this.internalModel === null) {
      throw new Error('The model has been unloaded');
    }
    const zip = new JSZip();
    function addFilesRecursively(path: string) {
      const entries = FS.readdir(path);
      for (const entry of entries) {
        if (entry === '.' || entry === '..') continue;
        const fullPath = `${path}/${entry}`;
        // This assumes the YDF directory has no subdirectories
        const fileContent = FS.readFile(fullPath, {encoding: 'binary'});
        console.log(entry);
        zip.file(entry, fileContent); // Remove leading '/'
      }
    }

    const modelFolderPath = `/model_${Math.floor(Math.random() * 0xffffffff)}`;
    this.internalModel.save(modelFolderPath);
    console.log(modelFolderPath);
    addFilesRecursively(modelFolderPath);
    return zip.generateAsync({type: 'blob'});
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
   * TODO: Unload the model automatically using "finalizers" once available
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
 * Loads a model from a blob containing a zipped Yggdrasil model.
 */
export async function loadModelFromZipBlob(serializedModel: Blob) {
  // Create model directory in RAM.
  const modelPath = `model_${Math.floor(Math.random() * 0xffffffff)}`;
  FS.mkdirTree(modelPath);

  const zippedModel = await JSZip.loadAsync(serializedModel);

  // Extract model
  const promiseUncompressed: any[] = [];

  zippedModel.forEach((filename: string, file: any) => {
    promiseUncompressed.push(
      file
        .async('blob')
        .then((data: any) => conversion.blobToArrayBuffer(data))
        .then((data: any) => {
          if (filename.endsWith('/')) {
            throw new Error(
              'The model zipfile is expected to be a flat zip file, but it contains a sub-directory. If zipping the model manually with the `zip` tool, make sure to use the `-j` option.',
            );
          }
          FS.writeFile(modelPath + '/' + filename, new Uint8Array(data), {
            'encoding': 'binary',
          });
        }),
    );
  });

  await Promise.all(promiseUncompressed);

  // Load model in Yggdrasil.
  const modelWasm = Module.InternalLoadModel(modelPath);

  // Delete the model on disk.
  for (const filename of FS.readdir(modelPath)) {
    if (filename === '.' || filename === '..') {
      continue;
    }
    FS.unlink(modelPath + '/' + filename);
  }
  FS.rmdir(modelPath);

  if (modelWasm == null) {
    throw new Error('Cannot parse model');
  }

  return new Model(modelWasm);
}
