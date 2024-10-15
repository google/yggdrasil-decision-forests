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

// TypeScript bindings for emscripten-generated code.
// Used for OSS only.
interface WasmModule {}

type EmbindString =
  | ArrayBuffer
  | Uint8Array
  | Uint8ClampedArray
  | Int8Array
  | string;
export interface Dataset {
  addCategoricalColumn(_0: EmbindString, _1: vectorString, _2: boolean): void;
  addNumericalColumn(_0: EmbindString, _1: vectorFloat): void;
  delete(): void;
}

export interface InternalModel {
  getInputFeatures(): vectorInputFeature;
  predict(): vectorFloat;
  getLabelClasses(): vectorString;
  newBatchOfExamples(_0: number): void;
  setBoolean(_0: number, _1: number, _2: boolean): void;
  setCategoricalInt(_0: number, _1: number, _2: number): void;
  setCategoricalSetString(_0: number, _1: number, _2: vectorString): void;
  setCategoricalSetInt(_0: number, _1: number, _2: vectorInt): void;
  setNumerical(_0: number, _1: number, _2: number): void;
  predictFromPath(_0: EmbindString): vectorFloat;
  describe(): string;
  save(_0: EmbindString): void;
  setCategoricalString(_0: number, _1: number, _2: EmbindString): void;
  delete(): void;
}

export interface vectorInputFeature {
  size(): number;
  get(_0: number): InputFeature | undefined;
  push_back(_0: InputFeature): void;
  resize(_0: number, _1: InputFeature): void;
  set(_0: number, _1: InputFeature): boolean;
  delete(): void;
}

export interface InternalLearner {
  trainFromDataset(_0: Dataset): InternalModel;
  init(_0: EmbindString, _1: EmbindString, _2: EmbindString): void;
  trainFromPath(_0: EmbindString): InternalModel;
  delete(): void;
}

export interface vectorFloat {
  size(): number;
  get(_0: number): number | undefined;
  push_back(_0: number): void;
  resize(_0: number, _1: number): void;
  set(_0: number, _1: number): boolean;
  delete(): void;
}

export interface vectorInt {
  push_back(_0: number): void;
  resize(_0: number, _1: number): void;
  size(): number;
  get(_0: number): number | undefined;
  set(_0: number, _1: number): boolean;
  delete(): void;
}

export interface vectorVectorFloat {
  push_back(_0: vectorFloat): void;
  resize(_0: number, _1: vectorFloat): void;
  size(): number;
  get(_0: number): vectorFloat | undefined;
  set(_0: number, _1: vectorFloat): boolean;
  delete(): void;
}

export interface vectorString {
  size(): number;
  get(_0: number): EmbindString | undefined;
  push_back(_0: EmbindString): void;
  resize(_0: number, _1: EmbindString): void;
  set(_0: number, _1: EmbindString): boolean;
  delete(): void;
}

export type InputFeature = {
  name: EmbindString;
  type: EmbindString;
  internalIdx: number;
  specIdx: number;
};

interface EmbindModule {
  Dataset: {new (): Dataset};
  InternalModel: {new (): InternalModel};
  vectorInputFeature: {new (): vectorInputFeature};
  InternalLearner: {new (): InternalLearner};
  vectorFloat: {new (): vectorFloat};
  vectorInt: {new (): vectorInt};
  vectorVectorFloat: {new (): vectorVectorFloat};
  vectorString: {new (): vectorString};
  CreateVectorString(_0: number): vectorString;
  CreateVectorInt(_0: number): vectorInt;
  CreateVectorFloat(_0: number): vectorFloat;
  InternalLoadModel(_0: EmbindString): InternalModel;
}

export type MainModule = WasmModule & EmbindModule;
export default function MainModuleFactory(
  options?: unknown,
): Promise<MainModule>;
