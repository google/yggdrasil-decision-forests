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

// Package canonical registers the "canonical" models.
//
// Model implementations are made accessible through a registration mechanism for three main usecases:
// - A user wants all the available official models (called "canonical" models") to be available.
//
//	In this case, the user import this "canonical" package. Importing this package registers the
//	canonical models.
//
// - A user is developing a custom model. This implementation is not canonical. The user import the
//
//	implementation package manually once.
//
// - A user is developing a size critical binary that serves a single type of model. The user import
//
//	the implementation package of the corresponding model.
//
// Remark, this registration system is similar as the one used in the C++ code. However, unlike the
// c++ code that only require for models to be added as "deps", the go code requires for models to
// be added as "deps" AND imported once.
//
// Like for the c++ code. Models (in the "/model" directory) and engines (i.e. optimized code to run
// models on specific hardware (in the "/serving" directory) are independent.
package canonical

import (
	_ "github.com/google/yggdrasil-decision-forests/yggdrasil_decision_forests/port/go/model/gradientboostedtrees" // Okay
)
