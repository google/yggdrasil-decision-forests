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

#include "yggdrasil_decision_forests/utils/time.h"

#include <cmath>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"

namespace yggdrasil_decision_forests::utils {

std::string FormatDurationForLogs(const absl::Duration& duration) {
  std::string result;
  const double all_seconds = absl::ToDoubleSeconds(duration);

  double seconds = all_seconds;
  const int hours = seconds / 3600;
  seconds -= hours * 3600;
  const int minutes = seconds / 60;
  seconds -= minutes * 60;

  const bool display_hours = hours > 0;
  const bool display_seconds = seconds > 0 || (all_seconds == 0);
  const bool display_minutes =
      minutes > 0 || (display_hours && display_seconds);

  // Hours
  if (display_hours) {
    absl::StrAppend(&result, hours, "h");
  }

  // Minutes
  if (display_minutes) {
    absl::StrAppend(&result, minutes, "m");
  }
  // Seconds
  // Print with a maximum precision of 2 decimals.
  if (display_seconds) {
    if (seconds == std::round(seconds)) {
      absl::StrAppendFormat(&result, "%.0fs", seconds);
    } else if ((seconds * 10) == std::round(seconds * 10)) {
      absl::StrAppendFormat(&result, "%.1fs", seconds);
    } else {
      absl::StrAppendFormat(&result, "%.2fs", seconds);
    }
  }
  return result;
}

}  // namespace yggdrasil_decision_forests::utils
