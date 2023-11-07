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

// Simplified logging library.
// To replace with absl c++ logging when available.
// IWYU pragma: private, include "yggdrasil_decision_forests/utils/logging.h"
#ifndef YGGDRASIL_DECISION_FORESTS_UTILS_LOGGING_DEFAULT_H_
#define YGGDRASIL_DECISION_FORESTS_UTILS_LOGGING_DEFAULT_H_

#include <iostream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/time/time.h"

ABSL_DECLARE_FLAG(bool, alsologtostderr);

// Initialize the logging. Should be called when the program starts.
void InitLogging(const char* usage, int* argc, char*** argv, bool remove_flags);

enum Severity { INFO, WARNING, FATAL };

// Logging.
//
// Usage example:
//   YDF_LOG(INFO) << "Hello world";
//
#define YDF_LOG(sev) YDF_LOG_##sev

#define YDF_LOG_INFO ::internal::LogMessage(INFO, __FILE__, __LINE__)
#define YDF_LOG_WARNING ::internal::LogMessage(WARNING, __FILE__, __LINE__)
#define YDF_LOG_FATAL ::internal::FatalLogMessage(__FILE__, __LINE__)
#define YDF_LOG_ERROR ::internal::LogMessage(WARNING, __FILE__, __LINE__)
#define YDF_LOG_QFATAL ::internal::FatalLogMessage(__FILE__, __LINE__)
#define YDF_LOG_DFATAL ::internal::FatalLogMessage(__FILE__, __LINE__)

namespace yggdrasil_decision_forests {
namespace logging {

extern int logging_level;
extern bool show_details;

// Sets the amount of logging:
// 0: Only fatal i.e. before a crash of the program.
// 1: Only warning and fatal.
// 2: Info, warning and fatal i.e. all logging. Default.
inline void SetLoggingLevel(int level) { logging_level = level; }

// If true, logging messages include the timestamp, filename and line. True by
// default.
inline void SetDetails(bool value) { show_details = value; }

// Tests if a given logging level should be visible.
inline bool IsLoggingEnabled(Severity severity) {
#ifdef YDF_ALWAYS_LOG
  return true;
#else
  if (!absl::GetFlag(FLAGS_alsologtostderr)) {
    return false;
  }
#endif

  switch (severity) {
    case Severity::INFO:
      return logging_level >= 2;
    case Severity::WARNING:
      return logging_level >= 1;
    case Severity::FATAL:
      return true;
  }
}

}  // namespace logging
}  // namespace yggdrasil_decision_forests

// Evaluates an expression returning an absl::Status. If the status is not "OK"
// emit a fatal error message.
//
// Usage example:
//   CHECK_OK(status) << "Hello world";
//
#if __cplusplus >= 201703L

#ifndef CHECK_OK
#define CHECK_OK(expr)                                            \
  if (const auto status = expr; ABSL_PREDICT_FALSE(!status.ok())) \
  YDF_LOG(FATAL) << status
#endif

#else

#ifndef COMBINE_TERMS
#define COMBINE_TERMS_SUB(X, Y) X##Y  // helper macro
#define COMBINE_TERMS(X, Y) COMBINE_TERMS_SUB(X, Y)
#endif

#ifndef CHECK_OK
#define CHECK_OK(expr)                                                 \
  {                                                                    \
    const auto COMBINE_TERMS(status_for, __LINE__) = expr;             \
    if (ABSL_PREDICT_FALSE(!COMBINE_TERMS(status_for, __LINE__).ok())) \
      YDF_LOG(FATAL) << COMBINE_TERMS(status_for, __LINE__);           \
  }                                                                    \
  nullptr
#endif
#endif

#ifndef QCHECK_OK
#define QCHECK_OK(expr) CHECK_OK(expr)
#endif

// Evaluates an expression returning a boolean. If the returned value is false
// emit a fatal error message.
//
// Usage example:
//   CHECK(x==y) << "Hello world";
//
#ifndef CHECK
// NOLINTBEGIN
#define CHECK(expr) \
  if (!(expr)) YDF_LOG(FATAL) << "Check failed " #expr
// NOLINTEND
#endif

#ifndef QCHECK
#define QCHECK(expr) CHECK(expr)
#endif

// Like check in debug mode. No-op in release mode.
//
// Usage example:
//   DCHECK(x==y) << "Hello world";
//
#ifndef NDEBUG
// NOLINTBEGIN
#define DCHECK(expr) \
  if (!(expr)) YDF_LOG(FATAL) << "Check failed " #expr
// NOLINTEND
#else
#define DCHECK(expr) ::internal::NullSink()
#endif

// Check helpers.
//
// Usage example:
//   CHECK_EQ(x, y) << "Hello world";
//
#ifndef CHECK_EQ
#define CHECK_EQ(a, b) CHECK(a == b) << " with a=" << a << " and b=" << b
#define CHECK_NE(a, b) CHECK(a != b) << " with a=" << a << " and b=" << b
#define CHECK_GE(a, b) CHECK(a >= b) << " with a=" << a << " and b=" << b
#define CHECK_LE(a, b) CHECK(a <= b) << " with a=" << a << " and b=" << b
#define CHECK_GT(a, b) CHECK(a > b) << " with a=" << a << " and b=" << b
#define CHECK_LT(a, b) CHECK(a < b) << " with a=" << a << " and b=" << b

#define DCHECK_EQ(a, b) DCHECK(a == b) << " with a=" << a << " and b=" << b
#define DCHECK_NE(a, b) DCHECK(a != b) << " with a=" << a << " and b=" << b
#define DCHECK_GE(a, b) DCHECK(a >= b) << " with a=" << a << " and b=" << b
#define DCHECK_LE(a, b) DCHECK(a <= b) << " with a=" << a << " and b=" << b
#define DCHECK_GT(a, b) DCHECK(a > b) << " with a=" << a << " and b=" << b
#define DCHECK_LT(a, b) DCHECK(a < b) << " with a=" << a << " and b=" << b
#endif

namespace internal {

// Extraction the filename from a path.
inline absl::string_view ExtractFilename(absl::string_view path) {
  auto last_sep = path.find_last_of("/\\");
  if (last_sep == std::string::npos) {
    // Start of filename no found.
    return path;
  }
  return path.substr(last_sep + 1);
}

class LogMessage {
 public:
  LogMessage(Severity sev, absl::string_view file, int line) : sev_(sev) {
    if (!yggdrasil_decision_forests::logging::IsLoggingEnabled(sev)) {
      return;
    }

    if (yggdrasil_decision_forests::logging::show_details) {
      std::clog << "[";
      switch (sev) {
        case INFO:
          std::clog << "INFO";
          break;
        case WARNING:
          std::clog << "WARNING";
          break;
        case FATAL:
          std::clog << "FATAL";
          break;
        default:
          std::clog << "UNDEF";
          break;
      }
      std::clog << absl::FormatTime(" %y-%m-%d %H:%M:%E4S %Z ", absl::Now(),
                                    absl::LocalTimeZone())
                << ExtractFilename(file) << ":" << line << "] ";
    } else {
      std::clog << "["
                << absl::FormatTime("%y-%m-%d %H:%M:%E4S %Z", absl::Now(),
                                    absl::LocalTimeZone())
                << "] ";
    }
  }

  virtual ~LogMessage() {
    if (!yggdrasil_decision_forests::logging::IsLoggingEnabled(sev_)) {
      return;
    }
    std::clog << std::endl;
  }

  template <typename T>
  LogMessage& operator<<(const T& v) {
    if (!yggdrasil_decision_forests::logging::IsLoggingEnabled(sev_)) {
      return *this;
    }
    std::clog << v;
    return *this;
  }

 protected:
  Severity sev_;
};

class FatalLogMessage : public LogMessage {
 public:
  FatalLogMessage(absl::string_view file, int line)
      : LogMessage(FATAL, file, line) {}

  [[noreturn]] ~FatalLogMessage() {
    if (yggdrasil_decision_forests::logging::IsLoggingEnabled(sev_)) {
      std::clog << std::endl;
      std::clog.flush();
    }
    std::exit(1);
  }
};

// Consumes and ignores all input streams.
class NullSink {
 public:
  template <typename T>
  NullSink& operator<<(const T& v) {
    return *this;
  }
};

}  // namespace internal

#endif  // YGGDRASIL_DECISION_FORESTS_UTILS_LOGGING_DEFAULT_H_
