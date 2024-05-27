# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Uniform display and controls the logs displayed on the different surfaces.

For the library developer:

  Python user facing logs are printed using "log.info" and "log.warning".
  Methods that can produce c++ logs should be wrapped in a CCLog context.

For the user:

  The logs are controlled globally with "ydf.verbose()".

Compatibility

  This code is tested with python, ipython, colab and jupyter notebook.
"""

import contextlib
import enum
import io
import sys
from typing import Any, Optional, Set, Union

from ydf.cc import ydf

# Current verbose level. See "verbose" for details.
_VERBOSE_LEVEL: int = 1

# The strick mode prints more warning messages.
_STRICT: bool = False


@enum.unique
class WarningMessage(enum.Enum):
  """All possible warning messages.

  Used to avoid showing warning messages multiple times.
  """

  CANNOT_SHOW_DETAILS_LOGS = 0
  CAST_NUMERICAL_TO_FLOAT32 = 1
  TO_TF_SAVED_MODEL_KERAS_MODE = 2


# List of already showed warning message that should not be displayed again.
_ALREADY_DISPLAYED_WARNING_IDS: Set[WarningMessage] = set()


def verbose(level: Union[int, bool] = 2) -> int:
  """Sets the verbose level of YDF.

  The verbose levels are:
    0 or False: Print no logs.
    1 or True: Print a few logs in a colab or notebook cell. Print all the logs
      in the console. This is the default verbose level.
    2: Prints all the logs on all surfaces.

  Usage example:

  ```python
  import ydf

  save_verbose = ydf.verbose(0)  # Hide all logs
  learner = ydf.RandomForestLearner(label="label")
  model = learner.train(pd.DataFrame({"feature": [0, 1], "label": [0, 1]}))
  ydf.verbose(save_verbose)  # Restore verbose level
  ```

  Args:
    level: New verbose level.

  Returns:
    The previous verbose level.
  """

  if isinstance(level, bool):
    level = 1 if level else 0

  global _VERBOSE_LEVEL
  old = _VERBOSE_LEVEL
  _VERBOSE_LEVEL = level
  return old


def current_log_level() -> int:
  """Returns the log level currently set."""
  return _VERBOSE_LEVEL


def info(msg: str, *args: Any) -> None:
  """Print an info message visible when verbose >=1.

  Usage example:
    info("Hello %s", "world")

  Args:
    msg: String message with replacement placeholders e.g. %s.
    *args: Placeholder replacement values.
  """

  if _VERBOSE_LEVEL >= 1:
    print(msg % args, flush=True)


def warning(
    msg: str,
    *args: Any,
    message_id: Optional[WarningMessage] = None,
    is_strict: bool = False
) -> None:
  """Print a warning message.

  A warning message is similar to an info message, except that:
  - There is a "Warning:" prefix.
  - When displaying multiple warning messages with the same "message_id", only
    the first one will be displayed.

  Usage example:
    warning("Hello %s", "world")

  Args:
    msg: String message with replacement placeholders e.g. %s.
    *args: Placeholder replacement values.
    message_id: Id of the warning message. If set, the message is only displayed
      once.
    is_strict: If true, the warning message is only disabled if the strict mode
      is enabled.
  """

  if is_strict and not _STRICT:
    return

  if message_id is not None:
    if message_id in _ALREADY_DISPLAYED_WARNING_IDS:
      return
    _ALREADY_DISPLAYED_WARNING_IDS.add(message_id)

  if _VERBOSE_LEVEL >= 1:
    print("Warning:", msg % args, flush=True, file=sys.stderr)


def strict(value: bool = True) -> None:
  """Sets the strict mode.

  When strict mode is enabled, more warnings are displayed.


  Args:
    value: New value for the strict mode.
  """

  global _STRICT
  _STRICT = value


def is_direct_output(stream=sys.stdout):
  """Checks if output stream redirects to the shell/console directly."""

  if stream.isatty():
    return True
  if isinstance(stream, io.TextIOWrapper):
    return is_direct_output(stream.buffer)
  if isinstance(stream, io.BufferedWriter):
    return is_direct_output(stream.raw)
  if isinstance(stream, io.FileIO):
    return stream.fileno() in [1, 2]
  return False


@contextlib.contextmanager
def _no_op_context():
  """Does nothing."""
  yield


@contextlib.contextmanager
def _hide_cc_logs():
  """Hide the CC logs in public build."""
  ydf.SetLoggingLevel(0, False)
  try:
    yield
  finally:
    ydf.SetLoggingLevel(2, True)


def cc_log_context():
  """Creates a context to display correctly C++ logs to the user."""

  if _VERBOSE_LEVEL == 0:
    return _hide_cc_logs()

  elif _VERBOSE_LEVEL == 1:
    # Only show CC logs in the console, but not in colab / notebook cells

    if is_direct_output():
      return _no_op_context()

    # Hide logs if in notebook. Logs are already hidden in colabs.
    return _hide_cc_logs()

  else:
    # Show CC logs everywhere

    # pylint: disable=g-import-not-at-top
    try:
      from colabtools.googlelog import CaptureLog  # pytype: disable=import-error
      # This is a Google Colab
      return CaptureLog()
    except ImportError:
      # Wurlitzer hangs when logs are shown directly.
      if is_direct_output():
        return _no_op_context()
      try:
        from wurlitzer import sys_pipes  # pytype: disable=import-error

        return sys_pipes()
      except ImportError:
        warning(
            "ydf.verbose(2) but logs cannot be displayed in the cell. Check"
            " colab logs or install wurlitzer with 'pip install wurlitzer'",
            message_id=WarningMessage.CANNOT_SHOW_DETAILS_LOGS,
        )
      return _no_op_context()
    # pylint: enable=g-import-not-at-top
