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
from typing import Any, Iterator, Optional, Set, TypeVar, Union

from absl import logging

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
  USE_DISCRETIZED = 3
  USE_DISTRIBUTED = 4
  DONT_USE_PICKLE = 5
  TRAINING_VERTICAL_DATASET = 6
  # TODO: Remove when solved.
  WEIGHTED_NOT_SET_IN_EVAL = 7
  DEPRECATED_EVALUATION_TASK = 8
  UNNECESSARY_TASK_ARGUMENT = 9
  UNNECESSARY_LABEL_ARGUMENT = 10
  TFE_READING_IN_PYTHON_IS_SLOW = 11
  CATEGORICAL_LOOK_LIKE_NUMERICAL = 12
  TRAINING_NEURAL_NET_WITHOUT_VALID = 13
  TRAIN_TRANSFORMER_ON_CPU = 14
  AD_PERMUTATION_VARIABLE_IMPORTANCE_NOT_ENABLED = 15


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


def reduce_verbose(v: Union[int, bool]) -> int:
  """Reduces verbose by "one level"."""
  if isinstance(v, int):
    return max(0, v - 1)
  else:
    return 0


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
    logging.info(msg, *args)


def debug(msg: str, *args: Any) -> None:
  """Print an info message visible when verbose >=2.

  Usage example:
    debug("Hello %s", "world")

  Args:
    msg: String message with replacement placeholders e.g. %s.
    *args: Placeholder replacement values.
  """

  if _VERBOSE_LEVEL >= 2:
    print(msg % args, flush=True)
    logging.info(msg, *args)


def warning(
    msg: str,
    *args: Any,
    message_id: Optional[WarningMessage] = None,
    is_strict: bool = False,
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
    print("[Warning]", msg % args, flush=True, file=sys.stderr)
  logging.warning(msg, *args)


def strict(value: bool = True) -> None:
  """Sets the strict mode.

  When strict mode is enabled, more warnings are displayed.


  Args:
    value: New value for the strict mode.
  """

  global _STRICT
  _STRICT = value


def maybe_warning_large_dataset(
    num_training_examples: int,
    distributed: bool,
    discretize_numerical_columns: bool,
):
  """Prints a warning if large training is not optimal."""

  num_example_limit_non_distributed = 10_000_000
  num_example_limit_non_discretized = 2_000_000

  if (
      not distributed
      and num_training_examples >= num_example_limit_non_distributed
  ):
    warning(
        "On large datasets, distributed training can significantly speed"
        " training. See:"
        " https://ydf.readthedocs.io/en/latest/tutorial/distributed_training",
        message_id=WarningMessage.USE_DISTRIBUTED,
    )

  if (
      not discretize_numerical_columns
      and num_training_examples >= num_example_limit_non_discretized
  ):
    warning(
        "On large datasets, using discretized numerical features (i.e."
        " `discretize_numerical_columns=True`) can significantly speed-up"
        " training without impact on model quality.",
        message_id=WarningMessage.USE_DISCRETIZED,
    )


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


@contextlib.contextmanager
def _show_cc_logs():
  """Show the CC logs in stderr for the public build."""
  ydf.SetLoggingLevel(2, False)
  try:
    yield
  finally:
    ydf.SetLoggingLevel(0, True)


@contextlib.contextmanager
def cc_log_context():
  """Creates a context to display correctly C++ logs to the user.

  "cc_log_context" should wrap all the C++ calls.

  Informally, C++ logs are only directly visible to the user when verbose == 2
  i.e. visible in colab or in a terminal if using one. However, c++ logs are
  always visible to absl sinks (e.g., for google cloud logging).

  Assumes the current status is the output of "InitLoggingLib":
    minloglevel = info
    stderrthreshold = error

  Yields:
    Empty yield in context.
  """

  if _VERBOSE_LEVEL <= 1:
    # Don't show anything (except for fatal messages).
    with _no_op_context():
      yield
    return

  # Show as much c++ logs as possible.

  # pylint: disable=g-import-not-at-top
  try:
    from colabtools.googlelog import CaptureLog  # pytype: disable=import-error
    # We are in a Google Colab
    with _show_cc_logs():
      with CaptureLog():
        yield
    return

  except ImportError:
    if is_direct_output():
      # We are in a terminal
      # Note: Wurlitzer hangs when logs are shown directly.
      with _show_cc_logs():
        yield
      return
    try:
      # We are in a Notebook
      from wurlitzer import sys_pipes  # pytype: disable=import-error
      # We are in a Notebook with Wurlitzer
      with _show_cc_logs():
        with sys_pipes():
          yield
      return
    except ImportError:
      # We are in a Notebook without Wurlitzer
      warning(
          "ydf.verbose(2) but logs cannot be displayed in the cell. Check"
          " colab logs or install wurlitzer with 'pip install wurlitzer'",
          message_id=WarningMessage.CANNOT_SHOW_DETAILS_LOGS,
      )
    with _show_cc_logs():
      yield
    return
  # pylint: enable=g-import-not-at-top


T = TypeVar("T")


def maybe_tqdm(iterable: Iterator[T], *args, **kwargs) -> Iterator[T]:
  """Shows a tqdm progress bar if tqdm is installed and loggin level>=1."""

  if _VERBOSE_LEVEL == 0:
    return iterable

  try:
    # pylint: disable=g-import-not-at-top
    # pytype: disable=import-error
    import tqdm
    # pytype: enable=import-error
    # pylint: enable=g-import-not-at-top
    return tqdm.tqdm(iterable, *args, **kwargs)
  except ImportError:
    return iterable
