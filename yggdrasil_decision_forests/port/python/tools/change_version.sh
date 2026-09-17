#!/bin/bash
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


#
# Usage:
#   # Interactive
#   third_party/yggdrasil_decision_forests/port/python/tools/change_version.sh
#
#   # Non-interactive
#   ./tools/change_version.sh --version=0.17.0 --yes

set -euo pipefail

SRC="."

NEW_VERSION=""
ASSUME_YES=0

for arg in "$@"; do
  case "${arg}" in
    --version=*) NEW_VERSION="${arg#*=}" ;;
    --yes|-y) ASSUME_YES=1 ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      echo "Usage: $0 [--version=X.Y.Z] [--yes]" >&2
      exit 1
      ;;
  esac
done

confirmation() {
  if [[ "${ASSUME_YES}" = 1 ]]; then
    return
  fi
  read -p "Do you want to continue? (Ny) " -n 1 -r
  echo
  if [[ ! $REPLY = 'y' ]] ; then
      exit 1
  fi
}

VERSION_FILE="${SRC}/ydf/version.py"
CHANGELOG_FILE="${SRC}/CHANGELOG.md"

for f in "${VERSION_FILE}" "${CHANGELOG_FILE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: ${f} not found. Run this script from the root of the" >&2
    echo "google3 workspace (internally) or from port/python (OSS)." >&2
    exit 1
  fi
done

# Warning message.
echo "You are about to prepare the release of a new version of the YDF Python API"
confirmation

# Get the current version. Matches `version = "1.2.3"`, including the
# commented-out form used inside google3.
CURRENT_VERSION=$(sed -nE 's/^#? ?version = "([^"]+)"$/\1/p' "${VERSION_FILE}")
if [[ -z "${CURRENT_VERSION}" ]]; then
  echo "ERROR: Could not determine the current version from ${VERSION_FILE}" >&2
  exit 1
fi
echo "The current version is: ${CURRENT_VERSION}"

# Ask for the new version.
if [[ -z "${NEW_VERSION}" ]]; then
  echo
  echo "What is the new version? Follow the pattern <a>.<b>.<c>."
  read NEW_VERSION
fi

if [[ ! "${NEW_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+)?$ ]]; then
  echo "ERROR: '${NEW_VERSION}' is not a valid version." >&2
  echo "Expected <major>.<minor>.<patch> with an optional 'rcN' suffix." >&2
  exit 1
fi

if [[ "${NEW_VERSION}" = "${CURRENT_VERSION}" ]]; then
  echo "ERROR: The new version is identical to the current version." >&2
  exit 1
fi

echo "The new version is: ${NEW_VERSION}"

# Update the version number.
echo "Update ${VERSION_FILE}"
sed -i -e "s/version = \"${CURRENT_VERSION}\"/version = \"${NEW_VERSION}\"/" "${VERSION_FILE}"
if ! grep -q "version = \"${NEW_VERSION}\"" "${VERSION_FILE}"; then
  echo "ERROR: Failed to update the version in ${VERSION_FILE}." >&2
  exit 1
fi

# Stamp the changelog
echo "Update ${CHANGELOG_FILE}"
RELEASE_DATE=$(date +%Y-%m-%d)
if [[ "$(grep -m 1 '^## ' "${CHANGELOG_FILE}")" != "## Head" ]]; then
  echo "ERROR: Expected '## Head' to be the first section in ${CHANGELOG_FILE}." >&2
  exit 1
fi
sed -i -e "0,/^## Head$/s//## Head\n\n## ${NEW_VERSION} - ${RELEASE_DATE}/" "${CHANGELOG_FILE}"
if ! grep -q "^## ${NEW_VERSION} - ${RELEASE_DATE}$" "${CHANGELOG_FILE}"; then
  echo "ERROR: Failed to stamp the changelog at ${CHANGELOG_FILE}." >&2
  exit 1
fi

echo
echo "Version updated to ${NEW_VERSION}."
echo "Updated files:"
echo "  ${VERSION_FILE}"
echo "  ${CHANGELOG_FILE}"
echo
echo "Please review the changelog entry for ${NEW_VERSION} (add a summary of"
echo "the release, and the traditional release music) before sending the CL."
