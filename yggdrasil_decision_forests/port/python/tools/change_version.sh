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



# Update the version of the addon.

confirmation () {
  read -p "Do you want to continue? (Ny) " -n 1 -r
  echo
  if [[ ! $REPLY = 'y' ]] ; then
      exit 1
  fi
}

# Warning message.
echo "You are about to prepare the release of a new version of the YDF Python API"
confirmation

SRC=""


# Get version
CURRENT_VERSION=$(cat ${SRC}/ydf/version.py | grep -o "[\.0-9]\+")
echo "The current version is: ${CURRENT_VERSION}"

# Ask for new version
echo
echo "What is the new version? Follow the pattern <a>.<b>.<c>."
read NEW_VERSION
echo "The new version is: ${NEW_VERSION}"

# Update version number
echo "Update version"
sed -i -e "s/_VERSION = \"[_rcv\.0-9]\+\"/_VERSION = \"${NEW_VERSION}\"/" ${SRC}/config/setup.py
sed -i -e "s/version = \"[_rcv\.0-9]\+\"/version = \"${NEW_VERSION}\"/" ${SRC}/ydf/version.py
# TODO: Fail if the version cannot be updated
# TODO: Update the version in the release_windows.bat script.

echo
echo "Check that the following files have been updated correctly:"
echo ${SRC}/config/setup.py
echo ${SRC}/ydf/version.py
echo "Update the changelog:"
echo ${SRC}/CHANGELOG.md
echo ${SRC}/tools/release_windows.bat
