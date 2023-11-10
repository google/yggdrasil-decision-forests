# Update the version of the addon.

confirmation () {
  read -p "Do you want to continue? (Ny) " -n 1 -r
  echo
  if [[ ! $REPLY = 'y' ]] ; then
      exit 1
  fi
}

# Warning message.
echo "You are about to release a new version of the YDF Python API"
confirmation

SRC="third_party/yggdrasil_decision_forests/port/python"

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

echo
echo "Check that the following files have been updated correctly:"
echo ${SRC}/config/setup.py
echo ${SRC}/ydf/version.py
echo "Update the changelog:"
echo ${SRC}/CHANGELOG.md
