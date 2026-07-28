#!/bin/bash
if command -v git &> /dev/null; then
  BUILD_REV=$(git describe --abbrev=7 --dirty --always --tags 2> /dev/null || echo 'unknown-out-of-tree')
else
  BUILD_REV='unknown'
fi
echo $BUILD_REV
