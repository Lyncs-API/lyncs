#!/bin/sh

BUILD_DIR=./build

FORCE_BUILD=false
if [ "$1" = "-f" ]; then
  FORCE_BUILD=true
fi

await_confirm() {
  if ! $FORCE_BUILD; then
    echo ""
    echo "   To build using these settings, hit ENTER"
    read confirm
  fi
}

exit_message() {
  echo "--------------------------------------------------------"
  echo "Done. To install LinCs, run    make install    in $BUILD_DIR"
}

# Configure with default release build settings:
mkdir -p $BUILD_DIR
rm -Rf $BUILD_DIR/*
(cd $BUILD_DIR && cmake -DCMAKE_BUILD_TYPE=Release \
                        -DBUILD_SHARED_LIBS=ON \
                        -DINSTALL_PREFIX=$HOME/opt/LinCs/ \
                        \
                        -DBUILD_EXAMPLES=ON \
                        -DBUILD_TESTS=ON \
                        -DBUILD_DOCS=ON \
                        ../ && \
 await_confirm && \
 make -j 4) && \
exit_message

