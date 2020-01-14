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

done_message() {
  echo "--------------------------------------------------------"
  echo "Done. Lyncs installed succesfully."
  if ! $FORCE_BUILD; then
      echo ""
      echo "   To run the tests, hit ENTER"
      read confirm
  fi
}

# Configure with default release build settings:
mkdir -p $BUILD_DIR
rm -Rf $BUILD_DIR/*
(cd $BUILD_DIR && cmake -DCMAKE_BUILD_TYPE=DEVELOP \
                        -DINSTALL_PREFIX=$HOME/.local/ \
                        \
                        -DBUILD_EXAMPLES=ON \
                        -DBUILD_DOCS=ON \
			\
			-DENABLE_QUDA=OFF \
                        ../ && \
     await_confirm && \
     make -j &&
     make -j install) && \
	 python3 setup.py develop --user &&
	 done_message && \
	 (cd $BUILD_DIR && make -j CTEST_OUTPUT_ON_FAILURE=1 test)
