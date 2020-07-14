cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

project(c-lime C)

include(ExternalProject)

set(EXTERNAL_INSTALL_LOCATION "${CMAKE_BINARY_DIR}" CACHE STRING "Path where to install the library")

FILE(GLOB_RECURSE PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
ExternalProject_Add(c-lime
  GIT_REPOSITORY https://github.com/usqcd-software/c-lime
  GIT_TAG master
  PATCH_COMMAND git apply ${PATCHES}
  CONFIGURE_COMMAND /bin/sh -c "test -f Makefile || (./autogen.sh && ./configure CC=${CMAKE_C_COMPILER} --prefix=${EXTERNAL_INSTALL_LOCATION})"
  BUILD_COMMAND make -j && make -j install
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND ""
)

