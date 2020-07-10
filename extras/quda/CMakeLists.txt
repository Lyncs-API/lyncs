cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(QUDA CXX)

include(ExternalProject)

set(EXTERNAL_INSTALL_LOCATION "${CMAKE_BINARY_DIR}" CACHE STRING "Path where to install the library")

FILE(GLOB_RECURSE PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")

MESSAGE(STATUS ${QUDA_CMAKE_ARGS})

ExternalProject_Add(quda
  GIT_REPOSITORY https://github.com/lattice/quda
  GIT_TAG v1.0.0
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} ${QUDA_CMAKE_ARGS}
  PATCH_COMMAND git apply ${PATCHES}
)


