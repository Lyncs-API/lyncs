cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

project(tmLQCD C)

include(ExternalProject)

FILE(GLOB_RECURSE PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")

enable_language(Fortran)

ExternalProject_Add(tmLQCD
  GIT_REPOSITORY https://github.com/etmc/tmLQCD
  GIT_TAG master
  PATCH_COMMAND git apply ${PATCHES}
  CONFIGURE_COMMAND /bin/sh -c "autoconf && ./configure CC=${CMAKE_C_COMPILER} F77=${CMAKE_Fortran_COMPILER} CFLAGS=\"-std=gnu99 -O3 -ffast-math -fopenmp\" LIBS=-lopenblas --prefix=${EXTERNAL_INSTALL_LOCATION} --enable-mpi=no --with-limedir=${LIME_PATH}"
  BUILD_COMMAND make -j && make -j install
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND ""
)

set(TMLQCD_PATH ${EXTERNAL_INSTALL_LOCATION})
set(TMLQCD_INSTALL ON)


