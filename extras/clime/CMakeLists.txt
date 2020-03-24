cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

option(ENABLE_CLIME "Specify whether to enable the c-lime library" off)

project(c-lime)

if(NOT ENABLE_CLIME)
  message(STATUS "clime\t\tdisabled")
  return()
endif()

include(ExternalProject)


set(CLIME_PATH "" CACHE STRING "Path to the c-lime installation dir")
set(EXTERNAL_INSTALL_LOCATION "${CMAKE_BINARY_DIR}" CACHE STRING "Path where to install the library")

if(NOT CLIME_PATH)
  message(STATUS "No CLIME_PATH given; we will download it.")
  set(CLIME_PATH ${EXTERNAL_INSTALL_LOCATION})
  set(CLIME_INSTALL ON)

  FILE(GLOB_RECURSE CLIME_PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
  ExternalProject_Add(c-lime
    GIT_REPOSITORY https://github.com/usqcd-software/c-lime
    GIT_TAG master
    PATCH_COMMAND git apply ${CLIME_PATCHES}
    CONFIGURE_COMMAND /bin/sh -c "test -f Makefile || (./autogen.sh && ./configure CC=${CMAKE_C_COMPILER} --prefix=${EXTERNAL_INSTALL_LOCATION})"
    BUILD_COMMAND make -j && make -j install
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
  )
endif()


if(EXISTS "${CLIME_PATH}/lib/liblime.so" AND EXISTS "${CLIME_PATH}/include/lime.h")
  set(CLIME_FOUND ON)
endif()

