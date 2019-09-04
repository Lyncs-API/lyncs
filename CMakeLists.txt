cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_RULE_MESSAGES OFF)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 11)

enable_language(C)
enable_language(CXX)

project(LinCs)

set(LINCS_VERSION_MAJOR 0 CACHE STRING "LINCS major version number.")
set(LINCS_VERSION_MINOR 0 CACHE STRING "LINCS minor version number.")
set(LINCS_VERSION_PATCH 0 CACHE STRING "LINCS patch version number.")
mark_as_advanced(
  LINCS_VERSION_MAJOR
  LINCS_VERSION_MINOR
  LINCS_VERSION_PATCH)
set(LINCS_VERSION
    "${LINCS_VERSION_MAJOR}.${LINCS_VERSION_MINOR}.${LINCS_VERSION_PATCH}"
    CACHE STRING INTERNAL FORCE)

## Install path
if (INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX ${INSTALL_PREFIX})
else()
  set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/opt/")
endif()

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

## Build options
option(BUILD_TESTS
       "Whether tests should be built" on)
option(INSTALL_TESTS
       "Whether tests should be installed" off)
option(BUILD_DOCS
       "Whether doxygen documentation should be installed" off)
option(BUILD_EXAMPLES
       "Specify whether to build examples" on)
option(BUILD_SHARED_LIBS
       "Specify whether libraries should be built as shared objects" off)

include(${CMAKE_SOURCE_DIR}/CMakeExt/MessageColor.cmake)
include(${CMAKE_SOURCE_DIR}/CMakeExt/GTest.cmake)

message(EMPH "LinCs version ${LINCS_VERSION}")
message(EMPH "-----------------------------------------------------------")
message(EMPH "Install prefix:           (INSTALL_PREFIX)                 "
        ${CMAKE_INSTALL_PREFIX})
message(EMPH "Build type:               (CMAKE_BUILD_TYPE)               "
        ${CMAKE_BUILD_TYPE})
message(EMPH "Build shared libraries:   (BUILD_SHARED_LIBS)              "
        ${BUILD_SHARED_LIBS})

message(INFO "Build tests:              (BUILD_TESTS)                    "
        ${BUILD_TESTS})
message(INFO "C   compiler id:          ${CMAKE_C_COMPILER_ID}")
message(INFO "C++ compiler id:          ${CMAKE_CXX_COMPILER_ID}")

if (BUILD_TESTS)
  if (GTEST_FOUND)
    message(INFO "googletest                enabled")
  else()
    message(NOTE "googletest                to install")
  endif()
else()
  message(INFO "googletest                disabled")
endif()
