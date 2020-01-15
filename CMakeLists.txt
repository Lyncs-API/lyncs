cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_RULE_MESSAGES OFF)
set(CMAKE_VERBOSE_MAKEFILE OFF)

project(Lyncs)

set(LYNCS_VERSION_MAJOR 0 CACHE STRING "LYNCS major version number.")
set(LYNCS_VERSION_MINOR 0 CACHE STRING "LYNCS minor version number.")
set(LYNCS_VERSION_PATCH 0 CACHE STRING "LYNCS patch version number.")
mark_as_advanced(
  LYNCS_VERSION_MAJOR
  LYNCS_VERSION_MINOR
  LYNCS_VERSION_PATCH)
set(LYNCS_VERSION
    "${LYNCS_VERSION_MAJOR}.${LYNCS_VERSION_MINOR}.${LYNCS_VERSION_PATCH}"
    CACHE STRING INTERNAL FORCE)

## Install path
if (INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX ${INSTALL_PREFIX})
else()
  set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/opt/")
endif()


## Build type
set(_BUILD_TYPE RELEASE DEVELOP)
if (NOT CMAKE_BUILD_TYPE IN_LIST _BUILD_TYPE)
   unset(CMAKE_BUILD_TYPE CACHE)
endif()

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE CACHE STRING "Choose type of build. Options are RELEASE, DEVELOP")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${_BUILD_TYPE})
endif()


## Build options
option(BUILD_DOCS
       "Whether doxygen documentation should be installed" off)
option(BUILD_EXAMPLES
       "Specify whether to build examples" on)
option(ENABLE_MPI
       "Specify whether to build using MPI" on)

if(ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()

include(${CMAKE_SOURCE_DIR}/CMakeExt/MessageColor.cmake)
include(${CMAKE_SOURCE_DIR}/CMakeExt/pytest.cmake)

set(_EXT_LIBS QUDA TMLQCD C-LIME DDALPHAAMG)
foreach(lib ${_EXT_LIBS})
  option(ENABLE_${lib}
         "Specify whether to enable ${LIB} library" off)
  if(ENABLE_${lib})
    include(${CMAKE_SOURCE_DIR}/CMakeExt/${lib}.cmake)
  endif()
endforeach(lib)

include(${CMAKE_SOURCE_DIR}/CMakeExt/lyncs_config.cmake)

message(EMPH "Lyncs version ${LYNCS_VERSION}")
message(EMPH "-----------------------------------------------------------")
message(EMPH "Install prefix:\t(INSTALL_PREFIX)\t"
        ${CMAKE_INSTALL_PREFIX})
message(EMPH "Build type:\t\t(CMAKE_BUILD_TYPE)\t"
        ${CMAKE_BUILD_TYPE})

message(INFO "C   compiler id:\t${CMAKE_C_COMPILER_ID}")
message(INFO "C++ compiler id:\t${CMAKE_CXX_COMPILER_ID}")

foreach(lib ${_EXT_LIBS})
  # TODO: improve formatting
  if (ENABLE_${lib})
    if (${lib}_FOUND)
      message(EMPH "${lib}\t\tenabled")
    elseif(${lib}_INSTALL)
      message(NOTE "${lib}\t\tto install")
    else()
      message(NOTE "${lib}\t\tnot found in ${lib}_LIBRARY_PATH=${${lib}_LIBRARY_PATH}")
    endif()
  else()
    message(INFO "${lib}\t\tdisabled")
  endif()
endforeach(lib)

