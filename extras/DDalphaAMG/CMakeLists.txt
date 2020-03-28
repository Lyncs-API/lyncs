cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(DDalphaAMG C)

include(ExternalProject)

find_package(MPI REQUIRED)

if(NOT MPI_FOUND)
  message(ERROR "MPI not found. DDalphaAMG requires MPI")
endif()

set(COMPILER_FLAGS "${COMPILER_FLAGS} ${MPI_C_COMPILER_FLAGS}")
#find_package(OpenMP)
#if(OpenMP_FOUND)
#  set(COMPILER_FLAGS "${COMPILER_FLAGS} ${OpenMP_C_FLAGS} -DOPENMP")
#endif()

set(EXTERNAL_INSTALL_LOCATION "${CMAKE_BINARY_DIR}" CACHE STRING "Path where to install the library")

set(LIME_PATH "" CACHE STRING "Path where to look for clime")

if(EXISTS "${LIME_PATH}/lib/liblime.so" AND EXISTS "${LIME_PATH}/include/lime.h")		
  set(LIME_FOUND ON)
else()
  message(ERROR "C-LIME not found. DDalphaAMG requires it.")
endif()

FILE(GLOB_RECURSE PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")

ExternalProject_Add(DDalphaAMG
  GIT_REPOSITORY https://github.com/sbacchio/DDalphaAMG
  GIT_TAG master
  PATCH_COMMAND git apply ${PATCHES}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND make library MPI_C_COMPILER=${MPI_C_COMPILER} MPI_C_COMPILER_FLAGS=${COMPILER_FLAGS} LIMEDIR=${LIME_PATH} && make -j install PREFIX=${EXTERNAL_INSTALL_LOCATION}
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND ""
)
