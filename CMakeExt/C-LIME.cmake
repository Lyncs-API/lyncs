include(ExternalProject)

if(NOT C-LIME_LIBRARY_PATH)
  message(STATUS "No C-LIME_LIBRARY_PATH given; we will download it.")
  set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
  FILE(GLOB_RECURSE C-LIME_PATCHES "${CMAKE_SOURCE_DIR}/patches/c-lime/*.patch")
  ExternalProject_Add(c-lime
    GIT_REPOSITORY https://github.com/usqcd-software/c-lime
    GIT_TAG master
    PATCH_COMMAND git apply ${C-LIME_PATCHES}
    CONFIGURE_COMMAND ./autogen.sh && ./configure CC=${CMAKE_C_COMPILER} --prefix=${EXTERNAL_INSTALL_LOCATION}
    BUILD_COMMAND make -j
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND make -j install
  )
  set(C-LIME_LIBRARY_PATH ${EXTERNAL_INSTALL_LOCATION})
  set(C-LIME_INSTALL ON)
endif()

message(STATUS "Searching for tmLQCD library")
if(EXISTS "${C-LIME_LIBRARY_PATH}/lib/libhmc.so" AND EXISTS "${C-LIME_LIBRARY_PATH}/include/hmc.h")
  set(C-LIME_FOUND ON)
endif()

