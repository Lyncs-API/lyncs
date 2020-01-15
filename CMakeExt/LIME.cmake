include(ExternalProject)

if(NOT LIME_PATH)
  message(STATUS "No LIME_PATH given; we will download it.")
  set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
  set(LIME_PATH ${EXTERNAL_INSTALL_LOCATION})
  set(LIME_INSTALL ON)

  FILE(GLOB_RECURSE LIME_PATCHES "${CMAKE_SOURCE_DIR}/patches/c-lime/*.patch")
  ExternalProject_Add(c-lime
    GIT_REPOSITORY https://github.com/usqcd-software/c-lime
    GIT_TAG master
    PATCH_COMMAND git apply ${LIME_PATCHES}
    CONFIGURE_COMMAND /bin/sh -c "test -f Makefile || (./autogen.sh && ./configure CC=${CMAKE_C_COMPILER} --prefix=${EXTERNAL_INSTALL_LOCATION})"
    BUILD_COMMAND make -j && make -j install
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
  )
endif()


if(EXISTS "${LIME_PATH}/lib/liblime.so" AND EXISTS "${LIME_PATH}/include/lime.h")
  set(LIME_FOUND ON)
endif()

